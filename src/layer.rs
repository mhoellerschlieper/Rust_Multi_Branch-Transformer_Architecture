// layer.rs
// Description: Model layers and core LLM implementation (forward, backward, train, predict).
//              Adds MTB (multi branch transformer) support via a parallel block group layer.
// History:
// - 2026-02-01: Consolidate project into 6 files: main, layer, train, math, tokenizer, utils.
// - 2026-02-01: Add checkpoint save and load for model parameters and tokenizer.
// Author: Marcus Schlieper


use std::cmp::Ordering;

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};
use rand_distr::weighted::WeightedIndex;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::any::Any;

use crate::{EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN};
use crate::math;
use crate::tokenizer::{BpeTokenizer, BpeTokenizerCheckpoint, S_EOS};
use crate::utils;
use std::ops::AddAssign;
use serde::{Serialize, Deserialize};

pub const DEFAULT_RESIDUAL_DROPOUT_P: f32 = 0.001;

#[derive(Clone, Debug)]
pub struct Vocab {
    pub encode: std::collections::HashMap<String, usize>,
    pub decode: std::collections::HashMap<usize, String>,
    pub words: Vec<String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new(Self::default_words())
    }
}

impl Vocab {
    pub fn new(v_words: Vec<&str>) -> Self {
        let mut m_encode: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut m_decode: std::collections::HashMap<usize, String> = std::collections::HashMap::new();

        for (i_id, s_word) in v_words.iter().enumerate() {
            m_encode.insert((*s_word).to_string(), i_id);
            m_decode.insert(i_id, (*s_word).to_string());
        }

        Self {
            encode: m_encode,
            decode: m_decode,
            words: v_words.iter().map(|w| (*w).to_string()).collect(),
        }
    }

    pub fn encode(&self, s_word: &str) -> Option<usize> {
        self.encode.get(s_word).copied()
    }

    pub fn decode(&self, i_token_id: usize) -> Option<&String> {
        self.decode.get(&i_token_id)
    }

    pub fn default_words() -> Vec<&'static str> {
        // Stable, explicit special tokens. Must match tokenizer.rs constants.
        vec!["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
    }
}

pub trait Layer {
    fn layer_type(&self) -> &str;

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32>;

    fn parameters(&self) -> usize;

    fn get_parameters_flat(&self) -> Vec<f32>;

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String>;

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct AdamW {
    d_beta1: f32,
    d_beta2: f32,
    d_eps: f32,
    d_weight_decay: f32,
    i_t: usize,
    m_m: Array2<f32>,
    m_v: Array2<f32>,
}

impl AdamW {
    pub fn new(t_shape: (usize, usize), d_weight_decay: f32) -> Self {
        let d_wd = if d_weight_decay.is_finite() && d_weight_decay >= 0.0 {
            d_weight_decay
        } else {
            0.0
        };

        Self {
            d_beta1: 0.9,
            d_beta2: 0.999,
            d_eps: 1e-8,
            d_weight_decay: d_wd,
            i_t: 0,
            m_m: Array2::zeros(t_shape),
            m_v: Array2::zeros(t_shape),
        }
    }

    pub fn set_weight_decay(&mut self, d_weight_decay: f32) {
        if d_weight_decay.is_finite() && d_weight_decay >= 0.0 {
            self.d_weight_decay = d_weight_decay;
        }
    }

    // AdamW step:
    // - Decoupled weight decay: params = params - lr * wd * params
    // - Adam moments update uses gradients only (not including wd term)
    pub fn step(&mut self, a_params: &mut Array2<f32>, a_grads: &Array2<f32>, d_lr: f32) {
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return;
        }
        if a_params.raw_dim() != a_grads.raw_dim() {
            return;
        }

        self.i_t = self.i_t.saturating_add(1);

        // Decoupled weight decay (safe no-op if wd is zero).
        if self.d_weight_decay > 0.0 {
            let d_decay = d_lr * self.d_weight_decay;
            if d_decay.is_finite() && d_decay > 0.0 {
                *a_params = &*a_params - &(d_decay * &*a_params);
            }
        }

        // m = b1*m + (1-b1)*g
        self.m_m = &self.m_m * self.d_beta1 + a_grads * (1.0 - self.d_beta1);

        // v = b2*v + (1-b2)*g^2
        let a_grads_sq = a_grads.mapv(|x| x * x);
        self.m_v = &self.m_v * self.d_beta2 + a_grads_sq * (1.0 - self.d_beta2);

        let d_t = self.i_t as f32;
        let d_b1t = self.d_beta1.powf(d_t);
        let d_b2t = self.d_beta2.powf(d_t);

        let a_m_hat = self
            .m_m
            .mapv(|x| x / (1.0 - d_b1t).max(1e-12));
        let a_v_hat = self
            .m_v
            .mapv(|x| x / (1.0 - d_b2t).max(1e-12));

        let a_denom = a_v_hat.mapv(|x| x.sqrt() + self.d_eps);
        let a_update = a_m_hat / a_denom;

        *a_params = &*a_params - &(d_lr * a_update);
    }
}


pub struct Embeddings {
    vocab: crate::layer::Vocab,
    w_embed: Array2<f32>,
    cached_ids: Option<Vec<usize>>,
    optimizer: AdamW,
}

impl Embeddings {
    pub fn new(vocab: crate::layer::Vocab) -> Self {
        let i_vocab = vocab.words.len();
        let mut rng = rand::rng();

        let std = (2.0 / (i_vocab as f32).max(1.0)).sqrt();
        let normal = rand_distr::Normal::new(0.0, std).unwrap();

        Self {
            vocab,
            w_embed: Array2::from_shape_fn((i_vocab, crate::EMBEDDING_DIM), |_| {
                normal.sample(&mut rng)
            }),
            cached_ids: None,
            optimizer: AdamW::new((i_vocab, crate::EMBEDDING_DIM), 0.01),
        }
    }
}

impl Layer for Embeddings {
    fn layer_type(&self) -> &str {
        "Embeddings"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        // a_input: [1, seq_len] containing token ids as f32
        let (i_rows, i_cols) = a_input.dim();
        if i_rows != 1 || i_cols == 0 {
            return Array2::zeros((0, EMBEDDING_DIM));
        }

        let mut v_ids: Vec<usize> = Vec::with_capacity(i_cols);
        for j in 0..i_cols {
            let d_val = a_input[[0, j]];
            if !d_val.is_finite() {
                v_ids.push(0);
                continue;
            }
            let i_id = d_val.max(0.0) as usize;
            v_ids.push(i_id.min(self.vocab.words.len().saturating_sub(1)));
        }
        self.cached_ids = Some(v_ids.clone());

        let mut a_out = Array2::zeros((i_cols, EMBEDDING_DIM));
        for (i_pos, &i_id) in v_ids.iter().enumerate() {
            if i_id < self.w_embed.nrows() {
                a_out.row_mut(i_pos).assign(&self.w_embed.row(i_id));
            }
        }
        a_out
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let v_ids = match self.cached_ids.as_ref() {
            Some(v) => v,
            None => return Array2::zeros((1, 0)),
        };

        // Accumulate gradients into embedding matrix.
        let mut a_grad_w = Array2::zeros(self.w_embed.raw_dim());
        for (i_pos, &i_id) in v_ids.iter().enumerate() {
            if i_pos < a_grads.nrows() && i_id < a_grad_w.nrows() {
                let a_row = a_grads.row(i_pos);
                a_grad_w.row_mut(i_id).add_assign(&a_row);
            }
        }

        self.optimizer.step(&mut self.w_embed, &a_grad_w, d_lr);

        // Gradient wrt token ids is undefined, return shape [1, seq_len] zeros.
        Array2::zeros((1, v_ids.len()))
    }

    fn parameters(&self) -> usize {
        self.w_embed.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        self.w_embed.iter().copied().collect()
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w_embed.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_embeddings".to_string());
        }
        for (i, d) in v_params.iter().take(i_needed).enumerate() {
            self.w_embed.as_slice_mut().unwrap()[i] = *d;
        }
        Ok(i_needed)
    }
}

pub struct LayerNorm {
    epsilon: f32,
    gamma: Array2<f32>,
    beta: Array2<f32>,

    cached_input: Option<Array2<f32>>,
    cached_mean: Option<Array2<f32>>,
    cached_std: Option<Array2<f32>>,

    optimizer_gamma: AdamW,
    optimizer_beta: AdamW,
}

impl LayerNorm {
    pub fn new(i_embedding_dim: usize) -> Self {
        Self {
            epsilon: 1e-5,
            gamma: Array2::ones((1, i_embedding_dim)),
            beta: Array2::zeros((1, i_embedding_dim)),
            cached_input: None,
            cached_mean: None,
            cached_std: None,
            optimizer_gamma: AdamW::new((1, i_embedding_dim), 0.0),
            optimizer_beta: AdamW::new((1, i_embedding_dim), 0.0),
        }
    }

    pub fn normalize(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }

        let a_mean = a_input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let a_std = a_input.std_axis(Axis(1), 0.0).insert_axis(Axis(1));

        self.cached_input = Some(a_input.clone());
        self.cached_mean = Some(a_mean.clone());
        self.cached_std = Some(a_std.clone());

        let a_normed = (a_input - &a_mean) / (&a_std + self.epsilon);
        &self.gamma * &a_normed + &self.beta
    }
}

impl Layer for LayerNorm {
    fn layer_type(&self) -> &str {
        "LayerNorm"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        self.normalize(a_input)
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = match self.cached_input.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_mean = self.cached_mean.as_ref().unwrap();
        let a_std = self.cached_std.as_ref().unwrap();

        let a_normalized = (a_input - a_mean) / (a_std + self.epsilon);
        let d_n_features = a_input.ncols() as f32;

        let a_grad_gamma = (&a_normalized * a_grads).sum_axis(Axis(0)).insert_axis(Axis(0));
        let a_grad_beta = a_grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        let a_grad_normalized = &self.gamma * a_grads;

        let a_variance = a_std * a_std + self.epsilon;
        let a_grad_var = (&a_grad_normalized * &a_normalized)
            .sum_axis(Axis(1))
            .insert_axis(Axis(1))
            * (-0.5)
            / a_variance
                .mapv(|x| x * x.sqrt())
                .mapv(|x| x.max(1e-12));

        let a_grad_mean = a_grad_normalized.sum_axis(Axis(1)).insert_axis(Axis(1)) * (-1.0)
            / (a_std + self.epsilon)
            + &a_grad_var
                * (a_input - a_mean).sum_axis(Axis(1)).insert_axis(Axis(1))
                * (-2.0)
                / d_n_features.max(1.0);

        let a_grad_input = &a_grad_normalized / (a_std + self.epsilon)
            + &a_grad_var * 2.0 * (a_input - a_mean) / d_n_features.max(1.0)
            + &a_grad_mean / d_n_features.max(1.0);

        self.optimizer_gamma.step(&mut self.gamma, &a_grad_gamma, d_lr);
        self.optimizer_beta.step(&mut self.beta, &a_grad_beta, d_lr);

        a_grad_input
    }

    fn parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::with_capacity(self.gamma.len() + self.beta.len());
        v.extend(self.gamma.iter().copied());
        v.extend(self.beta.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.gamma.len() + self.beta.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_layer_norm".to_string());
        }

        let i_gamma = self.gamma.len();
        let v_gamma = &v_params[..i_gamma];
        let v_beta = &v_params[i_gamma..i_needed];

        for (i, d) in v_gamma.iter().enumerate() {
            self.gamma.as_slice_mut().unwrap()[i] = *d;
        }
        for (i, d) in v_beta.iter().enumerate() {
            self.beta.as_slice_mut().unwrap()[i] = *d;
        }

        Ok(i_needed)
    }
}

// ---------------------------
// RMSNorm
// ---------------------------
// RMSNorm(x) = x / rms(x) * gamma
// where rms(x) = sqrt(mean(x^2) + eps)
// - normalization is per token (row-wise), over embedding_dim (columns)
// - no beta term and no mean subtraction
pub struct RmsNorm {
    epsilon: f32,
    gamma: Array2<f32>, // [1, embedding_dim]

    // caches for backward
    cached_input: Option<Array2<f32>>,
    cached_rms: Option<Array2<f32>>, // [seq, 1]
    cached_x_hat: Option<Array2<f32>>, // [seq, emb]

    optimizer_gamma: AdamW,
}

impl RmsNorm {
    pub fn new(i_embedding_dim: usize) -> Self {
        if i_embedding_dim == 0 {
            panic!("rmsnorm_embedding_dim_must_be_positive");
        }

        Self {
            epsilon: 1e-5,
            gamma: Array2::ones((1, i_embedding_dim)),
            cached_input: None,
            cached_rms: None,
            cached_x_hat: None,
            optimizer_gamma: AdamW::new((1, i_embedding_dim), 0.0),
        }
    }

    fn normalize(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }

        let i_emb = a_input.ncols() as f32;
        let d_inv = 1.0 / i_emb.max(1.0);

        // rms: [seq, 1]
        let a_mean_sq = a_input
            .mapv(|x| x * x)
            .sum_axis(Axis(1))
            .insert_axis(Axis(1))
            .mapv(|s| s * d_inv);

        let a_rms = a_mean_sq.mapv(|m| (m + self.epsilon).sqrt().max(1e-12));
        let a_x_hat = a_input / &a_rms;

        self.cached_input = Some(a_input.clone());
        self.cached_rms = Some(a_rms.clone());
        self.cached_x_hat = Some(a_x_hat.clone());

        // Broadcast gamma: [1, emb] over rows
        &a_x_hat * &self.gamma
    }
}

impl Layer for RmsNorm {
    fn layer_type(&self) -> &str {
        "RmsNorm"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        self.normalize(a_input)
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        // If caches are missing, return passthrough gradients (safe fallback).
        let a_input = match self.cached_input.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_rms = match self.cached_rms.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_x_hat = match self.cached_x_hat.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };

        if a_input.raw_dim() != a_grads.raw_dim() {
            return a_grads.clone();
        }

        let i_seq = a_input.nrows();
        let i_emb = a_input.ncols();

        if i_seq == 0 || i_emb == 0 {
            return a_grads.clone();
        }

        // y = x_hat * gamma
        // dL/dgamma = sum_over_rows(dL/dy * x_hat)
        let a_grad_gamma = (a_grads * a_x_hat)
            .sum_axis(Axis(0))
            .insert_axis(Axis(0));

        // dL/dx_hat = dL/dy * gamma
        let a_grad_x_hat = a_grads * &self.gamma;

        // x_hat = x / rms
        // rms = sqrt(mean(x^2) + eps)
        // For each row:
        // dx = (1/rms) * d_x_hat - x * (sum(d_x_hat * x) / (N * rms^3))
        let d_n = i_emb as f32;
        let mut a_grad_x = Array2::<f32>::zeros(a_input.raw_dim());

        for i in 0..i_seq {
            let d_r = a_rms[[i, 0]].max(1e-12);
            let d_inv_r = 1.0 / d_r;
            let d_inv_r3 = 1.0 / (d_r * d_r * d_r).max(1e-12);

            // sum_j d_x_hat[j] * x[j]
            let mut d_dot: f32 = 0.0;
            for j in 0..i_emb {
                let d_dxhat = a_grad_x_hat[[i, j]];
                let d_x = a_input[[i, j]];
                d_dot += d_dxhat * d_x;
            }

            let d_scale2 = d_dot / d_n.max(1.0);

            for j in 0..i_emb {
                let d_dxhat = a_grad_x_hat[[i, j]];
                let d_x = a_input[[i, j]];
                let d_val = d_dxhat * d_inv_r - d_x * d_scale2 * d_inv_r3;

                a_grad_x[[i, j]] = if d_val.is_finite() { d_val } else { 0.0 };
            }
        }

        // Update gamma (weight decay typically 0.0 for norm params).
        self.optimizer_gamma.step(&mut self.gamma, &a_grad_gamma, d_lr);

        a_grad_x
    }

    fn parameters(&self) -> usize {
        self.gamma.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        self.gamma.iter().copied().collect()
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.gamma.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_rms_norm".to_string());
        }

        // ndarray is contiguous here by construction
        let a_slice = self.gamma.as_slice_mut().ok_or_else(|| "rmsnorm_gamma_not_contiguous".to_string())?;
        for i in 0..i_needed {
            let d = v_params[i];
            a_slice[i] = if d.is_finite() { d } else { 0.0 };
        }

        Ok(i_needed)
    }
}
pub struct FeedForward {
    w1: Array2<f32>,
    b1: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,

    cached_input: Option<Array2<f32>>,
    cached_hidden_pre: Option<Array2<f32>>,
    cached_hidden_post: Option<Array2<f32>>,

    opt_w1: AdamW,
    opt_b1: AdamW,
    opt_w2: AdamW,
    opt_b2: AdamW,

    // residual dropout on FF branch before residual add.
    residual_dropout: Dropout,
}

impl FeedForward {
    pub fn new(i_embedding_dim: usize, i_hidden_dim: usize) -> Self {
        let mut rng = rand::rng();

        let std_w1 = (2.0 / (i_embedding_dim as f32).max(1.0)).sqrt();
        let std_w2 = (2.0 / (i_hidden_dim as f32).max(1.0)).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1).unwrap();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();

        Self {
            w1: Array2::from_shape_fn((i_embedding_dim, i_hidden_dim), |_| normal_w1.sample(&mut rng)),
            b1: Array2::zeros((1, i_hidden_dim)),
            w2: Array2::from_shape_fn((i_hidden_dim, i_embedding_dim), |_| normal_w2.sample(&mut rng)),
            b2: Array2::zeros((1, i_embedding_dim)),
            cached_input: None,
            cached_hidden_pre: None,
            cached_hidden_post: None,
            opt_w1: AdamW::new((i_embedding_dim, i_hidden_dim), 0.0),
            opt_b1: AdamW::new((1, i_hidden_dim), 0.0),
            opt_w2: AdamW::new((i_hidden_dim, i_embedding_dim), 0.0),
            opt_b2: AdamW::new((1, i_embedding_dim), 0.0),

            residual_dropout: Dropout::new(DEFAULT_RESIDUAL_DROPOUT_P, 7777),
        }
    }

    pub fn set_training(&mut self, b_training: bool) {
        self.residual_dropout.set_training(b_training);
    }

    pub fn set_residual_dropout_p(&mut self, d_p: f32) {
        self.residual_dropout.set_p(d_p);
    }

    pub fn reseed_dropout(&mut self, u64_seed: u64) {
        self.residual_dropout.reseed(u64_seed);
    }


    fn relu(a: &Array2<f32>) -> Array2<f32> {
        a.mapv(|x| x.max(0.0))
    }
}

impl Layer for FeedForward {
    fn layer_type(&self) -> &str {
        "FeedForward"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        let a_hidden_pre = a_input.dot(&self.w1) + &self.b1;
        let a_hidden_post = Self::relu(&a_hidden_pre);
        let a_out = a_hidden_post.dot(&self.w2) + &self.b2;

        self.cached_input = Some(a_input.clone());
        self.cached_hidden_pre = Some(a_hidden_pre);
        self.cached_hidden_post = Some(a_hidden_post);

        // residual dropout on FF branch only.
        let a_ff_dropped = self.residual_dropout.apply(&a_out);

        a_ff_dropped + a_input
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = self.cached_input.as_ref().ok_or("cache_missing_ff_input".to_string()).unwrap();
        let a_hidden_pre = self.cached_hidden_pre.as_ref().unwrap();
        let a_hidden_post = self.cached_hidden_post.as_ref().unwrap();

        let a_grad_w2 = a_hidden_post.t().dot(a_grads);
        let a_grad_b2 = a_grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        let a_grad_hidden_post = a_grads.dot(&self.w2.t());
        let a_relu_grad = a_hidden_pre.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let a_grad_hidden_pre = a_grad_hidden_post * a_relu_grad;

        let a_grad_w1 = a_input.t().dot(&a_grad_hidden_pre);
        let a_grad_b1 = a_grad_hidden_pre.sum_axis(Axis(0)).insert_axis(Axis(0));

        let a_grad_input_ff = a_grad_hidden_pre.dot(&self.w1.t());

        // Residual gradient.
        let a_grad_input = a_grad_input_ff + a_grads;

        self.opt_w2.step(&mut self.w2, &a_grad_w2, d_lr);
        self.opt_b2.step(&mut self.b2, &a_grad_b2, d_lr);
        self.opt_w1.step(&mut self.w1, &a_grad_w1, d_lr);
        self.opt_b1.step(&mut self.b1, &a_grad_b1, d_lr);

        a_grad_input
    }

    fn parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.w1.iter().copied());
        v.extend(self.b1.iter().copied());
        v.extend(self.w2.iter().copied());
        v.extend(self.b2.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_feed_forward".to_string());
        }

        let mut i_pos: usize = 0;

        for i in 0..self.w1.len() {
            self.w1.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.b1.len() {
            self.b1.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w2.len() {
            self.w2.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.b2.len() {
            self.b2.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

pub struct SelfAttention {
    embedding_dim: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    cached_input: Option<Array2<f32>>,
    opt_w_q: AdamW,
    opt_w_k: AdamW,
    opt_w_v: AdamW,
}

impl SelfAttention {
    pub fn new(i_embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let std = (2.0 / (i_embedding_dim as f32).max(1.0)).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        Self {
            embedding_dim: i_embedding_dim,
            w_q: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            cached_input: None,
            opt_w_q: AdamW::new((i_embedding_dim, i_embedding_dim), 0.0),
            opt_w_k: AdamW::new((i_embedding_dim, i_embedding_dim), 0.0),
            opt_w_v: AdamW::new((i_embedding_dim, i_embedding_dim), 0.0),
        }
    }

    fn softmax(a_scores: &Array2<f32>) -> Array2<f32> {
        math::softmax_rows(a_scores)
    }

    fn compute_qkv(&self, a_input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        (a_input.dot(&self.w_q), a_input.dot(&self.w_k), a_input.dot(&self.w_v))
    }

    fn attention(&self, a_q: &Array2<f32>, a_k: &Array2<f32>, a_v: &Array2<f32>) -> Array2<f32> {
        let d_scale = (self.embedding_dim as f32).sqrt().max(1e-12);
        let mut a_scores = a_q.dot(&a_k.t()) / d_scale;

        // Causal mask.
        let i_seq_len = a_scores.nrows();
        for i in 0..i_seq_len {
            for j in (i + 1)..i_seq_len {
                a_scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let a_weights = Self::softmax(&a_scores);
        a_weights.dot(a_v)
    }

    fn softmax_backward(a_softmax: &Array2<f32>, a_grad_out: &Array2<f32>) -> Array2<f32> {
        // Row-wise Jacobian-vector product for softmax.
        let mut a_grad_in = a_softmax.clone();
        for i in 0..a_softmax.nrows() {
            let a_row = a_softmax.row(i);
            let a_grow = a_grad_out.row(i);

            let d_dot: f32 = a_row.iter().zip(a_grow.iter()).map(|(&y, &dy)| y * dy).sum();
            for j in 0..a_softmax.ncols() {
                a_grad_in[[i, j]] = a_softmax[[i, j]] * (a_grad_out[[i, j]] - d_dot);
            }
        }
        a_grad_in
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "SelfAttention"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(a_input.clone());
        let (a_q, a_k, a_v) = self.compute_qkv(a_input);
        let a_attn = self.attention(&a_q, &a_k, &a_v);
        a_attn + a_input
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = self.cached_input.as_ref().expect("forward must be run first");
        let (a_q, a_k, a_v) = self.compute_qkv(a_input);

        let d_scale = (self.embedding_dim as f32).sqrt().max(1e-12);
        let mut a_scores = a_q.dot(&a_k.t()) / d_scale;

        let i_seq_len = a_scores.nrows();
        for i in 0..i_seq_len {
            for j in (i + 1)..i_seq_len {
                a_scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let a_weights = Self::softmax(&a_scores);

        let a_grad_weights = a_grads.dot(&a_v.t());
        let a_grad_v = a_weights.t().dot(a_grads);

        let a_grad_scores = Self::softmax_backward(&a_weights, &a_grad_weights);

        let a_grad_q = a_grad_scores.dot(&a_k) / d_scale;
        let a_grad_k = a_grad_scores.t().dot(&a_q) / d_scale;

        let a_grad_w_q = a_input.t().dot(&a_grad_q);
        let a_grad_w_k = a_input.t().dot(&a_grad_k);
        let a_grad_w_v = a_input.t().dot(&a_grad_v);

        let a_grad_in_attn =
            a_grad_q.dot(&self.w_q.t()) + a_grad_k.dot(&self.w_k.t()) + a_grad_v.dot(&self.w_v.t());

        let a_grad_in = a_grad_in_attn + a_grads;

        self.opt_w_q.step(&mut self.w_q, &a_grad_w_q, d_lr);
        self.opt_w_k.step(&mut self.w_k, &a_grad_w_k, d_lr);
        self.opt_w_v.step(&mut self.w_v, &a_grad_w_v, d_lr);

        a_grad_in
    }

    fn parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.w_q.iter().copied());
        v.extend(self.w_k.iter().copied());
        v.extend(self.w_v.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w_q.len() + self.w_k.len() + self.w_v.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_self_attention".to_string());
        }

        let mut i_pos: usize = 0;

        for i in 0..self.w_q.len() {
            self.w_q.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_k.len() {
            self.w_k.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_v.len() {
            self.w_v.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

// ---------------------------
// MultiHeadSelfAttention
// ---------------------------
pub struct MultiHeadSelfAttention {
    i_embedding_dim: usize,
    i_num_heads: usize,
    i_head_dim: usize,

    // Projection matrices: [embedding_dim, embedding_dim]
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,

    // Cache for backward
    cached_input: Option<Array2<f32>>,
    cached_q_all: Option<Array2<f32>>,
    cached_k_all: Option<Array2<f32>>,
    cached_v_all: Option<Array2<f32>>,
    cached_concat: Option<Array2<f32>>,
    cached_weights: Option<Vec<Array2<f32>>>, // per head: [seq, seq]

    opt_w_q: AdamW,
    opt_w_k: AdamW,
    opt_w_v: AdamW,
    opt_w_o: AdamW,

    // residual dropout applied to projected output before residual add.
    residual_dropout: Dropout,
}

impl MultiHeadSelfAttention {
    pub fn new(i_embedding_dim: usize, i_num_heads: usize) -> Self {
        // Validation, fail early.
        if i_embedding_dim == 0 {
            panic!("embedding_dim_must_be_positive");
        }
        if i_num_heads == 0 {
            panic!("num_heads_must_be_positive");
        }
        if i_embedding_dim % i_num_heads != 0 {
            panic!("embedding_dim_must_be_divisible_by_num_heads");
        }

        let i_head_dim = i_embedding_dim / i_num_heads;

        let mut rng = rand::rng();
        let d_std = (2.0 / (i_embedding_dim as f32).max(1.0)).sqrt();
        let normal = Normal::new(0.0, d_std).unwrap();

        Self {
            i_embedding_dim,
            i_num_heads,
            i_head_dim,

            w_q: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),
            w_o: Array2::from_shape_fn((i_embedding_dim, i_embedding_dim), |_| normal.sample(&mut rng)),

            cached_input: None,
            cached_q_all: None,
            cached_k_all: None,
            cached_v_all: None,
            cached_concat: None,
            cached_weights: None,

            opt_w_q: AdamW::new((i_embedding_dim, i_embedding_dim), 0.0),
            opt_w_k: AdamW::new((i_embedding_dim, i_embedding_dim), 0.0),
            opt_w_v: AdamW::new((i_embedding_dim, i_embedding_dim), 0.0),
            opt_w_o: AdamW::new((i_embedding_dim, i_embedding_dim), 0.0),

            residual_dropout: Dropout::new(DEFAULT_RESIDUAL_DROPOUT_P, 4242),
        }
    }

    pub fn set_training(&mut self, b_training: bool) {
        self.residual_dropout.set_training(b_training);
    }

    pub fn set_residual_dropout_p(&mut self, d_p: f32) {
        self.residual_dropout.set_p(d_p);
    }

    pub fn reseed_dropout(&mut self, u64_seed: u64) {
        self.residual_dropout.reseed(u64_seed);
    }

    fn softmax(a_scores: &Array2<f32>) -> Array2<f32> {
        math::softmax_rows(a_scores)
    }

    fn softmax_backward(a_softmax: &Array2<f32>, a_grad_out: &Array2<f32>) -> Array2<f32> {
        // Row-wise Jacobian-vector product for softmax: dS = S * (dY - sum(dY * S))
        let mut a_grad_in = a_softmax.clone();
        for i in 0..a_softmax.nrows() {
            let a_row = a_softmax.row(i);
            let a_grow = a_grad_out.row(i);

            let d_dot: f32 = a_row
                .iter()
                .zip(a_grow.iter())
                .map(|(&y, &dy)| y * dy)
                .sum();

            for j in 0..a_softmax.ncols() {
                a_grad_in[[i, j]] = a_softmax[[i, j]] * (a_grad_out[[i, j]] - d_dot);
            }
        }
        a_grad_in
    }

    fn split_heads(&self, a_x: &Array2<f32>) -> Result<Vec<Array2<f32>>, String> {
        // a_x: [seq, embedding] -> vec heads [seq, head_dim]
        if a_x.ncols() != self.i_embedding_dim {
            return Err("mhsa_split_heads_dim_mismatch".to_string());
        }

        let i_seq_len = a_x.nrows();
        let mut v_heads: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);

        for i_h in 0..self.i_num_heads {
            let i_start = i_h * self.i_head_dim;
            let i_end = i_start + self.i_head_dim;
            let a_view = a_x.slice(ndarray::s![.., i_start..i_end]).to_owned();

            if a_view.nrows() != i_seq_len || a_view.ncols() != self.i_head_dim {
                return Err("mhsa_split_heads_slice_error".to_string());
            }
            v_heads.push(a_view);
        }

        Ok(v_heads)
    }

    fn concat_heads(&self, v_heads: &[Array2<f32>]) -> Result<Array2<f32>, String> {
        if v_heads.len() != self.i_num_heads {
            return Err("mhsa_concat_heads_count_mismatch".to_string());
        }

        let i_seq_len = v_heads[0].nrows();
        for a_h in v_heads.iter() {
            if a_h.nrows() != i_seq_len || a_h.ncols() != self.i_head_dim {
                return Err("mhsa_concat_heads_shape_mismatch".to_string());
            }
        }

        let mut a_out = Array2::<f32>::zeros((i_seq_len, self.i_embedding_dim));
        for i_h in 0..self.i_num_heads {
            let i_start = i_h * self.i_head_dim;
            let i_end = i_start + self.i_head_dim;
            let mut a_slice = a_out.slice_mut(ndarray::s![.., i_start..i_end]);
            a_slice.assign(&v_heads[i_h]);
        }

        Ok(a_out)
    }

    fn apply_causal_mask_inplace(a_scores: &mut Array2<f32>) {
        let i_seq_len = a_scores.nrows();
        for i in 0..i_seq_len {
            for j in (i + 1)..i_seq_len {
                a_scores[[i, j]] = f32::NEG_INFINITY;
            }
        }
    }

    fn attention_head_forward(
        &self,
        a_q: &Array2<f32>,
        a_k: &Array2<f32>,
        a_v: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // Returns (head_out, weights)
        let d_scale = (self.i_head_dim as f32).sqrt().max(1e-12);

        let mut a_scores = a_q.dot(&a_k.t()) / d_scale;
        Self::apply_causal_mask_inplace(&mut a_scores);

        let a_weights = Self::softmax(&a_scores);
        let a_out = a_weights.dot(a_v);

        (a_out, a_weights)
    }
}

impl Layer for MultiHeadSelfAttention {
    fn layer_type(&self) -> &str {
        "MultiHeadSelfAttention"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        // a_input: [seq, embedding_dim]
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }
        if a_input.ncols() != self.i_embedding_dim {
            return Array2::<f32>::zeros((0, 0));
        }

        // Cache input.
        self.cached_input = Some(a_input.clone());

        // Project.
        let a_q_all = a_input.dot(&self.w_q);
        let a_k_all = a_input.dot(&self.w_k);
        let a_v_all = a_input.dot(&self.w_v);

        self.cached_q_all = Some(a_q_all.clone());
        self.cached_k_all = Some(a_k_all.clone());
        self.cached_v_all = Some(a_v_all.clone());

        // Split heads.
        let v_q = match self.split_heads(&a_q_all) {
            Ok(v) => v,
            Err(_) => return Array2::<f32>::zeros((0, 0)),
        };
        let v_k = match self.split_heads(&a_k_all) {
            Ok(v) => v,
            Err(_) => return Array2::<f32>::zeros((0, 0)),
        };
        let v_v = match self.split_heads(&a_v_all) {
            Ok(v) => v,
            Err(_) => return Array2::<f32>::zeros((0, 0)),
        };

        // Per-head attention.
        let mut v_head_out: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_weights: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);

        for i_h in 0..self.i_num_heads {
            let (a_h_out, a_w) = self.attention_head_forward(&v_q[i_h], &v_k[i_h], &v_v[i_h]);
            v_head_out.push(a_h_out);
            v_weights.push(a_w);
        }

        self.cached_weights = Some(v_weights);

        // Concat and output projection.
        let a_concat = match self.concat_heads(&v_head_out) {
            Ok(a) => a,
            Err(_) => return Array2::<f32>::zeros((0, 0)),
        };
        self.cached_concat = Some(a_concat.clone());

        let a_proj = a_concat.dot(&self.w_o);

        // residual dropout on projected residual branch only.
        let a_proj_dropped = self.residual_dropout.apply(&a_proj);

        a_proj_dropped + a_input
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        // Correct backward for MHSA with causal scaled dot product attention.
        // Shapes:
        // a_input: [seq, emb]
        // a_grads: [seq, emb] gradient wrt output of (proj + residual)
        //
        // Output:
        // grad wrt input: [seq, emb]

        // Safety checks.
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return a_grads.clone();
        }

        let a_input = match self.cached_input.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_q_all = match self.cached_q_all.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_k_all = match self.cached_k_all.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_v_all = match self.cached_v_all.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_concat = match self.cached_concat.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let v_weights = match self.cached_weights.as_ref() {
            Some(v) => v,
            None => return a_grads.clone(),
        };

        if a_input.raw_dim() != a_grads.raw_dim() {
            return a_grads.clone();
        }

        // Residual path: output = proj + input
        // So grad splits: dL/dproj = a_grads, dL/dinput_residual = a_grads
        let a_grad_proj = a_grads;

        // Output projection: proj = concat * w_o
        let a_grad_w_o = a_concat.t().dot(a_grad_proj);
        let a_grad_concat = a_grad_proj.dot(&self.w_o.t()); // [seq, emb]

        // Split grad_concat to heads.
        let v_grad_head_out = match self.split_heads(&a_grad_concat) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };

        // Split q/k/v to heads.
        let v_q = match self.split_heads(a_q_all) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };
        let v_k = match self.split_heads(a_k_all) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };
        let v_v = match self.split_heads(a_v_all) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };

        // Accumulate per-head grads.
        let mut v_grad_q: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_grad_k: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_grad_v: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);

        let d_scale = (self.i_head_dim as f32).sqrt().max(1e-12);
        let i_seq_len = a_input.nrows();

        for i_h in 0..self.i_num_heads {
            let a_q = &v_q[i_h];
            let a_k = &v_k[i_h];
            let a_v = &v_v[i_h];
            let a_w = &v_weights[i_h]; // [seq, seq]
            let a_grad_h_out = &v_grad_head_out[i_h]; // [seq, head_dim]

            // head_out = W * V
            // dW = dO * V^T
            // dV = W^T * dO
            let a_grad_w = a_grad_h_out.dot(&a_v.t()); // [seq, seq]
            let a_grad_v_h = a_w.t().dot(a_grad_h_out); // [seq, head_dim]

            // W = softmax(scores)
            // dScores = softmax_backward(W, dW)
            let mut a_grad_scores = Self::softmax_backward(a_w, &a_grad_w); // [seq, seq]

            // Apply causal mask gradient: masked positions are constant -inf in forward,
            // therefore treat gradients there as zero to avoid spurious updates.
            for i in 0..i_seq_len {
                for j in (i + 1)..i_seq_len {
                    a_grad_scores[[i, j]] = 0.0;
                }
            }

            // scores = (Q K^T) / scale
            // dQ = dScores * K / scale
            // dK = dScores^T * Q / scale
            let a_grad_q_h = a_grad_scores.dot(a_k) / d_scale;
            let a_grad_k_h = a_grad_scores.t().dot(a_q) / d_scale;

            v_grad_q.push(a_grad_q_h);
            v_grad_k.push(a_grad_k_h);
            v_grad_v.push(a_grad_v_h);
        }

        // Concat per-head grads to embedding space.
        let a_grad_q_all = match self.concat_heads(&v_grad_q) {
            Ok(a) => a,
            Err(_) => return a_grads.clone(),
        };
        let a_grad_k_all = match self.concat_heads(&v_grad_k) {
            Ok(a) => a,
            Err(_) => return a_grads.clone(),
        };
        let a_grad_v_all = match self.concat_heads(&v_grad_v) {
            Ok(a) => a,
            Err(_) => return a_grads.clone(),
        };

        // Linear projections:
        // Q = X * w_q, K = X * w_k, V = X * w_v
        // dW = X^T * dY
        // dX = dY * W^T
        let a_grad_w_q = a_input.t().dot(&a_grad_q_all);
        let a_grad_w_k = a_input.t().dot(&a_grad_k_all);
        let a_grad_w_v = a_input.t().dot(&a_grad_v_all);

        let a_grad_x_from_q = a_grad_q_all.dot(&self.w_q.t());
        let a_grad_x_from_k = a_grad_k_all.dot(&self.w_k.t());
        let a_grad_x_from_v = a_grad_v_all.dot(&self.w_v.t());

        // Total grad wrt input is sum of:
        // - residual path
        // - projections paths via q/k/v
        let a_grad_input_total = a_grads.clone() + a_grad_x_from_q + a_grad_x_from_k + a_grad_x_from_v;

        // Update parameters.
        self.opt_w_o.step(&mut self.w_o, &a_grad_w_o, d_lr);
        self.opt_w_q.step(&mut self.w_q, &a_grad_w_q, d_lr);
        self.opt_w_k.step(&mut self.w_k, &a_grad_w_k, d_lr);
        self.opt_w_v.step(&mut self.w_v, &a_grad_w_v, d_lr);

        a_grad_input_total
    }

    fn parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.w_q.iter().copied());
        v.extend(self.w_k.iter().copied());
        v.extend(self.w_v.iter().copied());
        v.extend(self.w_o.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_multi_head_self_attention".to_string());
        }

        let mut i_pos: usize = 0;

        for i in 0..self.w_q.len() {
            self.w_q.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_k.len() {
            self.w_k.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_v.len() {
            self.w_v.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.w_o.len() {
            self.w_o.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

// ---------------------------
// TransformerSequence
// ---------------------------
// Sequentially applies multiple TransformerBlocks as one logical Layer.
// Forward:  x -> tb1 -> tb2 -> ... -> out
// Backward: reverse order backprop through internal blocks.
//
// This enables "block sequences" to be used as branches inside ParallelBlockGroup.
pub struct TransformerSequence {
    v_blocks: Vec<TransformerBlock>,
}

impl TransformerSequence {
    pub fn new(v_blocks: Vec<TransformerBlock>) -> Result<Self, String> {
        if v_blocks.is_empty() {
            return Err("transformer_sequence_empty".to_string());
        }
        Ok(Self { v_blocks })
    }

    pub fn set_training(&mut self, b_training: bool) {
        for tb in self.v_blocks.iter_mut() {
            tb.set_training(b_training);
        }
    }

    pub fn set_residual_dropout_p(&mut self, d_p: f32) {
        for tb in self.v_blocks.iter_mut() {
            tb.set_residual_dropout_p(d_p);
        }
    }

    pub fn reseed_dropout(&mut self, u64_seed: u64) {
        for (i_idx, tb) in self.v_blocks.iter_mut().enumerate() {
            let u64_mix = (i_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            tb.reseed_dropout(u64_seed ^ u64_mix);
        }
    }
}

impl Layer for TransformerSequence {
    fn layer_type(&self) -> &str {
        "TransformerSequence"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }

        let mut a_act = a_input.clone();
        for tb in self.v_blocks.iter_mut() {
            a_act = tb.forward(&a_act);
            if a_act.nrows() == 0 || a_act.ncols() == 0 {
                return Array2::zeros((0, 0));
            }
        }
        a_act
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        if a_grads.nrows() == 0 || a_grads.ncols() == 0 {
            return a_grads.clone();
        }

        let mut a_g = a_grads.clone();
        for tb in self.v_blocks.iter_mut().rev() {
            a_g = tb.backward(&a_g, d_lr);
            if a_g.nrows() == 0 || a_g.ncols() == 0 {
                return Array2::zeros((0, 0));
            }
        }
        a_g
    }

    fn parameters(&self) -> usize {
        self.v_blocks.iter().map(|b| b.parameters()).sum()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        for b in self.v_blocks.iter() {
            v.extend(b.get_parameters_flat());
        }
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let mut i_used: usize = 0;
        for b in self.v_blocks.iter_mut() {
            let i_n = b.set_parameters_flat(&v_params[i_used..])?;
            i_used = i_used.saturating_add(i_n);
        }
        Ok(i_used)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

// ---------------------------
// ParallelBlockGroupMetrics
// ---------------------------
// TransformerBlock update
// ---------------------------
pub struct TransformerBlock {
    attention: MultiHeadSelfAttention,
    feed_forward: FeedForward,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl TransformerBlock {
    pub fn new(i_embedding_dim: usize, i_hidden_dim: usize) -> Self {
        // Expert default: embedding_dim 128 is divisible by 4.
        // If EMBEDDING_DIM changes, new() panics early if invalid.
        let i_num_heads: usize = 4;

        Self {
            attention: MultiHeadSelfAttention::new(i_embedding_dim, i_num_heads),
            feed_forward: FeedForward::new(i_embedding_dim, i_hidden_dim),
            norm1: RmsNorm::new(i_embedding_dim),
            norm2: RmsNorm::new(i_embedding_dim),
        }
    }

    pub fn set_training(&mut self, b_training: bool) {
        self.attention.set_training(b_training);
        self.feed_forward.set_training(b_training);
    }

    pub fn set_residual_dropout_p(&mut self, d_p: f32) {
        self.attention.set_residual_dropout_p(d_p);
        self.feed_forward.set_residual_dropout_p(d_p);
    }

    pub fn reseed_dropout(&mut self, u64_seed: u64) {
        // Derive distinct seeds per submodule deterministically.
        self.attention.reseed_dropout(u64_seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        self.feed_forward.reseed_dropout(u64_seed ^ 0x5A5A_5A5A_5A5A_5A5A);
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        let a_attn = self.attention.forward(a_input);
        let a_n1 = self.norm1.normalize(&a_attn);
        let a_ff = self.feed_forward.forward(&a_n1);
        self.norm2.normalize(&a_ff)
    }


    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_g2 = self.norm2.backward(a_grads, d_lr);
        let a_gff = self.feed_forward.backward(&a_g2, d_lr);
        let a_g1 = self.norm1.backward(&a_gff, d_lr);
        self.attention.backward(&a_g1, d_lr)
    }

    fn parameters(&self) -> usize {
        self.attention.parameters()
            + self.feed_forward.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.attention.get_parameters_flat());
        v.extend(self.norm1.get_parameters_flat());
        v.extend(self.feed_forward.get_parameters_flat());
        v.extend(self.norm2.get_parameters_flat());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let mut i_used: usize = 0;

        let i1 = self.attention.set_parameters_flat(&v_params[i_used..])?;
        i_used += i1;

        let i2 = self.norm1.set_parameters_flat(&v_params[i_used..])?;
        i_used += i2;

        let i3 = self.feed_forward.set_parameters_flat(&v_params[i_used..])?;
        i_used += i3;

        let i4 = self.norm2.set_parameters_flat(&v_params[i_used..])?;
        i_used += i4;

        Ok(i_used)
    }
}

pub struct OutputProjection {
    w_out: Array2<f32>,
    b_out: Array2<f32>,
    optimizer_w: AdamW,
    optimizer_b: AdamW,
    cached_input: Option<Array2<f32>>,
}

impl OutputProjection {
    pub fn new(i_embedding_dim: usize, i_vocab_size: usize) -> Self {
        let mut rng = rand::rng();
        let std = (2.0 / (i_embedding_dim as f32).max(1.0)).sqrt();
        let normal = rand_distr::Normal::new(0.0, std).unwrap();

        Self {
            w_out: Array2::from_shape_fn((i_embedding_dim, i_vocab_size), |_| {
                normal.sample(&mut rng)
            }),
            b_out: Array2::zeros((1, i_vocab_size)),
            optimizer_w: AdamW::new((i_embedding_dim, i_vocab_size), 0.01),
            optimizer_b: AdamW::new((1, i_vocab_size), 0.0),
            cached_input: None,
        }
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(a_input.clone());
        a_input.dot(&self.w_out) + &self.b_out
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = self
            .cached_input
            .as_ref()
            .expect("forward must be run first");

        let a_grad_w = a_input.t().dot(a_grads);
        let a_grad_b = a_grads.sum_axis(Axis(0)).insert_axis(Axis(0));
        let a_grad_in = a_grads.dot(&self.w_out.t());

        self.optimizer_w.step(&mut self.w_out, &a_grad_w, d_lr);
        self.optimizer_b.step(&mut self.b_out, &a_grad_b, d_lr);

        a_grad_in
    }

    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        v.extend(self.w_out.iter().copied());
        v.extend(self.b_out.iter().copied());
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let i_needed = self.w_out.len() + self.b_out.len();
        if v_params.len() < i_needed {
            return Err("checkpoint_not_enough_params_output_projection".to_string());
        }

        let mut i_pos: usize = 0;
        for i in 0..self.w_out.len() {
            self.w_out.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }
        for i in 0..self.b_out.len() {
            self.b_out.as_slice_mut().unwrap()[i] = v_params[i_pos];
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

#[derive(Clone, Debug)]
pub struct Dropout {
    d_p: f32,
    b_training: bool,
    rng: StdRng,
}

impl Dropout {
    pub fn new(d_p: f32, u64_seed: u64) -> Self {
        let d_pp = if d_p.is_finite() { d_p.clamp(0.0, 0.95) } else { 0.0 };
        Self {
            d_p: d_pp,
            b_training: true,
            rng: StdRng::seed_from_u64(u64_seed),
        }
    }

    pub fn set_training(&mut self, b_training: bool) {
        self.b_training = b_training;
    }

    pub fn set_p(&mut self, d_p: f32) {
        if d_p.is_finite() {
            self.d_p = d_p.clamp(0.0, 0.95);
        }
    }

    pub fn reseed(&mut self, u64_seed: u64) {
        self.rng = StdRng::seed_from_u64(u64_seed);
    }

    // Inverted dropout:
    // y = x * mask / (1 - p)
    pub fn apply(&mut self, a_x: &Array2<f32>) -> Array2<f32> {
        if !self.b_training {
            return a_x.clone();
        }
        if self.d_p <= 0.0 {
            return a_x.clone();
        }
        if a_x.nrows() == 0 || a_x.ncols() == 0 {
            return a_x.clone();
        }

        let d_keep = 1.0 - self.d_p;
        if d_keep <= 0.0 || !d_keep.is_finite() {
            return Array2::<f32>::zeros(a_x.raw_dim());
        }

        let d_scale = 1.0 / d_keep;
        let mut a_out = a_x.clone();

        for d in a_out.iter_mut() {
            let d_u: f32 = self.rng.gen_range(0.0..1.0);
            let b_keep: bool = d_u < d_keep;
            if b_keep {
                let d_v = *d * d_scale;
                *d = if d_v.is_finite() { d_v } else { 0.0 };
            } else {
                *d = 0.0;
            }
        }

        a_out
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct Llm {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
    pub bpe_tokenizer: Option<BpeTokenizer>,

    // Residual dropout control
    pub b_training: bool,
    pub u64_dropout_seed: u64,
    pub d_residual_dropout_p: f32,

    // Sampling configuration for generation.
    // NOTE: These fields are used by forward_generate via sample_next_token_from_logits.
    pub d_temperature: f32,
    pub i_top_k: usize,
    pub d_top_p: f32,
    pub rng_sampling: StdRng,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmCheckpoint {
    pub s_magic: String,
    pub s_version: String,
    pub i_max_seq_len: usize,
    pub i_embedding_dim: usize,
    pub i_hidden_dim: usize,
    pub tokenizer: BpeTokenizerCheckpoint,
    pub v_params: Vec<f32>,
}

impl LlmCheckpoint {
    pub fn new(
        tokenizer: BpeTokenizerCheckpoint,
        v_params: Vec<f32>,
        i_max_seq_len: usize,
        i_embedding_dim: usize,
        i_hidden_dim: usize,
    ) -> Self {
        Self {
            s_magic: "EXCHAT_LLM_CHECKPOINT".to_string(),
            s_version: "1".to_string(),
            i_max_seq_len,
            i_embedding_dim,
            i_hidden_dim,
            tokenizer,
            v_params,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.s_magic != "EXCHAT_LLM_CHECKPOINT" {
            return Err("checkpoint_magic_mismatch".to_string());
        }
        if self.s_version != "1" {
            return Err("checkpoint_version_unsupported".to_string());
        }
        if self.i_max_seq_len != MAX_SEQ_LEN {
            return Err("checkpoint_max_seq_len_mismatch".to_string());
        }
        if self.i_embedding_dim != EMBEDDING_DIM {
            return Err("checkpoint_embedding_dim_mismatch".to_string());
        }
        if self.i_hidden_dim != HIDDEN_DIM {
            return Err("checkpoint_hidden_dim_mismatch".to_string());
        }
        if self.v_params.is_empty() {
            return Err("checkpoint_empty_params".to_string());
        }
        Ok(())
    }
}

// ---------------------------
// Llm: diagnostics hook after load
// ---------------------------

impl crate::layer::Llm {
    // Create small deterministic inputs to the MTB group based on real model flow:
    // - run embeddings and layers until the first ParallelBlockGroup
    // - collect that activation as input for the group
    fn collect_parallel_block_group_inputs_for_diagnostics(
        &mut self,
        v_texts: &[String],
        i_max_samples: usize,
    ) -> Result<Vec<Array2<f32>>, String> {
        if v_texts.is_empty() {
            return Err("diagnostics_texts_empty".to_string());
        }
        if i_max_samples == 0 {
            return Err("diagnostics_max_samples_zero".to_string());
        }

        let mut v_inputs: Vec<Array2<f32>> = Vec::new();

        // Identify index of first ParallelBlockGroup in the network.
        let mut opt_idx: Option<usize> = None;
        for (i_idx, layer) in self.network.iter_mut().enumerate() {
            if layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                .is_some()
            {
                opt_idx = Some(i_idx);
                break;
            }
        }

        let i_pg_idx = opt_idx.ok_or_else(|| "diagnostics_no_parallel_block_group".to_string())?;

        let i_take = i_max_samples.min(v_texts.len());
        for s in v_texts.iter().take(i_take) {
            let v_ids = match self.tokenize(s) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v_ids.len() < 2 {
                continue;
            }

            let a_token_input: Array2<f32> = Array2::from_shape_vec(
                (1, v_ids.len()),
                v_ids.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            )
            .map_err(|_| "diagnostics_shape_error_token_input".to_string())?;

            // Forward until the ParallelBlockGroup input.
            let mut a_act = a_token_input;
            for i_l in 0..i_pg_idx {
                a_act = self.network[i_l].forward(&a_act);
                if a_act.nrows() == 0 || a_act.ncols() == 0 {
                    break;
                }
            }

            if a_act.nrows() == 0 || a_act.ncols() == 0 {
                continue;
            }

            v_inputs.push(a_act);
        }

        if v_inputs.is_empty() {
            return Err("diagnostics_no_valid_inputs".to_string());
        }

        Ok(v_inputs)
    }

    // Compute and print diagnostics for all ParallelBlockGroup layers in the network.
    // History:
    // - 2026-02-07: Add post-load MTB metrics computation (PSI, diversity, plus 5 more).
    pub fn run_post_load_mtb_diagnostics_ascii(&mut self) {
        // Best effort: diagnostics must not break load.
        let b_prev = self.b_training;
        self.set_training(false);

        // Collect deterministic text prompts if available.
        // Use a compact fixed set to avoid dependency on Dataset in this module.
        let v_prompts: Vec<String> = vec![
            "User: Wo liegt der Mount Everest?".to_string(),
            "User: Wie alt koennen Eichen werden?".to_string(),
            "User: Wie ist Jupiter?".to_string(),
            "User: was ist KUENSTLICHE intelligenz im engen sinne?".to_string(),
            "User: KI eng definiert - was ist damit gemeint?".to_string(),
            "User: In welchem Nationalpark liegt der nepalesische Zugang zum Mount Everest?".to_string(),
            "User: Wer bestieg als erster den mount everest?".to_string(),
            "User: Wie sind Pinguine?".to_string(),
        ];

        let v_inputs = match self.collect_parallel_block_group_inputs_for_diagnostics(&v_prompts, 8) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("MTB diagnostics skipped: {}", e);
                self.set_training(b_prev);
                return;
            }
        };

        for layer in self.network.iter_mut() {
            let opt_pg = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
            if opt_pg.is_none() {
                continue;
            }
            let pg = opt_pg.unwrap();

            match pg.compute_metrics_from_inputs(&v_inputs) {
                Ok(m) => {
                    for s_line in m.to_ascii_report_lines() {
                        println!("{}", s_line);
                    }
                }
                Err(e) => {
                    eprintln!("MTB diagnostics failed: {}", e);
                }
            }
        }

        self.set_training(b_prev);
    }
}

impl Llm {
     pub fn new(vocab: Vocab, network: Vec<Box<dyn Layer>>) -> Self {
        Self {
            vocab,
            network,
            bpe_tokenizer: None,
            b_training: true,
            u64_dropout_seed: 1337,
            d_residual_dropout_p: DEFAULT_RESIDUAL_DROPOUT_P,

            // Safe defaults:
            // - temperature 1.0 (no scaling)
            // - top_k 0 (disabled)
            // - top_p 0.0 (disabled)
            // - fixed seed for deterministic behavior unless changed by caller
            d_temperature: 1.0,
            i_top_k: 0,
            d_top_p: 0.0,
            rng_sampling: rand::rngs::StdRng::seed_from_u64(12345),
        }
    }

    // Sampling configuration used by predict and forward_generate.
    // Conventions:
    // - temperature: > 0.0, typical in [0.7..1.3], 1.0 means no scaling
    // - top_k: 0 disables, otherwise keeps only k highest probability tokens
    // - top_p: 0.0 disables, otherwise keeps smallest set with cumulative prob >= p (0 < p <= 1)
    // - seed: deterministic sampling when set; value is advanced internally
    pub fn set_sampling_config(
        &mut self,
        d_temperature: f32,
        i_top_k: usize,
        d_top_p: f32,
        u64_seed: u64,
    ) -> Result<(), String> {
        if !d_temperature.is_finite() || d_temperature <= 0.0 {
            return Err("sampling_temperature_invalid".to_string());
        }
        if d_top_p.is_finite() {
            if d_top_p < 0.0 || d_top_p > 1.0 {
                return Err("sampling_top_p_out_of_range".to_string());
            }
        } else {
            return Err("sampling_top_p_invalid".to_string());
        }

        // Store in existing struct by adding these fields to Llm:
        // pub d_temperature: f32,
        // pub i_top_k: usize,
        // pub d_top_p: f32,
        // pub rng_sampling: StdRng,
        self.d_temperature = d_temperature;
        self.i_top_k = i_top_k;
        self.d_top_p = d_top_p;
        self.rng_sampling = StdRng::seed_from_u64(u64_seed);

        Ok(())
    }


    pub fn set_training(&mut self, b_training: bool) {
        self.b_training = b_training;

        for layer in self.network.iter_mut() {
            if let Some(tb) = layer.as_any_mut().and_then(|a| a.downcast_mut::<TransformerBlock>()) {
                tb.set_training(b_training);
                continue;
            }
            if let Some(pg) = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>()) {
                pg.set_training(b_training);
                continue;
            }
        }
    }

   pub fn set_residual_dropout_p(&mut self, d_p: f32) {
        if d_p.is_finite() {
            self.d_residual_dropout_p = d_p.clamp(0.0, 0.95);
        }

        for layer in self.network.iter_mut() {
            if let Some(tb) = layer.as_any_mut().and_then(|a| a.downcast_mut::<TransformerBlock>()) {
                tb.set_residual_dropout_p(self.d_residual_dropout_p);
                continue;
            }
            if let Some(pg) = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>()) {
                pg.set_residual_dropout_p(self.d_residual_dropout_p);
                continue;
            }
        }
    }


    pub fn set_bpe_tokenizer(&mut self, bpe_tokenizer: BpeTokenizer) {
        self.vocab = bpe_tokenizer.vocab.clone();
        self.bpe_tokenizer = Some(bpe_tokenizer);
    }

    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|l| l.layer_type())
            .collect::<Vec<&str>>()
            .join(", ")
    }

    pub fn total_parameters(&self) -> usize {
        self.network.iter().map(|l| l.parameters()).sum()
    }

    pub fn decode_ids(&self, v_ids: &[usize]) -> String {
        if let Some(tok) = &self.bpe_tokenizer {
            return tok.decode_ids(v_ids);
        }
        utils::decode_via_vocab_ascii(&self.vocab, v_ids)
    }

    pub fn tokenize(&self, s_text: &str) -> Result<Vec<usize>, String> {
        let tok = self
            .bpe_tokenizer
            .as_ref()
            .ok_or_else(|| "tokenizer_not_set".to_string())?;

        let mut v_ids = tok.encode_text(s_text, false);
        if v_ids.is_empty() {
            return Err("tokenizer_returned_empty".to_string());
        }
        if v_ids.len() > MAX_SEQ_LEN {
            v_ids.truncate(MAX_SEQ_LEN);
        }
        Ok(v_ids)
    }

    // Loads checkpoint and returns a fresh Llm with correct dimensions.
    // This prevents vocab_size mismatches that change OutputProjection shape.
    pub fn load_checkpoint_rebuild(s_path: &str) -> Result<Llm, String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        let s_json =
            std::fs::read_to_string(s_path).map_err(|_| "checkpoint_read_error".to_string())?;
        let cp: LlmCheckpoint = crate::utils::checkpoint_from_json_ascii(&s_json)?;
        cp.validate()?;

        let bpe = BpeTokenizer::from_checkpoint(&cp.tokenizer)?;

        // Build new model using checkpoint tokenizer vocab size.
        let vocab = bpe.vocab.clone();
        let embeddings = Embeddings::new(vocab.clone());
        let block1 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);

        // MTB stage: parallel branches inside one logical layer position, now as sequences.
        let block2_1 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
        let block2_2 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
        let block2_3 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
        let block2_4 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);

        let seq_2_1 = TransformerSequence::new(vec![block2_1, block2_2])
            .expect("transformer_sequence_new_failed");
        let seq_2_2 = TransformerSequence::new(vec![block2_3, block2_4])
            .expect("transformer_sequence_new_failed");

        let parallel_block2 = ParallelBlockGroup::new(vec![
            Box::new(seq_2_1) as Box<dyn Layer>,
            Box::new(seq_2_2) as Box<dyn Layer>,
        ])
        .expect("parallel_block_group_new_failed");

        let block3 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
        let out = OutputProjection::new(crate::EMBEDDING_DIM, vocab.words.len());

        let mut llm = Llm::new(
            vocab,
            vec![
                Box::new(embeddings),
                Box::new(block1),
                Box::new(parallel_block2),
                Box::new(block3),
                Box::new(out),
            ],
        );
        llm.set_bpe_tokenizer(bpe);
        llm.set_residual_dropout_p(0.1);
        llm.set_training(true);
        if let Err(e) = llm.set_sampling_config(0.9, 40, 0.95, 987654321) {
            eprintln!("Sampling config failed: {}", e);
        }

        let i_expected: usize = llm
            .network
            .iter()
            .map(|l| l.get_parameters_flat().len())
            .sum();

        if i_expected != cp.v_params.len() {
            return Err("checkpoint_param_count_mismatch".to_string());
        }

        llm.assign_all_parameters_flat(&cp.v_params)?;
        llm.run_post_load_mtb_diagnostics_ascii();
        Ok(llm)
    }


    // Helper: choose an outage branch index for test purposes.
    // Uses RNG from sampling RNG to remain deterministic given a fixed seed.
    fn choose_outage_branch_idx_test_only(&mut self, i_num_paths: usize) -> Option<usize> {
        if i_num_paths == 0 {
            return None;
        }
        // Always drop exactly one path for the test, uniformly at random.
        let i_idx = self.rng_sampling.gen_range(0..i_num_paths);
        Some(i_idx)
    }

    // Helper: apply outage selection to all ParallelBlockGroup layers.
    // Test-only: select exactly one outage path per ParallelBlockGroup before each predict.
    fn set_predict_outage_for_all_parallel_groups_test_only(&mut self) {
        // Move RNG out of self to avoid borrowing self mutably while iterating mutably over self.network.
        // This keeps code safe and borrow-checker compliant.
        let mut rng_local: rand::rngs::StdRng =
            std::mem::replace(&mut self.rng_sampling, rand::rngs::StdRng::seed_from_u64(0));

        for layer in self.network.iter_mut() {
            let opt_pg = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
            if let Some(pg) = opt_pg {
                let i_k = pg.num_branches();

                let opt_drop: Option<usize> = if i_k == 0 {
                    None
                } else {
                    // Uniform random drop of exactly one path (test-only).
                    Some(rng_local.gen_range(0..i_k))
                };

                pg.set_fault_injection_enabled(true);
                pg.set_fault_drop_branch_idx(opt_drop);
            }
        }

        // Move RNG back into self.
        self.rng_sampling = rng_local;
    }

    // Helper: clear outage state after predict.
    fn clear_predict_outage_for_all_parallel_groups_test_only(&mut self) {
        for layer in self.network.iter_mut() {
            let opt_pg = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
            if let Some(pg) = opt_pg {
                pg.set_fault_drop_branch_idx(None);
                pg.set_fault_injection_enabled(false);
            }
        }
    }


    // - 2026-02-04: Ensure predict runs in eval mode (no dropout) and restores training state.
    // - 2026-02-07: Fault injection: drop one path before each predict (test-only).
    pub fn predict(&mut self, s_text: &str) -> Result<String, String> {
        let b_prev = self.b_training;
        self.set_training(false);

        // Test-only: simulate outage.
        self.set_predict_outage_for_all_parallel_groups_test_only();

        let r = self
            .forward_generate(s_text)
            .map(|v_out_ids| self.decode_ids(&v_out_ids));

        // Clear test-only outage state.
        self.clear_predict_outage_for_all_parallel_groups_test_only();

        self.set_training(b_prev);
        r
    }

    // - 2026-02-04: Replace greedy decoding by temperature/top-k/top-p sampling.
    fn forward_generate(&mut self, s_text: &str) -> Result<Vec<usize>, String> {
        let mut v_context = self.tokenize(s_text)?;
        let mut v_generated: Vec<usize> = Vec::new();

        if v_context.len() >= MAX_SEQ_LEN {
            return Ok(v_generated);
        }

        let opt_eos = self.vocab.encode(S_EOS);

        for _ in 0..(MAX_SEQ_LEN - v_context.len()) {
            let a_token_input: Array2<f32> = Array2::from_shape_vec(
                (1, v_context.len()),
                v_context.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            )
            .map_err(|_| "shape_error_token_input".to_string())?;

            let mut a_act = a_token_input;
            for layer in self.network.iter_mut() {
                a_act = layer.forward(&a_act);
            }

            let a_logits = a_act;
            if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
                return Err("empty_logits".to_string());
            }

            // Use last position logits.
            let a_last_logits = a_logits
                .row(a_logits.nrows().saturating_sub(1))
                .to_owned()
                .insert_axis(Axis(0));

            let i_next = self.sample_next_token_from_logits(&a_last_logits)?;
            v_generated.push(i_next);
            v_context.push(i_next);

            if let Some(i_eos) = opt_eos {
                if i_next == i_eos {
                    break;
                }
            }
        }

        Ok(v_generated)
    }

    // Sample next token from logits with:
    // - temperature scaling
    // - top-k filtering
    // - top-p (nucleus) filtering
    fn sample_next_token_from_logits(&mut self, a_last_logits: &Array2<f32>) -> Result<usize, String> {
        if a_last_logits.nrows() != 1 || a_last_logits.ncols() == 0 {
            return Err("sampling_logits_shape_invalid".to_string());
        }

        let i_vocab = a_last_logits.ncols();

        // Validate config fields exist and are sane.
        let d_temperature = if self.d_temperature.is_finite() && self.d_temperature > 0.0 {
            self.d_temperature
        } else {
            1.0
        };

        let i_top_k = self.i_top_k;
        let d_top_p = if self.d_top_p.is_finite() { self.d_top_p } else { 0.0 };

        // Temperature scaling applied to logits before softmax.
        // For numerical safety, clamp extremely small temperatures.
        let d_temp = d_temperature.max(1e-6);

        let mut a_scaled = a_last_logits.clone();
        for d in a_scaled.iter_mut() {
            if !d.is_finite() {
                *d = 0.0;
            } else {
                *d = *d / d_temp;
            }
        }

        // Convert to probabilities.
        let a_probs = math::softmax_rows(&a_scaled);
        if a_probs.nrows() != 1 || a_probs.ncols() != i_vocab {
            return Err("sampling_probs_shape_invalid".to_string());
        }

        // Build (token_id, prob) pairs.
        let mut v_pairs: Vec<(usize, f32)> = (0..i_vocab)
            .map(|i| (i, a_probs[[0, i]]))
            .filter(|(_, p)| p.is_finite() && *p > 0.0)
            .collect();

        if v_pairs.is_empty() {
            return Err("sampling_probs_empty".to_string());
        }

        // Sort descending by prob.
        v_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Apply top-k if enabled.
        let mut v_filtered: Vec<(usize, f32)> = if i_top_k > 0 && i_top_k < v_pairs.len() {
            v_pairs[..i_top_k].to_vec()
        } else {
            v_pairs
        };

        // Apply top-p if enabled.
        // Keep minimal set with cumulative sum >= top_p.
        if d_top_p > 0.0 && d_top_p < 1.0 {
            let mut d_cum: f32 = 0.0;
            let mut v_nucleus: Vec<(usize, f32)> = Vec::new();

            for (i_id, d_p) in v_filtered.iter().copied() {
                d_cum += d_p;
                v_nucleus.push((i_id, d_p));
                if d_cum >= d_top_p {
                    break;
                }
            }

            // Safety: ensure at least one token remains.
            if !v_nucleus.is_empty() {
                v_filtered = v_nucleus;
            }
        }

        // Renormalize filtered probabilities.
        let d_sum: f32 = v_filtered.iter().map(|(_, p)| *p).sum();
        if !d_sum.is_finite() || d_sum <= 0.0 {
            return Err("sampling_filtered_sum_invalid".to_string());
        }

        let v_weights: Vec<f32> = v_filtered.iter().map(|(_, p)| *p / d_sum).collect();
        let v_ids: Vec<usize> = v_filtered.iter().map(|(i, _)| *i).collect();

        // WeightedIndex expects non-negative weights.
        if v_weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err("sampling_weights_invalid".to_string());
        }

        let dist = WeightedIndex::new(&v_weights).map_err(|_| "sampling_weighted_index_error".to_string())?;
        let i_pick = dist.sample(&mut self.rng_sampling);
        v_ids.get(i_pick).copied().ok_or_else(|| "sampling_pick_oob".to_string())
    }

    pub fn train(&mut self, v_data: Vec<&str>, i_epochs: usize, d_lr: f32) -> Result<(), String> {
        self.set_training(true);
        if v_data.is_empty() || i_epochs == 0 {
            return Err("invalid_training_args".to_string());
        }
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return Err("invalid_learning_rate".to_string());
        }
        if self.bpe_tokenizer.is_none() {
            return Err("tokenizer_not_set".to_string());
        }

        let v_tokenized_data: Vec<Vec<usize>> = v_data
            .iter()
            .map(|s| self.tokenize(s))
            .collect::<Result<Vec<Vec<usize>>, String>>()?
            .into_iter()
            .filter(|v| v.len() >= 2)
            .collect();

        if v_tokenized_data.is_empty() {
            return Err("no_tokenized_rows".to_string());
        }

        // History:
        // - 2026-02-01: Consolidated training loop into layer.rs within Llm::train while keeping Result based error handling.
        for i_epoch in 0..i_epochs {
            let mut d_total_loss: f32 = 0.0;
            let mut i_used_rows: usize = 0;

            for v_row in v_tokenized_data.iter() {
                let v_input_ids = &v_row[..v_row.len() - 1];
                let v_target_ids = &v_row[1..];

                let mut a_input: Array2<f32> = Array2::zeros((1, v_input_ids.len()));
                let a_row = Array1::from_iter(v_input_ids.iter().map(|&x| x as f32));
                a_input.row_mut(0).assign(&a_row);

                let mut a_act = a_input;
                for layer in self.network.iter_mut() {
                    a_act = layer.forward(&a_act);
                }

                let a_logits = a_act;
                if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
                    continue;
                }

                let a_probs = math::softmax_rows(&a_logits);
                d_total_loss += math::cross_entropy_loss_step(&a_probs, v_target_ids);

                let mut a_grads = math::compute_gradients_step(&a_probs, v_target_ids);
                math::clip_gradients_global_norm(&mut a_grads, 5.0);

                for layer in self.network.iter_mut().rev() {
                    a_grads = layer.backward(&a_grads, d_lr);
                }

                i_used_rows += 1;
            }

            let d_avg_loss = if i_used_rows == 0 {
                0.0
            } else {
                d_total_loss / (i_used_rows as f32).max(1.0)
            };

            println!("Epoch {}: Loss = {:.4}", i_epoch, d_avg_loss);
        }

        Ok(())
    }

    fn greedy_decode(a_probs: &Array2<f32>) -> Vec<usize> {
        a_probs
            .map_axis(Axis(1), |a_row| {
                a_row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .to_vec()
    }

    fn collect_all_parameters_flat(&self) -> Vec<f32> {
        let mut v_params: Vec<f32> = Vec::new();
        for layer in self.network.iter() {
            v_params.extend(layer.get_parameters_flat());
        }
        v_params
    }

    fn assign_all_parameters_flat(&mut self, v_params: &[f32]) -> Result<(), String> {
        let mut i_pos: usize = 0;
        for layer in self.network.iter_mut() {
            let i_used = layer.set_parameters_flat(&v_params[i_pos..])?;
            i_pos += i_used;
        }

        if i_pos != v_params.len() {
            // Extra params are treated as mismatch.
            return Err("checkpoint_params_length_mismatch".to_string());
        }

        Ok(())
    }

    pub fn save_checkpoint(&self, s_path: &str) -> Result<(), String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        let tok = self
            .bpe_tokenizer
            .as_ref()
            .ok_or_else(|| "tokenizer_not_set".to_string())?;

        let tokenizer_cp = tok.to_checkpoint();
        let v_params = self.collect_all_parameters_flat();

        let cp = LlmCheckpoint::new(tokenizer_cp, v_params, MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM);

        let s_json = crate::utils::checkpoint_to_json_ascii(&cp)?;
        crate::utils::write_file_atomic_ascii(s_path, &s_json)?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, s_path: &str) -> Result<(), String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        let s_json = std::fs::read_to_string(s_path).map_err(|_| "checkpoint_read_error".to_string())?;
        let cp = crate::utils::checkpoint_from_json_ascii(&s_json)?;

        cp.validate()?;

        let tok = BpeTokenizer::from_checkpoint(&cp.tokenizer)?;
        self.set_bpe_tokenizer(tok);

        // Verify expected parameter count.
        let i_expected: usize = self.network.iter().map(|l| l.get_parameters_flat().len()).sum();
        if i_expected != cp.v_params.len() {
            return Err("checkpoint_param_count_mismatch".to_string());
        }

        self.assign_all_parameters_flat(&cp.v_params)?;
        Ok(())
    }
}
