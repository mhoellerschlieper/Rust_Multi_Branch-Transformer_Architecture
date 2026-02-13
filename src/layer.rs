// layer.rs
// Description: Model layers and core LLM implementation (forward, backward, train, predict).
//              Implements embeddings, transformer blocks (MHSA + FFN), RMSNorm, output projection,
//              AdamW optimizer, dropout, and checkpoint save/load.
//
//              Adds MTB (multi branch transformer) support via ParallelBlockGroup, which can run
//              branches in parallel (width) and aggregate outputs. Supports branch sequences via
//              TransformerSequence.
//
//              Adds post load diagnostics for ParallelBlockGroup metrics (path starvation, diversity,
//              and additional metrics) and test only fault injection to simulate one dropped path
//              before each predict.
//
// History:
// - 2026-02-01: Consolidate project into 6 files: main, layer, train, math, tokenizer, utils.
// - 2026-02-01: Add checkpoint save and load for model parameters and tokenizer.
// - 2026-02-04: Add robust sampling (temperature, top k, top p) and ensure predict runs eval mode.
// - 2026-02-07: Add MTB ParallelBlockGroup and TransformerSequence support.
// - 2026-02-07: Add MTB diagnostics and test only outage simulation with borrow safe RNG handling.
// - 2026-02-08: Add checkpoint topology spec and rebuild model from topology.
// - 2026-02-13: Add cooperative cancel and training progress events for background training.
// Author: Marcus Schlieper

use std::any::Any;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;

use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::{EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN};
use crate::math;
use crate::tokenizer::{BpeTokenizer, BpeTokenizerCheckpoint, S_EOS};
use crate::utils;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool};
use std::sync::mpsc::Sender;
use std::sync::atomic::Ordering as AtomicOrdering;

pub const DEFAULT_RESIDUAL_DROPOUT_P: f32 = 0.001;

// ----------------------------------------
// Vocab
// ----------------------------------------

#[derive(Clone, Debug)]
pub struct Vocab {
    pub encode: HashMap<String, usize>,
    pub decode: HashMap<usize, String>,
    pub words: Vec<String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new(Self::default_words())
    }
}

impl Vocab {
    pub fn new(v_words: Vec<&str>) -> Self {
        let mut m_encode: HashMap<String, usize> = HashMap::new();
        let mut m_decode: HashMap<usize, String> = HashMap::new();

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
        vec!["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
    }
}

// ----------------------------------------
// Layer trait
// ----------------------------------------

pub trait Layer: Send {
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

// ----------------------------------------
// AdamW
// ----------------------------------------

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

    pub fn step(&mut self, a_params: &mut Array2<f32>, a_grads: &Array2<f32>, d_lr: f32) {
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return;
        }
        if a_params.raw_dim() != a_grads.raw_dim() {
            return;
        }

        self.i_t = self.i_t.saturating_add(1);

        if self.d_weight_decay > 0.0 {
            let d_decay = d_lr * self.d_weight_decay;
            if d_decay.is_finite() && d_decay > 0.0 {
                *a_params = &*a_params - &(d_decay * &*a_params);
            }
        }

        self.m_m = &self.m_m * self.d_beta1 + a_grads * (1.0 - self.d_beta1);

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

// ----------------------------------------
// Dropout (inverted)
// ----------------------------------------

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
            return Array2::zeros(a_x.raw_dim());
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

// ----------------------------------------
// Embeddings
// ----------------------------------------

pub struct Embeddings {
    vocab: Vocab,
    w_embed: Array2<f32>,
    cached_ids: Option<Vec<usize>>,
    optimizer: AdamW,
}

impl Embeddings {
    pub fn new(vocab: Vocab) -> Self {
        let i_vocab = vocab.words.len();
        let mut rng = rand::rng();

        let std = (2.0 / (i_vocab as f32).max(1.0)).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        Self {
            vocab,
            w_embed: Array2::from_shape_fn((i_vocab, EMBEDDING_DIM), |_| normal.sample(&mut rng)),
            cached_ids: None,
            optimizer: AdamW::new((i_vocab, EMBEDDING_DIM), 0.01),
        }
    }
}

impl Layer for Embeddings {
    fn layer_type(&self) -> &str {
        "Embeddings"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
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

        let mut a_grad_w: Array2<f32> = Array2::zeros(self.w_embed.raw_dim());

        for (i_pos, &i_id) in v_ids.iter().enumerate() {
            if i_pos >= a_grads.nrows() || i_id >= a_grad_w.nrows() {
                continue;
            }

            for j in 0..a_grad_w.ncols() {
                let d_add = a_grads[[i_pos, j]];
                a_grad_w[[i_id, j]] = a_grad_w[[i_id, j]] + d_add;
            }
        }

        self.optimizer.step(&mut self.w_embed, &a_grad_w, d_lr);
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
        let a_slice = self
            .w_embed
            .as_slice_mut()
            .ok_or_else(|| "embeddings_not_contiguous".to_string())?;
        for i in 0..i_needed {
            let d = v_params[i];
            a_slice[i] = if d.is_finite() { d } else { 0.0 };
        }
        Ok(i_needed)
    }
}

// ----------------------------------------
// RMSNorm
// ----------------------------------------

pub struct RmsNorm {
    epsilon: f32,
    gamma: Array2<f32>,
    cached_input: Option<Array2<f32>>,
    cached_rms: Option<Array2<f32>>,
    cached_x_hat: Option<Array2<f32>>,
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
        let a_input = match self.cached_input.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let a_rms = match self.cached_rms.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
        let _a_x_hat = match self.cached_x_hat.as_ref() {
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

        let a_grad_gamma = (a_grads * _a_x_hat).sum_axis(Axis(0)).insert_axis(Axis(0));
        let a_grad_x_hat = a_grads * &self.gamma;

        let d_n = i_emb as f32;
        let mut a_grad_x = Array2::zeros(a_input.raw_dim());

        for i in 0..i_seq {
            let d_r = a_rms[[i, 0]].max(1e-12);
            let d_inv_r = 1.0 / d_r;
            let d_inv_r3 = 1.0 / (d_r * d_r * d_r).max(1e-12);

            let mut d_dot: f32 = 0.0;
            for j in 0..i_emb {
                d_dot += a_grad_x_hat[[i, j]] * a_input[[i, j]];
            }

            let d_scale2 = d_dot / d_n.max(1.0);
            for j in 0..i_emb {
                let d_dxhat = a_grad_x_hat[[i, j]];
                let d_x = a_input[[i, j]];
                let d_val = d_dxhat * d_inv_r - d_x * d_scale2 * d_inv_r3;
                a_grad_x[[i, j]] = if d_val.is_finite() { d_val } else { 0.0 };
            }
        }

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
        let a_slice = self
            .gamma
            .as_slice_mut()
            .ok_or_else(|| "rmsnorm_gamma_not_contiguous".to_string())?;
        for i in 0..i_needed {
            let d = v_params[i];
            a_slice[i] = if d.is_finite() { d } else { 0.0 };
        }
        Ok(i_needed)
    }
}


// ----------------------------------------
// FeedForward (with residual + dropout on branch)
// ----------------------------------------

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

        let a_ff_dropped = self.residual_dropout.apply(&a_out);
        a_ff_dropped + a_input
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let a_input = match self.cached_input.as_ref() {
            Some(x) => x,
            None => return a_grads.clone(),
        };
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

        let w1_slice = self.w1.as_slice_mut().ok_or_else(|| "ff_w1_not_contiguous".to_string())?;
        let b1_slice = self.b1.as_slice_mut().ok_or_else(|| "ff_b1_not_contiguous".to_string())?;
        let w2_slice = self.w2.as_slice_mut().ok_or_else(|| "ff_w2_not_contiguous".to_string())?;
        let b2_slice = self.b2.as_slice_mut().ok_or_else(|| "ff_b2_not_contiguous".to_string())?;

        let mut i_pos: usize = 0;
        for i in 0..w1_slice.len() {
            let d = v_params[i_pos];
            w1_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }
        for i in 0..b1_slice.len() {
            let d = v_params[i_pos];
            b1_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }
        for i in 0..w2_slice.len() {
            let d = v_params[i_pos];
            w2_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }
        for i in 0..b2_slice.len() {
            let d = v_params[i_pos];
            b2_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

// ----------------------------------------
// MultiHeadSelfAttention (causal) with residual + dropout
// ----------------------------------------

pub struct MultiHeadSelfAttention {
    i_embedding_dim: usize,
    i_num_heads: usize,
    i_head_dim: usize,

    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,

    cached_input: Option<Array2<f32>>,
    cached_q_all: Option<Array2<f32>>,
    cached_k_all: Option<Array2<f32>>,
    cached_v_all: Option<Array2<f32>>,
    cached_concat: Option<Array2<f32>>,
    cached_weights: Option<Vec<Array2<f32>>>,

    opt_w_q: AdamW,
    opt_w_k: AdamW,
    opt_w_v: AdamW,
    opt_w_o: AdamW,

    residual_dropout: Dropout,
}

impl MultiHeadSelfAttention {
    pub fn new(i_embedding_dim: usize, i_num_heads: usize) -> Self {
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

    fn apply_causal_mask_inplace(a_scores: &mut Array2<f32>) {
        let i_seq_len = a_scores.nrows();
        for i in 0..i_seq_len {
            for j in (i + 1)..i_seq_len {
                a_scores[[i, j]] = f32::NEG_INFINITY;
            }
        }
    }

    fn softmax_backward(a_softmax: &Array2<f32>, a_grad_out: &Array2<f32>) -> Array2<f32> {
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

        let mut a_out = Array2::zeros((i_seq_len, self.i_embedding_dim));
        for i_h in 0..self.i_num_heads {
            let i_start = i_h * self.i_head_dim;
            let i_end = i_start + self.i_head_dim;
            let mut a_slice = a_out.slice_mut(ndarray::s![.., i_start..i_end]);
            a_slice.assign(&v_heads[i_h]);
        }

        Ok(a_out)
    }

    fn attention_head_forward(&self, a_q: &Array2<f32>, a_k: &Array2<f32>, a_v: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let d_scale = (self.i_head_dim as f32).sqrt().max(1e-12);

        let mut a_scores = a_q.dot(&a_k.t()) / d_scale;
        Self::apply_causal_mask_inplace(&mut a_scores);

        let a_weights = math::softmax_rows(&a_scores);
        let a_out = a_weights.dot(a_v);

        (a_out, a_weights)
    }
}

impl Layer for MultiHeadSelfAttention {
    fn layer_type(&self) -> &str {
        "MultiHeadSelfAttention"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }
        if a_input.ncols() != self.i_embedding_dim {
            return Array2::zeros((0, 0));
        }

        self.cached_input = Some(a_input.clone());

        let a_q_all = a_input.dot(&self.w_q);
        let a_k_all = a_input.dot(&self.w_k);
        let a_v_all = a_input.dot(&self.w_v);

        self.cached_q_all = Some(a_q_all.clone());
        self.cached_k_all = Some(a_k_all.clone());
        self.cached_v_all = Some(a_v_all.clone());

        let v_q = match self.split_heads(&a_q_all) {
            Ok(v) => v,
            Err(_) => return Array2::zeros((0, 0)),
        };
        let v_k = match self.split_heads(&a_k_all) {
            Ok(v) => v,
            Err(_) => return Array2::zeros((0, 0)),
        };
        let v_v = match self.split_heads(&a_v_all) {
            Ok(v) => v,
            Err(_) => return Array2::zeros((0, 0)),
        };

        let mut v_head_out: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_weights: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);

        for i_h in 0..self.i_num_heads {
            let (a_h_out, a_w) = self.attention_head_forward(&v_q[i_h], &v_k[i_h], &v_v[i_h]);
            v_head_out.push(a_h_out);
            v_weights.push(a_w);
        }

        self.cached_weights = Some(v_weights);

        let a_concat = match self.concat_heads(&v_head_out) {
            Ok(a) => a,
            Err(_) => return Array2::zeros((0, 0)),
        };
        self.cached_concat = Some(a_concat.clone());

        let a_proj = a_concat.dot(&self.w_o);
        let a_proj_dropped = self.residual_dropout.apply(&a_proj);

        a_proj_dropped + a_input
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
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

        let a_grad_proj = a_grads;

        let a_grad_w_o = a_concat.t().dot(a_grad_proj);
        let a_grad_concat = a_grad_proj.dot(&self.w_o.t());

        let v_grad_head_out = match self.split_heads(&a_grad_concat) {
            Ok(v) => v,
            Err(_) => return a_grads.clone(),
        };

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

        let mut v_grad_q: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_grad_k: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);
        let mut v_grad_v: Vec<Array2<f32>> = Vec::with_capacity(self.i_num_heads);

        let d_scale = (self.i_head_dim as f32).sqrt().max(1e-12);
        let i_seq_len = a_input.nrows();

        for i_h in 0..self.i_num_heads {
            let a_q = &v_q[i_h];
            let a_k = &v_k[i_h];
            let a_v = &v_v[i_h];
            let a_w = &v_weights[i_h];
            let a_grad_h_out = &v_grad_head_out[i_h];

            let a_grad_w = a_grad_h_out.dot(&a_v.t());
            let a_grad_v_h = a_w.t().dot(a_grad_h_out);

            let mut a_grad_scores = Self::softmax_backward(a_w, &a_grad_w);

            for i in 0..i_seq_len {
                for j in (i + 1)..i_seq_len {
                    a_grad_scores[[i, j]] = 0.0;
                }
            }

            let a_grad_q_h = a_grad_scores.dot(a_k) / d_scale;
            let a_grad_k_h = a_grad_scores.t().dot(a_q) / d_scale;

            v_grad_q.push(a_grad_q_h);
            v_grad_k.push(a_grad_k_h);
            v_grad_v.push(a_grad_v_h);
        }

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

        let a_grad_w_q = a_input.t().dot(&a_grad_q_all);
        let a_grad_w_k = a_input.t().dot(&a_grad_k_all);
        let a_grad_w_v = a_input.t().dot(&a_grad_v_all);

        let a_grad_x_from_q = a_grad_q_all.dot(&self.w_q.t());
        let a_grad_x_from_k = a_grad_k_all.dot(&self.w_k.t());
        let a_grad_x_from_v = a_grad_v_all.dot(&self.w_v.t());

        let a_grad_input_total = a_grads.clone() + a_grad_x_from_q + a_grad_x_from_k + a_grad_x_from_v;

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

        let q_slice = self.w_q.as_slice_mut().ok_or_else(|| "mhsa_w_q_not_contiguous".to_string())?;
        let k_slice = self.w_k.as_slice_mut().ok_or_else(|| "mhsa_w_k_not_contiguous".to_string())?;
        let v_slice = self.w_v.as_slice_mut().ok_or_else(|| "mhsa_w_v_not_contiguous".to_string())?;
        let o_slice = self.w_o.as_slice_mut().ok_or_else(|| "mhsa_w_o_not_contiguous".to_string())?;

        let mut i_pos: usize = 0;
        for i in 0..q_slice.len() {
            let d = v_params[i_pos];
            q_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }
        for i in 0..k_slice.len() {
            let d = v_params[i_pos];
            k_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }
        for i in 0..v_slice.len() {
            let d = v_params[i_pos];
            v_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }
        for i in 0..o_slice.len() {
            let d = v_params[i_pos];
            o_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

// ----------------------------------------
// TransformerBlock (Norm, MHSA, Norm, FFN)
// ----------------------------------------

pub struct TransformerBlock {
    attention: MultiHeadSelfAttention,
    feed_forward: FeedForward,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl TransformerBlock {
    pub fn new(i_embedding_dim: usize, i_hidden_dim: usize) -> Self {
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
        let a_n1 = self.norm1.forward(&a_attn);
        let a_ff = self.feed_forward.forward(&a_n1);
        self.norm2.forward(&a_ff)
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

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

// ----------------------------------------
// TransformerSequence (sequential composition of blocks)
// ----------------------------------------

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

// ----------------------------------------
// OutputProjection
// ----------------------------------------

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
        let normal = Normal::new(0.0, std).unwrap();

        Self {
            w_out: Array2::from_shape_fn((i_embedding_dim, i_vocab_size), |_| normal.sample(&mut rng)),
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
        let a_input = self.cached_input.as_ref().expect("forward must be run first");

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

        let w_slice = self.w_out.as_slice_mut().ok_or_else(|| "out_w_not_contiguous".to_string())?;
        let b_slice = self.b_out.as_slice_mut().ok_or_else(|| "out_b_not_contiguous".to_string())?;

        let mut i_pos: usize = 0;
        for i in 0..w_slice.len() {
            let d = v_params[i_pos];
            w_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }
        for i in 0..b_slice.len() {
            let d = v_params[i_pos];
            b_slice[i] = if d.is_finite() { d } else { 0.0 };
            i_pos += 1;
        }

        Ok(i_needed)
    }
}

// ----------------------------------------
// MTB diagnostics structs and helpers
// ----------------------------------------

#[derive(Clone, Debug)]
pub struct ParallelBlockGroupMetrics {
    pub i_num_paths: usize,
    pub i_num_samples: usize,
    pub d_path_starvation_index: f32,
    pub d_diversity_cosine_distance_mean: f32,
    pub d_effective_num_paths: f32,
    pub d_gini_concentration: f32,
    pub d_top1_share: f32,
    pub d_margin_top1_top2: f32,
    pub d_output_energy_cv: f32,
    pub d_branch_correlation_mean: f32,
}

impl ParallelBlockGroupMetrics {
    pub fn to_ascii_report_lines(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        v.push("=== ParallelBlockGroup diagnostics ===".to_string());
        v.push(format!("num_paths: {}", self.i_num_paths));
        v.push(format!("num_samples: {}", self.i_num_samples));
        v.push(format!("path_starvation_index: {:.6}", self.d_path_starvation_index));
        v.push(format!(
            "diversity_cosine_distance_mean: {:.6}",
            self.d_diversity_cosine_distance_mean
        ));
        v.push(format!("effective_num_paths: {:.6}", self.d_effective_num_paths));
        v.push(format!("gini_concentration: {:.6}", self.d_gini_concentration));
        v.push(format!("top1_share: {:.6}", self.d_top1_share));
        v.push(format!("margin_top1_top2: {:.6}", self.d_margin_top1_top2));
        v.push(format!("output_energy_cv: {:.6}", self.d_output_energy_cv));
        v.push(format!(
            "branch_correlation_mean: {:.6}",
            self.d_branch_correlation_mean
        ));
        v
    }
}

fn sanitize_f32(d_x: f32) -> f32 {
    if d_x.is_finite() { d_x } else { 0.0 }
}

fn clamp_prob(d_x: f32) -> f32 {
    if !d_x.is_finite() {
        return 0.0;
    }
    if d_x < 0.0 {
        0.0
    } else if d_x > 1.0 {
        1.0
    } else {
        d_x
    }
}

fn entropy_nat(v_p: &[f32]) -> f32 {
    let mut d_h: f32 = 0.0;
    for &p in v_p.iter() {
        let d_p = clamp_prob(p);
        if d_p > 0.0 {
            d_h -= d_p * d_p.max(1e-12).ln();
        }
    }
    sanitize_f32(d_h)
}

fn normalize_distribution(v_x: &[f32]) -> Vec<f32> {
    if v_x.is_empty() {
        return Vec::new();
    }
    let mut d_sum: f32 = 0.0;
    for &d in v_x.iter() {
        d_sum += sanitize_f32(d).max(0.0);
    }
    if !d_sum.is_finite() || d_sum <= 0.0 {
        let d_u = 1.0 / (v_x.len() as f32).max(1.0);
        return vec![d_u; v_x.len()];
    }
    v_x.iter()
        .map(|&d| sanitize_f32(d).max(0.0) / d_sum)
        .collect()
}

fn gini_coefficient(v_p: &[f32]) -> f32 {
    let i_n = v_p.len();
    if i_n == 0 {
        return 0.0;
    }
    let mut v = v_p.iter().map(|&x| clamp_prob(x)).collect::<Vec<f32>>();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let d_n = i_n as f32;

    let mut d_sum: f32 = 0.0;
    for (i, &p) in v.iter().enumerate() {
        let d_i = i as f32;
        let d_weight = (d_n - d_i - 0.5) / d_n.max(1.0);
        d_sum += p * d_weight;
    }

    let d_g = 1.0 - 2.0 * d_sum;
    if d_g.is_finite() { d_g.clamp(0.0, 1.0) } else { 0.0 }
}

fn cosine_similarity(v_a: &[f32], v_b: &[f32]) -> f32 {
    if v_a.is_empty() || v_b.is_empty() || v_a.len() != v_b.len() {
        return 0.0;
    }
    let mut d_dot: f32 = 0.0;
    let mut d_na: f32 = 0.0;
    let mut d_nb: f32 = 0.0;

    for i in 0..v_a.len() {
        let d_x = sanitize_f32(v_a[i]);
        let d_y = sanitize_f32(v_b[i]);
        d_dot += d_x * d_y;
        d_na += d_x * d_x;
        d_nb += d_y * d_y;
    }

    let d_den = (d_na.sqrt() * d_nb.sqrt()).max(1e-12);
    let d_cos = d_dot / d_den;

    if d_cos.is_finite() { d_cos.clamp(-1.0, 1.0) } else { 0.0 }
}

fn flatten_array2(a_x: &Array2<f32>) -> Vec<f32> {
    a_x.iter().map(|&d| sanitize_f32(d)).collect()
}

fn mean_square_energy(a_x: &Array2<f32>) -> f32 {
    if a_x.len() == 0 {
        return 0.0;
    }
    let mut d_sum: f32 = 0.0;
    let mut d_cnt: f32 = 0.0;
    for &d in a_x.iter() {
        let d_v = sanitize_f32(d);
        d_sum += d_v * d_v;
        d_cnt += 1.0;
    }
    let d_m = d_sum / d_cnt.max(1.0);
    sanitize_f32(d_m)
}

fn coeff_of_variation(v_x: &[f32]) -> f32 {
    if v_x.is_empty() {
        return 0.0;
    }
    let mut d_mean: f32 = 0.0;
    for &d in v_x.iter() {
        d_mean += sanitize_f32(d);
    }
    d_mean /= (v_x.len() as f32).max(1.0);

    if !d_mean.is_finite() || d_mean.abs() < 1e-12 {
        return 0.0;
    }

    let mut d_var: f32 = 0.0;
    for &d in v_x.iter() {
        let d_v = sanitize_f32(d);
        let d_diff = d_v - d_mean;
        d_var += d_diff * d_diff;
    }
    d_var /= (v_x.len() as f32).max(1.0);

    let d_std = d_var.sqrt();
    let d_cv = d_std / d_mean.abs();

    if d_cv.is_finite() { d_cv } else { 0.0 }
}

// ----------------------------------------
// ParallelBlockGroup (MTB width layer) with diagnostics and test only outage injection
// ----------------------------------------

pub struct ParallelBlockGroup {
    v_branches: Vec<Box<dyn Layer>>,
    d_equal_weight: f32,

    // Test only fault injection:
    b_fault_injection_enabled: bool,
    opt_fault_drop_branch_idx: Option<usize>,
}

// layer.rs
// Description: ParallelBlockGroup implementation with parallel forward and backward execution
//              of branch layers using Rayon, including fault injection controls, training
//              configuration propagation, and MTB diagnostics input metrics computation.
//
// History:
// - 2026-02-07: Add MTB ParallelBlockGroup with fault injection and diagnostics.
// - 2026-02-11: Parallelize forward and backward branch execution via Rayon for training and predict.
// Author: Marcus Schlieper


impl ParallelBlockGroup {
    pub fn new(v_branches: Vec<Box<dyn Layer>>) -> Result<Self, String> {
        if v_branches.is_empty() {
            return Err("parallel_block_group_empty".to_string());
        }

        let d_w = 1.0f32 / (v_branches.len() as f32).max(1.0);

        Ok(Self {
            v_branches,
            d_equal_weight: d_w,
            b_fault_injection_enabled: false,
            opt_fault_drop_branch_idx: None,
        })
    }

    pub fn forward_with_availability_mask(
        &mut self,
        a_input: &Array2<f32>,
        v_active_mask: &[bool],
    ) -> Array2<f32> {
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }
        if v_active_mask.len() != self.v_branches.len() {
            return Array2::zeros((0, 0));
        }

        // Parallel forward for active branches only.
        let v_outs: Vec<Option<Array2<f32>>> = self
            .v_branches
            .par_iter_mut()
            .enumerate()
            .map(|(i_idx, br)| {
                if !v_active_mask[i_idx] {
                    return None;
                }
                let a_y = br.forward(a_input);
                if a_y.nrows() == 0 || a_y.ncols() == 0 {
                    None
                } else {
                    Some(a_y)
                }
            })
            .collect();

        let mut opt_sum: Option<Array2<f32>> = None;
        let mut i_used: usize = 0;

        for opt_a in v_outs.into_iter() {
            let a_y = match opt_a {
                Some(a) => a,
                None => continue,
            };
            match &mut opt_sum {
                None => {
                    opt_sum = Some(a_y);
                    i_used = i_used.saturating_add(1);
                }
                Some(a_acc) => {
                    if a_acc.raw_dim() != a_y.raw_dim() {
                        return Array2::zeros((0, 0));
                    }
                    *a_acc = &*a_acc + &a_y;
                    i_used = i_used.saturating_add(1);
                }
            }
        }

        let mut a_out = match opt_sum {
            None => Array2::zeros((0, 0)),
            Some(a) => a,
        };

        // Partial aggregation: average across active branches.
        let d_w = 1.0f32 / (i_used as f32).max(1.0);
        if d_w.is_finite() && d_w > 0.0 {
            a_out.mapv_inplace(|x| x * d_w);
        }

        a_out
    }
    pub fn backward_with_availability_mask(
        &mut self,
        a_grads: &Array2<f32>,
        d_lr: f32,
        v_active_mask: &[bool],
        opt_inv_participation_scale: Option<&[f32]>,
    ) -> Array2<f32> {
        if a_grads.nrows() == 0 || a_grads.ncols() == 0 {
            return a_grads.clone();
        }
        if v_active_mask.len() != self.v_branches.len() {
            return a_grads.clone();
        }

        let v_scales: Option<Vec<f32>> = opt_inv_participation_scale.map(|v| v.to_vec());

        let v_grad_x: Vec<Array2<f32>> = self
            .v_branches
            .par_iter_mut()
            .enumerate()
            .map(|(i_idx, br)| {
                if !v_active_mask[i_idx] {
                    return Array2::zeros(a_grads.raw_dim());
                }

                // Optionally compensate 1/p_i for unbiasedness in expectation.
                let mut a_g = a_grads.clone();
                if let Some(vs) = v_scales.as_ref() {
                    if i_idx < vs.len() {
                        let d_s = vs[i_idx];
                        if d_s.is_finite() && d_s > 0.0 {
                            a_g.mapv_inplace(|x| x * d_s);
                        }
                    }
                }

                br.backward(&a_g, d_lr)
            })
            .collect();

        let mut a_grad_x_total = Array2::zeros(a_grads.raw_dim());
        for a_gx in v_grad_x.into_iter() {
            if a_gx.raw_dim() != a_grad_x_total.raw_dim() {
                return a_grads.clone();
            }
            a_grad_x_total = a_grad_x_total + a_gx;
        }

        a_grad_x_total
    }


    // Return branch layer types for diagnostics and topology display.
    // This preserves encapsulation by not exposing v_branches directly.
    pub fn branch_layer_types_ascii(&self) -> Vec<String> {
        let mut v_types: Vec<String> = Vec::with_capacity(self.v_branches.len());
        for br in self.v_branches.iter() {
            v_types.push(br.layer_type().to_string());
        }
        v_types
    }

    pub fn num_branches(&self) -> usize {
        self.v_branches.len()
    }

    pub fn set_fault_injection_enabled(&mut self, b_enabled: bool) {
        self.b_fault_injection_enabled = b_enabled;
        if !b_enabled {
            self.opt_fault_drop_branch_idx = None;
        }
    }

    pub fn set_fault_drop_branch_idx(&mut self, opt_idx: Option<usize>) {
        if let Some(i_idx) = opt_idx {
            if i_idx >= self.v_branches.len() {
                self.opt_fault_drop_branch_idx = None;
                return;
            }
            self.opt_fault_drop_branch_idx = Some(i_idx);
        } else {
            self.opt_fault_drop_branch_idx = None;
        }
    }

    pub fn set_training(&mut self, b_training: bool) {
        for br in self.v_branches.iter_mut() {
            if let Some(tb) = br.as_any_mut().and_then(|a| a.downcast_mut::<TransformerBlock>()) {
                tb.set_training(b_training);
                continue;
            }
            if let Some(ts) = br
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<TransformerSequence>())
            {
                ts.set_training(b_training);
                continue;
            }
        }
    }

    pub fn set_residual_dropout_p(&mut self, d_p: f32) {
        for br in self.v_branches.iter_mut() {
            if let Some(tb) = br.as_any_mut().and_then(|a| a.downcast_mut::<TransformerBlock>()) {
                tb.set_residual_dropout_p(d_p);
                continue;
            }
            if let Some(ts) = br
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<TransformerSequence>())
            {
                ts.set_residual_dropout_p(d_p);
                continue;
            }
        }
    }

    pub fn reseed_dropout(&mut self, u64_seed: u64) {
        for (i_idx, br) in self.v_branches.iter_mut().enumerate() {
            let u64_mix = (i_idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            let u64_branch_seed = u64_seed ^ u64_mix;

            if let Some(tb) = br.as_any_mut().and_then(|a| a.downcast_mut::<TransformerBlock>()) {
                tb.reseed_dropout(u64_branch_seed);
                continue;
            }
            if let Some(ts) = br
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<TransformerSequence>())
            {
                ts.reseed_dropout(u64_branch_seed);
                continue;
            }
        }
    }

    // Sequential helper retained for diagnostics and for cases where deterministic
    // layer execution order is desired.
    pub fn forward_branches(&mut self, a_input: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut v_out: Vec<Array2<f32>> = Vec::with_capacity(self.v_branches.len());
        for br in self.v_branches.iter_mut() {
            v_out.push(br.forward(a_input));
        }
        v_out
    }

    // Compute MTB diagnostics metrics from a set of inputs that already represent the activation
    // directly before this ParallelBlockGroup.
    pub fn compute_metrics_from_inputs(
        &mut self,
        v_inputs: &[Array2<f32>],
    ) -> Result<ParallelBlockGroupMetrics, String> {
        let i_k = self.v_branches.len();
        if i_k == 0 {
            return Err("parallel_block_group_no_paths".to_string());
        }
        if v_inputs.is_empty() {
            return Err("parallel_block_group_metrics_empty_inputs".to_string());
        }

        let mut d_psi_sum: f32 = 0.0;
        let mut d_div_sum: f32 = 0.0;
        let mut d_eff_paths_sum: f32 = 0.0;
        let mut d_gini_sum: f32 = 0.0;
        let mut d_top1_sum: f32 = 0.0;
        let mut d_margin_sum: f32 = 0.0;
        let mut d_corr_sum: f32 = 0.0;
        let mut v_energy_all: Vec<f32> = Vec::new();
        let mut i_used: usize = 0;

        for a_in in v_inputs.iter() {
            if a_in.nrows() == 0 || a_in.ncols() == 0 {
                continue;
            }

            // NOTE: Uses sequential helper to keep metric computation stable and comparable.
            // The forward of the group itself may be parallel, but diagnostics typically aims
            // at interpretability rather than maximum throughput.
            let v_branch_out = self.forward_branches(a_in);

            let mut v_scores: Vec<f32> = Vec::with_capacity(i_k);
            let mut v_energy: Vec<f32> = Vec::with_capacity(i_k);
            let mut v_flat: Vec<Vec<f32>> = Vec::with_capacity(i_k);

            for a_y in v_branch_out.iter() {
                let d_e = mean_square_energy(a_y);
                v_scores.push(d_e);
                v_energy.push(d_e);
                v_flat.push(flatten_array2(a_y));
            }

            let a_scores_row = Array2::from_shape_vec((1, v_scores.len()), v_scores.clone())
                .map_err(|_| "parallel_block_group_metrics_shape_error".to_string())?;
            let a_p = math::softmax_rows(&a_scores_row);

            let mut v_p: Vec<f32> = Vec::with_capacity(i_k);
            for i in 0..i_k {
                v_p.push(clamp_prob(a_p[[0, i]]));
            }
            v_p = normalize_distribution(&v_p);

            let d_h = entropy_nat(&v_p);
            let d_h_max = (i_k as f32).max(1.0).ln().max(1e-12);
            let d_h_norm = (d_h / d_h_max).clamp(0.0, 1.0);
            let d_psi = 1.0 - d_h_norm;

            let d_eff = d_h.exp().clamp(1.0, i_k as f32);
            let d_gini = gini_coefficient(&v_p);

            let mut v_sorted = v_p.clone();
            v_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            let d_top1 = *v_sorted.get(0).unwrap_or(&0.0);
            let d_top2 = *v_sorted.get(1).unwrap_or(&0.0);
            let d_margin = (d_top1 - d_top2).max(0.0);

            let mut d_dist_sum: f32 = 0.0;
            let mut d_sim_sum: f32 = 0.0;
            let mut i_pairs: usize = 0;

            for i in 0..i_k {
                for j in (i + 1)..i_k {
                    let v_a = &v_flat[i];
                    let v_b = &v_flat[j];
                    if v_a.len() != v_b.len() || v_a.is_empty() {
                        continue;
                    }
                    let d_sim = cosine_similarity(v_a, v_b);
                    let d_dist = 1.0 - d_sim;
                    d_sim_sum += d_sim;
                    d_dist_sum += d_dist;
                    i_pairs = i_pairs.saturating_add(1);
                }
            }

            let d_div = if i_pairs == 0 {
                0.0
            } else {
                d_dist_sum / (i_pairs as f32).max(1.0)
            };

            let d_corr = if i_pairs == 0 {
                0.0
            } else {
                d_sim_sum / (i_pairs as f32).max(1.0)
            };

            v_energy_all.extend(v_energy.into_iter());

            d_psi_sum += d_psi;
            d_div_sum += d_div;
            d_eff_paths_sum += d_eff;
            d_gini_sum += d_gini;
            d_top1_sum += d_top1;
            d_margin_sum += d_margin;
            d_corr_sum += d_corr;

            i_used = i_used.saturating_add(1);
        }

        if i_used == 0 {
            return Err("parallel_block_group_metrics_no_valid_samples".to_string());
        }

        let d_n = (i_used as f32).max(1.0);
        let d_energy_cv = coeff_of_variation(&v_energy_all);

        Ok(ParallelBlockGroupMetrics {
            i_num_paths: i_k,
            i_num_samples: i_used,
            d_path_starvation_index: sanitize_f32(d_psi_sum / d_n),
            d_diversity_cosine_distance_mean: sanitize_f32(d_div_sum / d_n),
            d_effective_num_paths: sanitize_f32(d_eff_paths_sum / d_n),
            d_gini_concentration: sanitize_f32(d_gini_sum / d_n),
            d_top1_share: sanitize_f32(d_top1_sum / d_n),
            d_margin_top1_top2: sanitize_f32(d_margin_sum / d_n),
            d_output_energy_cv: sanitize_f32(d_energy_cv),
            d_branch_correlation_mean: sanitize_f32(d_corr_sum / d_n),
        })
    }
}


impl Layer for ParallelBlockGroup {
    fn layer_type(&self) -> &str {
        "ParallelBlockGroup"
    }

    fn forward(&mut self, a_input: &Array2<f32>) -> Array2<f32> {
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return a_input.clone();
        }

        let i_k_total = self.v_branches.len();
        if i_k_total == 0 {
            return Array2::zeros((0, 0));
        }

        let opt_drop: Option<usize> = if self.b_fault_injection_enabled {
            self.opt_fault_drop_branch_idx
        } else {
            None
        };

        // Parallel branch execution.
        let v_outs: Vec<Option<Array2<f32>>> = self
            .v_branches
            .par_iter_mut()
            .enumerate()
            .map(|(i_idx, br)| {
                if let Some(i_drop) = opt_drop {
                    if i_idx == i_drop {
                        return None;
                    }
                }

                let a_y = br.forward(a_input);
                if a_y.nrows() == 0 || a_y.ncols() == 0 {
                    None
                } else {
                    Some(a_y)
                }
            })
            .collect();

        // Sequential aggregation (stable behavior and clear shape checks).
        let mut opt_sum: Option<Array2<f32>> = None;
        let mut i_used_branches: usize = 0;

        for opt_a in v_outs.into_iter() {
            let a_y = match opt_a {
                Some(a) => a,
                None => continue,
            };

            match &mut opt_sum {
                None => {
                    opt_sum = Some(a_y);
                    i_used_branches = i_used_branches.saturating_add(1);
                }
                Some(a_acc) => {
                    if a_acc.raw_dim() != a_y.raw_dim() {
                        return Array2::zeros((0, 0));
                    }
                    *a_acc = &*a_acc + &a_y;
                    i_used_branches = i_used_branches.saturating_add(1);
                }
            }
        }

        let mut a_out = match opt_sum {
            None => Array2::zeros((0, 0)),
            Some(a) => a,
        };

        let d_w = 1.0f32 / (i_used_branches as f32).max(1.0);
        if d_w.is_finite() && d_w > 0.0 {
            a_out.mapv_inplace(|x| x * d_w);
        }

        a_out
    }

    fn backward(&mut self, a_grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        if a_grads.nrows() == 0 || a_grads.ncols() == 0 {
            return a_grads.clone();
        }

        // Parallel branch backward.
        let v_grad_x: Vec<Array2<f32>> = self
            .v_branches
            .par_iter_mut()
            .map(|br| br.backward(a_grads, d_lr))
            .collect();

        // Sequential sum with strict shape validation.
        let mut a_grad_x_total = Array2::zeros(a_grads.raw_dim());
        for a_gx in v_grad_x.into_iter() {
            if a_gx.raw_dim() != a_grad_x_total.raw_dim() {
                return a_grads.clone();
            }
            a_grad_x_total = a_grad_x_total + a_gx;
        }

        a_grad_x_total
    }

    fn parameters(&self) -> usize {
        self.v_branches.iter().map(|b| b.parameters()).sum()
    }

    fn get_parameters_flat(&self) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::new();
        for b in self.v_branches.iter() {
            v.extend(b.get_parameters_flat());
        }
        v
    }

    fn set_parameters_flat(&mut self, v_params: &[f32]) -> Result<usize, String> {
        let mut i_used: usize = 0;
        for b in self.v_branches.iter_mut() {
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
// Topology spec for checkpoint
// ---------------------------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct llm_topology_spec {
    pub v_layers: Vec<llm_layer_spec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "s_type")]
pub enum llm_layer_spec {
    embeddings,
    transformer_block {
        i_embedding_dim: usize,
        i_hidden_dim: usize,
        i_num_heads: usize,
    },
    transformer_sequence {
        v_blocks: Vec<llm_layer_spec>,
    },
    parallel_block_group {
        v_branches: Vec<llm_layer_spec>,
    },
    output_projection {
        i_embedding_dim: usize,
    },
}

// ---------------------------
// Checkpoint format (version bump)
// ---------------------------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct llm_checkpoint_v2 {
    pub s_magic: String,
    pub s_version: String,
    pub i_max_seq_len: usize,
    pub i_embedding_dim: usize,
    pub i_hidden_dim: usize,
    pub tokenizer: crate::tokenizer::BpeTokenizerCheckpoint,
    pub topology: llm_topology_spec,
    pub v_params: Vec<f32>,
}

impl llm_checkpoint_v2 {
    pub fn new(
        tokenizer: crate::tokenizer::BpeTokenizerCheckpoint,
        topology: llm_topology_spec,
        v_params: Vec<f32>,
        i_max_seq_len: usize,
        i_embedding_dim: usize,
        i_hidden_dim: usize,
    ) -> Self {
        Self {
            s_magic: "EXCHAT_LLM_CHECKPOINT".to_string(),
            s_version: "2".to_string(),
            i_max_seq_len,
            i_embedding_dim,
            i_hidden_dim,
            tokenizer,
            topology,
            v_params,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.s_magic != "EXCHAT_LLM_CHECKPOINT" {
            return Err("checkpoint_magic_mismatch".to_string());
        }
        if self.s_version != "2" {
            return Err("checkpoint_version_unsupported".to_string());
        }
        if self.i_max_seq_len != crate::MAX_SEQ_LEN {
            return Err("checkpoint_max_seq_len_mismatch".to_string());
        }
        if self.i_embedding_dim != crate::EMBEDDING_DIM {
            return Err("checkpoint_embedding_dim_mismatch".to_string());
        }
        if self.i_hidden_dim != crate::HIDDEN_DIM {
            return Err("checkpoint_hidden_dim_mismatch".to_string());
        }
        if self.topology.v_layers.is_empty() {
            return Err("checkpoint_topology_empty".to_string());
        }
        if self.v_params.is_empty() {
            return Err("checkpoint_empty_params".to_string());
        }
        Ok(())
    }
}

// ---------------------------
// Topology export from actual network (SAVE)
// ---------------------------
fn export_layer_spec_from_layer(layer: &Box<dyn crate::layer::Layer>) -> Result<llm_layer_spec, String> {
    // NOTE:
    // - The project uses layer_type() strings already.
    // - This avoids needing as_any() for immutable downcasts.
    // - For composite layers, we add explicit "export" helper methods below.
    let s_t = layer.layer_type();

    if s_t == "Embeddings" {
        return Ok(llm_layer_spec::embeddings);
    }

    if s_t == "TransformerBlock" {
        // The current TransformerBlock::new hardcodes num_heads=4 in this codebase.
        return Ok(llm_layer_spec::transformer_block {
            i_embedding_dim: crate::EMBEDDING_DIM,
            i_hidden_dim: crate::HIDDEN_DIM,
            i_num_heads: 4,
        });
    }

    if s_t == "TransformerSequence" {
        // Requires export support implemented on TransformerSequence via as_any_mut downcast.
        // Since we only have &Box<dyn Layer>, we cannot mutably downcast here.
        // Therefore, we export topology via Llm::export_topology_spec() where we can iterate
        // over &mut self.network and use as_any_mut safely.
        return Err("export_layer_spec_requires_llm_context_transformer_sequence".to_string());
    }

    if s_t == "ParallelBlockGroup" {
        return Err("export_layer_spec_requires_llm_context_parallel_block_group".to_string());
    }

    if s_t == "OutputProjection" {
        return Ok(llm_layer_spec::output_projection {
            i_embedding_dim: crate::EMBEDDING_DIM,
        });
    }

    Err("export_layer_spec_unknown_layer_type".to_string())
}

// ---------------------------
// Composite layer export helpers
// ---------------------------
impl crate::layer::TransformerSequence {
    pub fn export_spec(&self) -> llm_layer_spec {
        // Each internal element is a TransformerBlock in this codebase.
        let mut v_blocks: Vec<llm_layer_spec> = Vec::new();
        for _ in self.v_blocks.iter() {
            v_blocks.push(llm_layer_spec::transformer_block {
                i_embedding_dim: crate::EMBEDDING_DIM,
                i_hidden_dim: crate::HIDDEN_DIM,
                i_num_heads: 4,
            });
        }
        llm_layer_spec::transformer_sequence { v_blocks }
    }
}

impl crate::layer::ParallelBlockGroup {
    pub fn export_spec(&self) -> Result<llm_layer_spec, String> {
        let mut v_branches: Vec<llm_layer_spec> = Vec::new();
        for br in self.v_branches.iter() {
            let s_t = br.layer_type();
            if s_t == "TransformerBlock" {
                v_branches.push(llm_layer_spec::transformer_block {
                    i_embedding_dim: crate::EMBEDDING_DIM,
                    i_hidden_dim: crate::HIDDEN_DIM,
                    i_num_heads: 4,
                });
            } else if s_t == "TransformerSequence" {
                // We cannot immutable-downcast br to TransformerSequence without as_any().
                // However, in this project, branches are built as TransformerSequence,
                // and we can reconstruct sequence length by parameterization if we store it.
                //
                // Minimal safe approach: require branch to provide export via known type path
                // using as_any_mut in Llm::export_topology_spec().
                return Err("parallel_block_group_export_requires_llm_context".to_string());
            } else {
                return Err("parallel_block_group_export_branch_type_unsupported".to_string());
            }
        }
        Ok(llm_layer_spec::parallel_block_group { v_branches })
    }
}

// ---------------------------
// Topology reconstruction (LOAD)
// ---------------------------
fn build_layer_from_spec(
    spec: &llm_layer_spec,
    vocab: &crate::layer::Vocab,
) -> Result<Box<dyn crate::layer::Layer>, String> {
    match spec {
        llm_layer_spec::embeddings => Ok(Box::new(crate::layer::Embeddings::new(vocab.clone()))),

        llm_layer_spec::transformer_block {
            i_embedding_dim,
            i_hidden_dim,
            i_num_heads,
        } => {
            if *i_embedding_dim != crate::EMBEDDING_DIM {
                return Err("topology_embedding_dim_mismatch".to_string());
            }
            if *i_hidden_dim != crate::HIDDEN_DIM {
                return Err("topology_hidden_dim_mismatch".to_string());
            }
            if *i_num_heads != 4 {
                return Err("topology_num_heads_unsupported".to_string());
            }
            Ok(Box::new(crate::layer::TransformerBlock::new(*i_embedding_dim, *i_hidden_dim)))
        }

        llm_layer_spec::transformer_sequence { v_blocks } => {
            if v_blocks.is_empty() {
                return Err("topology_transformer_sequence_empty".to_string());
            }
            let mut v_tb: Vec<crate::layer::TransformerBlock> = Vec::new();
            for b in v_blocks.iter() {
                match b {
                    llm_layer_spec::transformer_block {
                        i_embedding_dim,
                        i_hidden_dim,
                        i_num_heads,
                    } => {
                        if *i_num_heads != 4 {
                            return Err("topology_num_heads_unsupported".to_string());
                        }
                        v_tb.push(crate::layer::TransformerBlock::new(*i_embedding_dim, *i_hidden_dim));
                    }
                    _ => return Err("topology_transformer_sequence_allows_only_blocks".to_string()),
                }
            }
            let seq = crate::layer::TransformerSequence::new(v_tb)?;
            Ok(Box::new(seq))
        }

        llm_layer_spec::parallel_block_group { v_branches } => {
            if v_branches.is_empty() {
                return Err("topology_parallel_block_group_empty".to_string());
            }
            let mut v_branch_layers: Vec<Box<dyn crate::layer::Layer>> = Vec::new();
            for br in v_branches.iter() {
                let b = build_layer_from_spec(br, vocab)?;
                v_branch_layers.push(b);
            }
            let pg = crate::layer::ParallelBlockGroup::new(v_branch_layers)?;
            Ok(Box::new(pg))
        }

        llm_layer_spec::output_projection { i_embedding_dim } => Ok(Box::new(
            crate::layer::OutputProjection::new(*i_embedding_dim, vocab.words.len()),
        )),
    }
}

fn build_network_from_topology(
    topology: &llm_topology_spec,
    vocab: &crate::layer::Vocab,
) -> Result<Vec<Box<dyn crate::layer::Layer>>, String> {
    let mut v_net: Vec<Box<dyn crate::layer::Layer>> = Vec::new();
    for layer_spec in topology.v_layers.iter() {
        v_net.push(build_layer_from_spec(layer_spec, vocab)?);
    }
    Ok(v_net)
}

#[derive(Clone, Debug)]
pub struct PredictStats {
    // Average probability of the selected token per generation step, in [0, 1].
    pub d_avg_selected_token_prob: f32,

    // Additional prediction metrics.
    // Perplexity proxy: exp(mean(-ln(p_selected))) computed from selected token probabilities.
    pub d_perplexity_selected: f32,

    // Average entropy (nats) of next token distribution at each generation step.
    pub d_avg_next_token_entropy_nat: f32,

    // Average margin between top1 and top2 probabilities at each generation step.
    pub d_avg_top1_top2_margin: f32,

    // Count of generation steps used for these stats.
    pub i_steps: usize,
}

impl PredictStats {
    pub fn empty() -> Self {
        Self {
            d_avg_selected_token_prob: 0.0,
            d_perplexity_selected: 0.0,
            d_avg_next_token_entropy_nat: 0.0,
            d_avg_top1_top2_margin: 0.0,
            i_steps: 0,
        }
    }
}

fn sanitize_f32_local(d_x: f32) -> f32 {
    if d_x.is_finite() { d_x } else { 0.0 }
}

fn clamp_prob_local(d_x: f32) -> f32 {
    if !d_x.is_finite() {
        return 0.0;
    }
    if d_x < 0.0 {
        0.0
    } else if d_x > 1.0 {
        1.0
    } else {
        d_x
    }
}

fn entropy_nat_local(v_p: &[f32]) -> f32 {
    let mut d_h: f32 = 0.0;
    for &p in v_p.iter() {
        let d_p = clamp_prob_local(p);
        if d_p > 0.0 {
            d_h -= d_p * d_p.max(1e-12).ln();
        }
    }
    sanitize_f32_local(d_h)
}

fn top1_top2_margin_local(v_p: &[f32]) -> f32 {
    if v_p.is_empty() {
        return 0.0;
    }

    let mut d_top1: f32 = -1.0;
    let mut d_top2: f32 = -1.0;

    for &p in v_p.iter() {
        let d_p = clamp_prob_local(p);
        if d_p > d_top1 {
            d_top2 = d_top1;
            d_top1 = d_p;
        } else if d_p > d_top2 {
            d_top2 = d_p;
        }
    }

    let d_margin = (d_top1 - d_top2).max(0.0);
    sanitize_f32_local(d_margin)
}

fn mean_vec_f32_local(v_x: &[f32]) -> f32 {
    if v_x.is_empty() {
        return 0.0;
    }
    let mut d_sum: f32 = 0.0;
    for &d in v_x.iter() {
        d_sum += sanitize_f32_local(d);
    }
    sanitize_f32_local(d_sum / (v_x.len() as f32).max(1.0))
}

fn compute_perplexity_from_selected_probs_local(v_p_sel: &[f32]) -> f32 {
    if v_p_sel.is_empty() {
        return 0.0;
    }

    let mut d_sum_nll: f32 = 0.0;
    for &p in v_p_sel.iter() {
        let d_p = clamp_prob_local(p).max(1e-12);
        d_sum_nll += -d_p.ln();
    }
    let d_mean_nll = d_sum_nll / (v_p_sel.len() as f32).max(1.0);
    sanitize_f32_local(d_mean_nll.exp())
}

// -----------------------------
// Background training support
// -----------------------------

#[derive(Clone, Debug)]
pub struct TrainingProgressEventAscii {
    pub s_phase: String,
    pub i_epoch_current: usize,
    pub i_epochs_total: usize,
    pub d_last_epoch_loss: f32,
    pub d_last_step_loss: f32,
    pub i_rows_used_last_epoch: usize,
    pub i_total_steps: usize,
}

// -----------------------------
// Continuous learning config
// -----------------------------
#[derive(Clone, Debug)]
pub struct ContinuousLearningConfig {
    // Availability probability per branch i, p_i > 0.
    // Used for unbiased scaling 1/p_i in expectation.
    pub v_branch_participation_p: Vec<f32>,

    // Enforce at least this many active branches per step.
    pub i_min_active_branches: usize,

    // If true, scale the gradient by 1/p_i when branch is active (unbiased estimator).
    // If false, no scaling is applied (may be biased if p differs across branches).
    pub b_scale_by_inverse_participation: bool,

    // RNG seed for deterministic mask sampling.
    pub u64_mask_seed: u64,
}

impl ContinuousLearningConfig {
    pub fn new_default_for_num_branches(i_num_branches: usize) -> Self {
        let d_p = if i_num_branches == 0 {
            1.0
        } else {
            0.75
        };
        Self {
            v_branch_participation_p: vec![d_p; i_num_branches.max(1)],
            i_min_active_branches: 1,
            b_scale_by_inverse_participation: true,
            u64_mask_seed: 20260213,
        }
    }

    pub fn validate(&self, i_num_branches: usize) -> Result<(), String> {
        if i_num_branches == 0 {
            return Err("continuous_learning_num_branches_zero".to_string());
        }
        if self.v_branch_participation_p.len() != i_num_branches {
            return Err("continuous_learning_participation_len_mismatch".to_string());
        }
        if self.i_min_active_branches == 0 || self.i_min_active_branches > i_num_branches {
            return Err("continuous_learning_min_active_invalid".to_string());
        }
        for &p in self.v_branch_participation_p.iter() {
            if !p.is_finite() || p <= 0.0 || p > 1.0 {
                return Err("continuous_learning_participation_p_invalid".to_string());
            }
        }
        Ok(())
    }
}


// Helper: compute step loss for a single forward pass.
// This mirrors math::cross_entropy_loss_step but returns scalar for the current batch.
fn compute_loss_step_local(a_probs: &Array2<f32>, v_target_ids: &[usize]) -> f32 {
    math::cross_entropy_loss_step(a_probs, v_target_ids)
}
// ----------------------------------------
// Llm
// ----------------------------------------

#[allow(clippy::upper_case_acronyms)]
pub struct Llm {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
    pub bpe_tokenizer: Option<BpeTokenizer>,

    pub b_training: bool,
    pub u64_dropout_seed: u64,
    pub d_residual_dropout_p: f32,

    pub d_temperature: f32,
    pub i_top_k: usize,
    pub d_top_p: f32,
    pub rng_sampling: StdRng,

    // Test-only fault injection switch for MTB outage simulation.
    // When false, no outage path is selected before predict.
    pub b_outage_simulation_enabled: bool,
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
            d_temperature: 1.0,
            i_top_k: 0,
            d_top_p: 0.0,
            rng_sampling: StdRng::seed_from_u64(12345),
            b_outage_simulation_enabled: false,
        }
    }

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
        if !d_top_p.is_finite() || d_top_p < 0.0 || d_top_p > 1.0 {
            return Err("sampling_top_p_invalid".to_string());
        }

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
            if let Some(ts) = layer.as_any_mut().and_then(|a| a.downcast_mut::<TransformerSequence>()) {
                ts.set_training(b_training);
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
            if let Some(ts) = layer.as_any_mut().and_then(|a| a.downcast_mut::<TransformerSequence>()) {
                ts.set_residual_dropout_p(self.d_residual_dropout_p);
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
            i_pos = i_pos.saturating_add(i_used);
        }

        if i_pos != v_params.len() {
            return Err("checkpoint_params_length_mismatch".to_string());
        }

        Ok(())
    }

    fn export_topology_spec(&mut self) -> Result<llm_topology_spec, String> {
        // NOTE: Uses &mut self so composite layers can be downcasted via as_any_mut().
        let mut v_layers: Vec<llm_layer_spec> = Vec::new();

        for layer in self.network.iter_mut() {
            let s_t = layer.layer_type();

            if s_t == "Embeddings" {
                v_layers.push(llm_layer_spec::embeddings);
                continue;
            }

            if s_t == "TransformerBlock" {
                v_layers.push(llm_layer_spec::transformer_block {
                    i_embedding_dim: crate::EMBEDDING_DIM,
                    i_hidden_dim: crate::HIDDEN_DIM,
                    i_num_heads: 4,
                });
                continue;
            }

            if s_t == "TransformerSequence" {
                let ts = layer
                    .as_any_mut()
                    .and_then(|a| a.downcast_mut::<crate::layer::TransformerSequence>())
                    .ok_or_else(|| "topology_export_downcast_transformer_sequence_failed".to_string())?;
                v_layers.push(ts.export_spec());
                continue;
            }

            if s_t == "ParallelBlockGroup" {
                let pg = layer
                    .as_any_mut()
                    .and_then(|a| a.downcast_mut::<crate::layer::ParallelBlockGroup>())
                    .ok_or_else(|| "topology_export_downcast_parallel_block_group_failed".to_string())?;

                // Export branches including TransformerSequence via downcast on each branch.
                let mut v_branches: Vec<llm_layer_spec> = Vec::new();
                for br in pg.v_branches.iter_mut() {
                    let s_bt = br.layer_type();
                    if s_bt == "TransformerBlock" {
                        v_branches.push(llm_layer_spec::transformer_block {
                            i_embedding_dim: crate::EMBEDDING_DIM,
                            i_hidden_dim: crate::HIDDEN_DIM,
                            i_num_heads: 4,
                        });
                    } else if s_bt == "TransformerSequence" {
                        let tsb = br
                            .as_any_mut()
                            .and_then(|a| a.downcast_mut::<crate::layer::TransformerSequence>())
                            .ok_or_else(|| "topology_export_downcast_branch_transformer_sequence_failed".to_string())?;
                        v_branches.push(tsb.export_spec());
                    } else {
                        return Err("topology_export_parallel_branch_type_unsupported".to_string());
                    }
                }

                v_layers.push(llm_layer_spec::parallel_block_group { v_branches });
                continue;
            }

            if s_t == "OutputProjection" {
                v_layers.push(llm_layer_spec::output_projection {
                    i_embedding_dim: crate::EMBEDDING_DIM,
                });
                continue;
            }

            return Err("topology_export_unknown_layer_type".to_string());
        }

        Ok(llm_topology_spec { v_layers })
    }

    pub fn save_checkpoint_llm_checkpoint_v2(&mut self, s_path: &str) -> Result<(), String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        // Step 1: Create topology first (requires &mut self).
        // This must happen before any long-lived immutable borrows from self.
        let topology = self.export_topology_spec()?;

        // Step 2: Materialize tokenizer checkpoint without keeping a borrow alive.
        // Use a narrow scope so the borrow ends immediately.
        let tokenizer_cp = {
            let tok = self
                .bpe_tokenizer
                .as_ref()
                .ok_or_else(|| "tokenizer_not_set".to_string())?;
            tok.to_checkpoint()
        };

        // Step 3: Collect parameters (can borrow self immutably; no conflict now).
        let v_params = self.collect_all_parameters_flat();

        let cp = llm_checkpoint_v2::new(
            tokenizer_cp,
            topology,
            v_params,
            crate::MAX_SEQ_LEN,
            crate::EMBEDDING_DIM,
            crate::HIDDEN_DIM,
        );

        let s_json = crate::utils::checkpoint_to_json_ascii(&cp)?;
        crate::utils::write_file_atomic_ascii(s_path, &s_json)?;
        Ok(())
    }

    pub fn load_checkpoint_llm_checkpoint_v2_rebuild(s_path: &str) -> Result<crate::layer::Llm, String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        let s_json = std::fs::read_to_string(s_path).map_err(|_| "checkpoint_read_error".to_string())?;
        let cp: llm_checkpoint_v2 = crate::utils::checkpoint_from_json_ascii(&s_json)?;
        cp.validate()?;

        let bpe = crate::tokenizer::BpeTokenizer::from_checkpoint(&cp.tokenizer)?;
        let vocab = bpe.vocab.clone();

        let network = build_network_from_topology(&cp.topology, &vocab)?;

        let mut llm = crate::layer::Llm::new(vocab, network);
        llm.set_bpe_tokenizer(bpe);

        llm.set_residual_dropout_p(0.1);
        llm.set_training(true);
        let _ = llm.set_sampling_config(0.9, 40, 0.95, 987654321);

        let i_expected: usize = llm.network.iter().map(|l| l.get_parameters_flat().len()).sum();
        if i_expected != cp.v_params.len() {
            return Err(format!(
                "checkpoint_param_count_mismatch expected={} got={}",
                i_expected,
                cp.v_params.len()
            ));
        }

        llm.assign_all_parameters_flat(&cp.v_params)?;
        //llm.run_post_load_mtb_diagnostics_ascii();
        
        Ok(llm)
    }

    

    fn sample_next_token_from_logits(&mut self, a_last_logits: &Array2<f32>) -> Result<usize, String> {
        if a_last_logits.nrows() != 1 || a_last_logits.ncols() == 0 {
            return Err("sampling_logits_shape_invalid".to_string());
        }

        let i_vocab = a_last_logits.ncols();

        let d_temperature = if self.d_temperature.is_finite() && self.d_temperature > 0.0 {
            self.d_temperature
        } else {
            1.0
        };

        let i_top_k = self.i_top_k;
        let d_top_p = if self.d_top_p.is_finite() { self.d_top_p } else { 0.0 };

        let d_temp = d_temperature.max(1e-6);

        let mut a_scaled = a_last_logits.clone();
        for d in a_scaled.iter_mut() {
            if !d.is_finite() {
                *d = 0.0;
            } else {
                *d = *d / d_temp;
            }
        }

        let a_probs = math::softmax_rows(&a_scaled);
        if a_probs.nrows() != 1 || a_probs.ncols() != i_vocab {
            return Err("sampling_probs_shape_invalid".to_string());
        }

        let mut v_pairs: Vec<(usize, f32)> = (0..i_vocab)
            .map(|i| (i, a_probs[[0, i]]))
            .filter(|(_, p)| p.is_finite() && *p > 0.0)
            .collect();

        if v_pairs.is_empty() {
            return Err("sampling_probs_empty".to_string());
        }

        v_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut v_filtered: Vec<(usize, f32)> = if i_top_k > 0 && i_top_k < v_pairs.len() {
            v_pairs[..i_top_k].to_vec()
        } else {
            v_pairs
        };

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

            if !v_nucleus.is_empty() {
                v_filtered = v_nucleus;
            }
        }

        let d_sum: f32 = v_filtered.iter().map(|(_, p)| *p).sum();
        if !d_sum.is_finite() || d_sum <= 0.0 {
            return Err("sampling_filtered_sum_invalid".to_string());
        }

        let v_weights: Vec<f32> = v_filtered.iter().map(|(_, p)| *p / d_sum).collect();
        let v_ids: Vec<usize> = v_filtered.iter().map(|(i, _)| *i).collect();

        if v_weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err("sampling_weights_invalid".to_string());
        }

        let dist = WeightedIndex::new(&v_weights).map_err(|_| "sampling_weighted_index_error".to_string())?;
        let i_pick = dist.sample(&mut self.rng_sampling);
        v_ids.get(i_pick).copied().ok_or_else(|| "sampling_pick_oob".to_string())
    }

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

    fn set_predict_outage_for_all_parallel_groups_test_only(&mut self) {
        // Borrow safe approach: move RNG out of self before iter_mut on network.
        let mut rng_local: StdRng =
            std::mem::replace(&mut self.rng_sampling, StdRng::seed_from_u64(0));

        for layer in self.network.iter_mut() {
            let opt_pg = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
            if let Some(pg) = opt_pg {
                let i_k = pg.num_branches();
                let opt_drop: Option<usize> = if i_k == 0 {
                    None
                } else {
                    Some(rng_local.gen_range(0..i_k))
                };
                pg.set_fault_injection_enabled(true);
                pg.set_fault_drop_branch_idx(opt_drop);
            }
        }

        self.rng_sampling = rng_local;
    }

    fn clear_predict_outage_for_all_parallel_groups_test_only(&mut self) {
        for layer in self.network.iter_mut() {
            let opt_pg = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
            if let Some(pg) = opt_pg {
                pg.set_fault_drop_branch_idx(None);
                pg.set_fault_injection_enabled(false);
            }
        }
    }

    // Enable or disable outage simulation (test-only).
    pub fn set_outage_simulation_enabled(&mut self, b_enabled: bool) {
        self.b_outage_simulation_enabled = b_enabled;
        if !b_enabled {
            // Best effort cleanup: ensure all groups are fully enabled.
            self.clear_predict_outage_for_all_parallel_groups_test_only();
        }
    }

    // Query current outage simulation state.
    pub fn is_outage_simulation_enabled(&self) -> bool {
        self.b_outage_simulation_enabled
    }

    // - 2026-02-04: Ensure predict runs in eval mode and restores training state.
    // - 2026-02-07: Test only outage injection: drop exactly one branch per ParallelBlockGroup.
    pub fn predict(&mut self, s_text: &str) -> Result<String, String> {
        let b_prev = self.b_training;
        self.set_training(false);


        // Test-only: simulate outage only if enabled.
        if self.b_outage_simulation_enabled {
            self.set_predict_outage_for_all_parallel_groups_test_only();
        }

        let r = self
            .forward_generate(s_text)
            .map(|v_out_ids| self.decode_ids(&v_out_ids));

        // Cleanup only if enabled.
        if self.b_outage_simulation_enabled {
            self.clear_predict_outage_for_all_parallel_groups_test_only();
        }

        self.set_training(b_prev);
        r
    }

    // - 2026-02-08: Add forward_generate_with_stats to support prediction metrics reporting.
    fn forward_generate_with_stats(&mut self, s_text: &str) -> Result<(Vec<usize>, PredictStats), String> {
        let mut v_context = self.tokenize(s_text)?;
        let mut v_generated: Vec<usize> = Vec::new();

        if v_context.len() >= MAX_SEQ_LEN {
            return Ok((v_generated, PredictStats::empty()));
        }

        let opt_eos = self.vocab.encode(S_EOS);

        // Collect per step stats.
        let mut v_selected_probs: Vec<f32> = Vec::new();
        let mut v_entropies: Vec<f32> = Vec::new();
        let mut v_margins: Vec<f32> = Vec::new();

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

            let a_last_logits = a_logits
                .row(a_logits.nrows().saturating_sub(1))
                .to_owned()
                .insert_axis(Axis(0));

            // Compute distribution for stats (and also for sampling).
            let i_vocab = a_last_logits.ncols();
            if i_vocab == 0 {
                return Err("sampling_vocab_zero".to_string());
            }

            let d_temperature = if self.d_temperature.is_finite() && self.d_temperature > 0.0 {
                self.d_temperature
            } else {
                1.0
            };
            let d_temp = d_temperature.max(1e-6);

            let mut a_scaled = a_last_logits.clone();
            for d in a_scaled.iter_mut() {
                if !d.is_finite() {
                    *d = 0.0;
                } else {
                    *d = *d / d_temp;
                }
            }

            let a_probs = crate::math::softmax_rows(&a_scaled);
            if a_probs.nrows() != 1 || a_probs.ncols() != i_vocab {
                return Err("sampling_probs_shape_invalid".to_string());
            }

            // Entropy and top1-top2 margin on full distribution.
            let mut v_p: Vec<f32> = Vec::with_capacity(i_vocab);
            for j in 0..i_vocab {
                v_p.push(clamp_prob_local(a_probs[[0, j]]));
            }
            v_entropies.push(entropy_nat_local(&v_p));
            v_margins.push(top1_top2_margin_local(&v_p));

            // Sample next token using existing method (keeps topk/topp semantics).
            let i_next = self.sample_next_token_from_logits(&a_last_logits)?;

            // Selected token probability from full distribution.
            let d_p_sel = if i_next < i_vocab { a_probs[[0, i_next]] } else { 0.0 };
            v_selected_probs.push(clamp_prob_local(d_p_sel));

            v_generated.push(i_next);
            v_context.push(i_next);

            if let Some(i_eos) = opt_eos {
                if i_next == i_eos {
                    break;
                }
            }
        }

        let d_avg_p_sel = mean_vec_f32_local(&v_selected_probs);
        let d_ppl = compute_perplexity_from_selected_probs_local(&v_selected_probs);
        let d_avg_h = mean_vec_f32_local(&v_entropies);
        let d_avg_margin = mean_vec_f32_local(&v_margins);

        let stats = PredictStats {
            d_avg_selected_token_prob: d_avg_p_sel,
            d_perplexity_selected: d_ppl,
            d_avg_next_token_entropy_nat: d_avg_h,
            d_avg_top1_top2_margin: d_avg_margin,
            i_steps: v_selected_probs.len(),
        };

        Ok((v_generated, stats))
    }

    // History:
    // - 2026-02-08: Add predict_with_stats for post predict metrics in main.rs.
    pub fn predict_with_stats(&mut self, s_text: &str) -> Result<(String, PredictStats), String> {
        let b_prev = self.b_training;
        self.set_training(false);

        if self.b_outage_simulation_enabled {
            self.set_predict_outage_for_all_parallel_groups_test_only();
        }

        let r = self.forward_generate_with_stats(s_text).map(|(v_out_ids, st)| {
            let s_out = self.decode_ids(&v_out_ids);
            (s_out, st)
        });

        if self.b_outage_simulation_enabled {
            self.clear_predict_outage_for_all_parallel_groups_test_only();
        }

        self.set_training(b_prev);
        r
    }


    pub fn export_parameters_snapshot(&self) -> Vec<f32> {
        // History:
        // - 2026-02-13: Add snapshot export to update serving model without blocking training.
        self.collect_all_parameters_flat()
    }

    pub fn import_parameters_snapshot(&mut self, v_params: &[f32]) -> Result<(), String> {
        // History:
        // - 2026-02-13: Add snapshot import for serving model hot updates.
        self.assign_all_parameters_flat(v_params)
    }

    fn sample_availability_mask(
        rng: &mut StdRng,
        v_p: &[f32],
        i_min_active: usize,
    ) -> Vec<bool> {
        let i_k = v_p.len();
        if i_k == 0 {
            return Vec::new();
        }

        let mut v_mask = vec![false; i_k];
        let mut i_active: usize = 0;

        for i in 0..i_k {
            let d_u: f32 = rng.gen_range(0.0..1.0);
            let d_p = v_p[i].clamp(0.0, 1.0);
            if d_u < d_p {
                v_mask[i] = true;
                i_active = i_active.saturating_add(1);
            }
        }

        // Enforce minimum participation by activating random inactive branches.
        if i_active < i_min_active {
            let mut v_inactive: Vec<usize> = Vec::new();
            for i in 0..i_k {
                if !v_mask[i] {
                    v_inactive.push(i);
                }
            }

            while i_active < i_min_active && !v_inactive.is_empty() {
                let i_pick = rng.gen_range(0..v_inactive.len());
                let i_idx = v_inactive.swap_remove(i_pick);
                v_mask[i_idx] = true;
                i_active = i_active.saturating_add(1);
            }
        }

        v_mask
    }

    pub fn train_with_progress_continuous_learning_ascii(
        &mut self,
        v_data: Vec<&str>,
        i_epochs: usize,
        d_lr: f32,
        b_cancel: Arc<AtomicBool>,
        tx_progress: Sender<TrainingProgressEventAscii>,
        s_phase: &str,
        opt_cl: Option<ContinuousLearningConfig>,
        i_snapshot_every_steps: usize,
        tx_snapshot: Option<Sender<Vec<f32>>>,
    ) -> Result<(), String> {
        // History:
        // - 2026-02-13: Add continuous learning with partial path availability and snapshot sending.
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

        // Find first ParallelBlockGroup for partial availability. If none, fallback to normal training.
        let mut opt_pg_idx: Option<usize> = None;
        for (i_idx, layer) in self.network.iter().enumerate() {
            if layer.layer_type() == "ParallelBlockGroup" {
                opt_pg_idx = Some(i_idx);
                break;
            }
        }

        let mut rng_mask = StdRng::seed_from_u64(
            opt_cl.as_ref().map(|c| c.u64_mask_seed).unwrap_or(20260213),
        );

        let mut i_total_steps: usize = 0;

        for i_epoch in 0..i_epochs {
            if b_cancel.load(AtomicOrdering::SeqCst) {
                return Ok(());
            }

            let mut d_total_loss: f32 = 0.0;
            let mut i_used_rows: usize = 0;
            let mut d_last_step_loss: f32 = 0.0;

            for v_row in v_tokenized_data.iter() {
                if b_cancel.load(AtomicOrdering::SeqCst) {
                    return Ok(());
                }

                let v_input_ids = &v_row[..v_row.len() - 1];
                let v_target_ids = &v_row[1..];

                let mut a_input: Array2<f32> = Array2::zeros((1, v_input_ids.len()));
                let a_row = Array1::from_iter(v_input_ids.iter().map(|&x| x as f32));
                a_input.row_mut(0).assign(&a_row);

                // Forward with optional partial availability on first ParallelBlockGroup.
                let mut a_act = a_input;

                if let Some(i_pg) = opt_pg_idx {
                    // Forward layers before pg normally.
                    for i_l in 0..i_pg {
                        a_act = self.network[i_l].forward(&a_act);
                        if a_act.nrows() == 0 || a_act.ncols() == 0 {
                            break;
                        }
                    }
                    if a_act.nrows() == 0 || a_act.ncols() == 0 {
                        continue;
                    }

                    // Prepare mask.
                    let i_k = match self.network[i_pg]
                        .as_any_mut()
                        .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                    {
                        Some(pg) => pg.num_branches(),
                        None => 0,
                    };
                    if i_k == 0 {
                        continue;
                    }

                    let cl_cfg = opt_cl.clone().unwrap_or_else(|| {
                        ContinuousLearningConfig::new_default_for_num_branches(i_k)
                    });
                    cl_cfg.validate(i_k)?;

                    let v_mask =
                        Self::sample_availability_mask(&mut rng_mask, &cl_cfg.v_branch_participation_p, cl_cfg.i_min_active_branches);

                    // Forward through pg with mask.
                    {
                        let pg = self.network[i_pg]
                            .as_any_mut()
                            .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                            .ok_or_else(|| "pg_downcast_failed".to_string())?;

                        a_act = pg.forward_with_availability_mask(&a_act, &v_mask);
                    }

                    if a_act.nrows() == 0 || a_act.ncols() == 0 {
                        continue;
                    }

                    // Forward remaining layers after pg.
                    for i_l in (i_pg + 1)..self.network.len() {
                        a_act = self.network[i_l].forward(&a_act);
                        if a_act.nrows() == 0 || a_act.ncols() == 0 {
                            break;
                        }
                    }
                    if a_act.nrows() == 0 || a_act.ncols() == 0 {
                        continue;
                    }

                    let a_logits = a_act;
                    let a_probs = math::softmax_rows(&a_logits);

                    let d_step_loss = math::cross_entropy_loss_step(&a_probs, v_target_ids);
                    d_last_step_loss = d_step_loss;
                    d_total_loss += d_step_loss;

                    let mut a_grads = math::compute_gradients_step(&a_probs, v_target_ids);
                    math::clip_gradients_global_norm(&mut a_grads, 5.0);

                    // Backward after pg normally down to pg+1.
                    for i_l in (i_pg + 1..self.network.len()).rev() {
                        a_grads = self.network[i_l].backward(&a_grads, d_lr);
                    }

                    // Backward through pg with mask and optional 1/p scaling.
                    let opt_scales: Option<Vec<f32>> = if cl_cfg.b_scale_by_inverse_participation {
                        Some(
                            cl_cfg
                                .v_branch_participation_p
                                .iter()
                                .map(|&p| (1.0 / p.max(1e-12)).clamp(0.0, 1.0e6))
                                .collect(),
                        )
                    } else {
                        None
                    };

                    {
                        let pg = self.network[i_pg]
                            .as_any_mut()
                            .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                            .ok_or_else(|| "pg_downcast_failed".to_string())?;

                        a_grads = pg.backward_with_availability_mask(
                            &a_grads,
                            d_lr,
                            &v_mask,
                            opt_scales.as_deref(),
                        );
                    }

                    // Backward for layers before pg.
                    for i_l in (0..i_pg).rev() {
                        a_grads = self.network[i_l].backward(&a_grads, d_lr);
                    }
                } else {
                    // Fallback: original full training if no ParallelBlockGroup exists.
                    for layer in self.network.iter_mut() {
                        a_act = layer.forward(&a_act);
                    }
                    let a_logits = a_act;
                    if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
                        continue;
                    }
                    let a_probs = math::softmax_rows(&a_logits);
                    let d_step_loss = math::cross_entropy_loss_step(&a_probs, v_target_ids);
                    d_last_step_loss = d_step_loss;
                    d_total_loss += d_step_loss;

                    let mut a_grads = math::compute_gradients_step(&a_probs, v_target_ids);
                    math::clip_gradients_global_norm(&mut a_grads, 5.0);
                    for layer in self.network.iter_mut().rev() {
                        a_grads = layer.backward(&a_grads, d_lr);
                    }
                }

                i_used_rows = i_used_rows.saturating_add(1);
                i_total_steps = i_total_steps.saturating_add(1);

                // Progress update.
                if (i_total_steps % 25) == 0 {
                    let _ = tx_progress.send(TrainingProgressEventAscii {
                        s_phase: s_phase.to_string(),
                        i_epoch_current: i_epoch + 1,
                        i_epochs_total: i_epochs,
                        d_last_epoch_loss: 0.0,
                        d_last_step_loss,
                        i_rows_used_last_epoch: 0,
                        i_total_steps,
                    });
                }

                // Snapshot update for serving.
                if i_snapshot_every_steps > 0 && (i_total_steps % i_snapshot_every_steps) == 0 {
                    if let Some(tx) = tx_snapshot.as_ref() {
                        let v_params = self.export_parameters_snapshot();
                        let _ = tx.send(v_params);
                    }
                }
            }

            let d_avg_loss = if i_used_rows == 0 {
                0.0
            } else {
                d_total_loss / (i_used_rows as f32).max(1.0)
            };

            let _ = tx_progress.send(TrainingProgressEventAscii {
                s_phase: s_phase.to_string(),
                i_epoch_current: i_epoch + 1,
                i_epochs_total: i_epochs,
                d_last_epoch_loss: d_avg_loss,
                d_last_step_loss,
                i_rows_used_last_epoch: i_used_rows,
                i_total_steps,
            });

         //   println!("Epoch {} ({}) : Loss = {:.6}", i_epoch + 1, s_phase, d_avg_loss);
        }

        Ok(())
    }
    // Cooperative training with progress events.
    //
    // History:
    // - 2026-02-13: Add train_with_progress_ascii to allow background training and live metrics.
    pub fn train_with_progress_ascii(
        &mut self,
        v_data: Vec<&str>,
        i_epochs: usize,
        d_lr: f32,
        b_cancel: Arc<AtomicBool>,
        tx_progress: Sender<TrainingProgressEventAscii>,
        s_phase: &str,
    ) -> Result<(), String> {
        // Existing logic, but ensure epoch_current is sent as (i_epoch + 1)
        // and total_steps is incremented when a row is actually used.

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
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .filter(|v| v.len() >= 2)
            .collect();

        if v_tokenized_data.is_empty() {
            return Err("no_tokenized_rows".to_string());
        }

        let mut i_total_steps: usize = 0;

        for i_epoch in 0..i_epochs {
             if b_cancel.load(AtomicOrdering::SeqCst) {
                return Ok(());
            }

            let mut d_total_loss: f32 = 0.0;
            let mut i_used_rows: usize = 0;
            let mut d_last_step_loss: f32 = 0.0;

            for v_row in v_tokenized_data.iter() {
                if b_cancel.load(AtomicOrdering::SeqCst) {
                    return Ok(());
                }

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

                let d_step_loss = math::cross_entropy_loss_step(&a_probs, v_target_ids);
                d_last_step_loss = d_step_loss;
                d_total_loss += d_step_loss;

                let mut a_grads = math::compute_gradients_step(&a_probs, v_target_ids);
                math::clip_gradients_global_norm(&mut a_grads, 5.0);

                for layer in self.network.iter_mut().rev() {
                    a_grads = layer.backward(&a_grads, d_lr);
                }

                i_used_rows = i_used_rows.saturating_add(1);
                i_total_steps = i_total_steps.saturating_add(1);

                if (i_total_steps % 25) == 0 {
                    let _ = tx_progress.send(TrainingProgressEventAscii {
                        s_phase: s_phase.to_string(),
                        i_epoch_current: i_epoch + 1,
                        i_epochs_total: i_epochs,
                        d_last_epoch_loss: 0.0,
                        d_last_step_loss,
                        i_rows_used_last_epoch: 0,
                        i_total_steps,
                    });
                }
            }

            let d_avg_loss = if i_used_rows == 0 {
                0.0
            } else {
                d_total_loss / (i_used_rows as f32).max(1.0)
            };

            let _ = tx_progress.send(TrainingProgressEventAscii {
                s_phase: s_phase.to_string(),
                i_epoch_current: i_epoch + 1,
                i_epochs_total: i_epochs,
                d_last_epoch_loss: d_avg_loss,
                d_last_step_loss,
                i_rows_used_last_epoch: i_used_rows,
                i_total_steps,
            });

            //println!("Epoch {} ({}) : Loss = {:.6}", i_epoch + 1, s_phase, d_avg_loss);
        }

        Ok(())
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
        // - 2026-02-01: Consolidated training loop into layer.rs within Llm::train.
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

        let mut opt_idx: Option<usize> = None;
        for (i_idx, layer) in self.network.iter().enumerate() {
            if layer.layer_type() == "ParallelBlockGroup" {
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

    // History:
    // - 2026-02-07: Add post load MTB diagnostics metrics computation.
    pub fn run_post_load_mtb_diagnostics_ascii(&mut self) {
        let b_prev = self.b_training;
        self.set_training(false);

        let v_prompts: Vec<String> = vec![
            "User: Explain transformers briefly.".to_string(),
            "User: Summarize the concept of attention.".to_string(),
            "User: What is gradient clipping?".to_string(),
            "User: Provide a short definition of entropy.".to_string(),
            "User: Describe tokenization.".to_string(),
            "User: How do mountains form?".to_string(),
            "User: Explain causal masking.".to_string(),
            "User: Define overfitting.".to_string(),
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
            let opt_pg = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
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
