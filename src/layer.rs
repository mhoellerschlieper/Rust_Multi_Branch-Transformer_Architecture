// layer.rs
// Description: Model layers and core LLM implementation (forward, backward, train, predict).
//              Implements embeddings, transformer blocks (MHSA + FFN), RMSNorm, output projection,
//              AdamW optimizer, dropout, and checkpoint save/load.
//
//              Adds MBT (multi branch transformer) support via ParallelBlockGroup, which can run
//              branches in parallel (width) and aggregate outputs. Supports branch sequences via
//              TransformerSequence.
//
//              Adds post load diagnostics for ParallelBlockGroup metrics (path starvation, diversity,
//              and additional metrics) and test only fault injection to simulate one dropped path
//              before each predict.
//
//              Continuous learning extensions:
//              - Partial branch availability via masks
//              - Inverse participation scaling for unbiased gradient estimates (in expectation)
//              - Cooperative cancel and progress events
//              - EMA stabilized branch selection (phase controlled)
//              - Experience replay buffer (phase controlled)
//              - Weighted aggregation and conservative injection for autonomous width expansion
//
// History:
// - 2026-02-01: Consolidate project into 6 files: main, layer, train, math, tokenizer, utils.
// - 2026-02-01: Add checkpoint save and load for model parameters and tokenizer.
// - 2026-02-04: Add robust sampling (temperature, top k, top p) and ensure predict runs eval mode.
// - 2026-02-07: Add MBT ParallelBlockGroup and TransformerSequence support.
// - 2026-02-07: Add MBT diagnostics and test only outage simulation with borrow safe RNG handling.
// - 2026-02-08: Add checkpoint topology spec and rebuild model from topology.
// - 2026-02-11: Parallelize ParallelBlockGroup forward and backward via Rayon.
// - 2026-02-13: Add cooperative cancel and training progress events for background training.
// - 2026-02-13: Add selective branch training with EMA loss and replay buffer.
// - 2026-02-14: Fix progress reporting: compute and report running epoch mean loss in step events.
// - 2026-02-14: Add phase oriented training strategy with ramp up and autonomous expansion hooks.
// - 2026-02-14: Add weighted aggregation and conservative injection primitives for branch expansion.
// - 2026-02-14: Add online training data ingestion via TrainingDataEventAscii (file and rows).
// Author: Marcus Schlieper (ExpChat.ai)
#![allow(warnings)]

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
use std::sync::mpsc::{Receiver, Sender};
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

// ----------------------------------------
// ParallelBlockGroup (MTB width layer) with diagnostics and test only outage injection
// ----------------------------------------

pub struct ParallelBlockGroup {
    v_branches: Vec<Box<dyn Layer>>,
    d_equal_weight: f32,

    // explicit weights for weighted aggregation and conservative injection.
    pub v_branch_weights: Vec<f32>,

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
        let v_w = vec![d_w; v_branches.len()];

        Ok(Self {
            v_branches,
            d_equal_weight: d_w,
            v_branch_weights: v_w,
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
                let d_e = math::mean_square_energy_f32(a_y);
                v_scores.push(d_e);
                v_energy.push(d_e);
                v_flat.push(math::flatten_array2_f32(a_y));
            }

            let a_scores_row = Array2::from_shape_vec((1, v_scores.len()), v_scores.clone())
                .map_err(|_| "parallel_block_group_metrics_shape_error".to_string())?;
            let a_p = math::softmax_rows(&a_scores_row);

            let mut v_p: Vec<f32> = Vec::with_capacity(i_k);
            for i in 0..i_k {
                v_p.push(math::clamp_prob_f32(a_p[[0, i]]));
            }
            v_p = math::normalize_distribution_f32(&v_p);

            let d_h = math::entropy_nat_f32(&v_p);
            let d_h_max = (i_k as f32).max(1.0).ln().max(1e-12);
            let d_h_norm = (d_h / d_h_max).clamp(0.0, 1.0);
            let d_psi = 1.0 - d_h_norm;

            let d_eff = d_h.exp().clamp(1.0, i_k as f32);
            let d_gini = math::gini_coefficient_f32(&v_p);

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
                    let d_sim = math::cosine_similarity_f32(v_a, v_b);
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
        let d_energy_cv = math::coeff_of_variation_f32(&v_energy_all);

        Ok(ParallelBlockGroupMetrics {
            i_num_paths: i_k,
            i_num_samples: i_used,
            d_path_starvation_index: math::sanitize_f32(d_psi_sum / d_n),
            d_diversity_cosine_distance_mean: math::sanitize_f32(d_div_sum / d_n),
            d_effective_num_paths: math::sanitize_f32(d_eff_paths_sum / d_n),
            d_gini_concentration: math::sanitize_f32(d_gini_sum / d_n),
            d_top1_share: math::sanitize_f32(d_top1_sum / d_n),
            d_margin_top1_top2: math::sanitize_f32(d_margin_sum / d_n),
            d_output_energy_cv: math::sanitize_f32(d_energy_cv),
            d_branch_correlation_mean: math::sanitize_f32(d_corr_sum / d_n),
        })
    }


    // Branch weights w_i >= 0, sum(w_i)=1.
    // NOTE: This field must be added to the struct definition:
    // pub v_branch_weights: Vec<f32>,

    fn normalize_branch_weights_ascii(v_w: &mut Vec<f32>) {
        if v_w.is_empty() {
            return;
        }

        let mut d_sum: f32 = 0.0;
        for w in v_w.iter_mut() {
            if !w.is_finite() || *w < 0.0 {
                *w = 0.0;
            }
            d_sum += *w;
        }

        if !d_sum.is_finite() || d_sum <= 0.0 {
            let d_u = 1.0 / (v_w.len() as f32).max(1.0);
            for w in v_w.iter_mut() {
                *w = d_u;
            }
            return;
        }

        for w in v_w.iter_mut() {
            *w /= d_sum;
        }
    }

    pub fn get_branch_weights_ascii(&self) -> Vec<f32> {
        self.v_branch_weights.clone()
    }

    pub fn set_branch_weights_ascii(&mut self, v_w: &[f32]) -> Result<(), String> {
        if v_w.len() != self.v_branches.len() {
            return Err("parallel_block_group_weights_len_mismatch".to_string());
        }
        let mut v_new = v_w.to_vec();
        Self::normalize_branch_weights_ascii(&mut v_new);
        self.v_branch_weights = v_new;
        Ok(())
    }

    // Conservative weight injection when adding branches:
    // old weights scaled by (1-eta), new branches share eta uniformly.
    pub fn add_branches_with_conservative_injection_ascii(
        &mut self,
        mut v_new_branches: Vec<Box<dyn Layer>>,
        d_eta_injection: f32,
    ) -> Result<(), String> {
        if v_new_branches.is_empty() {
            return Err("parallel_block_group_add_branches_empty".to_string());
        }
        let d_eta = if d_eta_injection.is_finite() {
            d_eta_injection.clamp(0.0, 0.5)
        } else {
            0.0
        };

        let i_old = self.v_branches.len();
        let i_add = v_new_branches.len();
        let i_new_total = i_old.saturating_add(i_add);
        if i_new_total == 0 {
            return Err("parallel_block_group_add_branches_new_total_zero".to_string());
        }

        // Ensure weights exist.
        if self.v_branch_weights.len() != i_old {
            self.v_branch_weights = vec![1.0 / (i_old as f32).max(1.0); i_old.max(1)];
            Self::normalize_branch_weights_ascii(&mut self.v_branch_weights);
        }

        // Scale old weights by (1-eta).
        for w in self.v_branch_weights.iter_mut() {
            *w *= 1.0 - d_eta;
        }

        // Append new branches and their initial weights.
        let d_new_share = if i_add == 0 { 0.0 } else { d_eta / (i_add as f32).max(1.0) };
        for _ in 0..i_add {
            self.v_branch_weights.push(d_new_share);
        }

        self.v_branches.append(&mut v_new_branches);

        Self::normalize_branch_weights_ascii(&mut self.v_branch_weights);
        Ok(())
    }

    // Weighted aggregation forward with availability mask.
    pub fn forward_weighted_with_availability_mask_ascii(
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
        if self.v_branch_weights.len() != self.v_branches.len() {
            return Array2::zeros((0, 0));
        }

        // Parallel forward on active branches.
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

        // Weighted sum; renormalize over active weights to preserve scale.
        let mut opt_sum: Option<Array2<f32>> = None;
        let mut d_active_w_sum: f32 = 0.0;

        for (i_idx, opt_a) in v_outs.into_iter().enumerate() {
            let a_y = match opt_a {
                Some(a) => a,
                None => continue,
            };
            let d_w = self.v_branch_weights[i_idx];
            if !d_w.is_finite() || d_w <= 0.0 {
                continue;
            }

            match &mut opt_sum {
                None => {
                    let mut a_first = a_y;
                    a_first.mapv_inplace(|x| x * d_w);
                    opt_sum = Some(a_first);
                }
                Some(a_acc) => {
                    if a_acc.raw_dim() != a_y.raw_dim() {
                        return Array2::zeros((0, 0));
                    }
                    *a_acc = &*a_acc + &(a_y.mapv(|x| x * d_w));
                }
            }

            d_active_w_sum += d_w;
        }

        let mut a_out = match opt_sum {
            None => Array2::zeros((0, 0)),
            Some(a) => a,
        };

        // Renormalize to sum of active weights (avoid shrinking when only few paths active).
        if d_active_w_sum.is_finite() && d_active_w_sum > 1e-12 {
            let d_inv = 1.0 / d_active_w_sum;
            a_out.mapv_inplace(|x| x * d_inv);
        }

        a_out
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
#[derive(Clone, Debug)]
pub enum TrainingDataEventAscii {
    // Add a file that contains JSON array of strings (same semantics as existing dataset JSON).
    add_training_file_json_array { s_path: String },

    // Add already loaded rows directly (each element is one training example string).
    add_training_rows { v_rows: Vec<String> },

    // Signal shutdown for cooperative termination of ingestion loop.
    shutdown,
}

#[derive(Clone, Debug)]
struct online_data_ingestion_state_ascii {
    i_max_file_bytes: usize,
    i_max_rows_per_event: usize,
    i_max_total_rows: usize,

    i_total_rows_added: usize,
    i_total_events_processed: usize,
    i_total_parse_errors: usize,
    i_total_rows_rejected: usize,

    // New: pending queue proxy, window stats.
    i_pending_events_observed_peak: usize,
    u64_window_start_ms: u64,
    i_rows_added_window: usize,
    i_events_processed_window: usize,
}

impl online_data_ingestion_state_ascii {
    fn new_default() -> Self {
        Self {
            i_max_file_bytes: 50 * 1024 * 1024,
            i_max_rows_per_event: 200_000,
            i_max_total_rows: 5_000_000,

            i_total_rows_added: 0,
            i_total_events_processed: 0,
            i_total_parse_errors: 0,
            i_total_rows_rejected: 0,

            i_pending_events_observed_peak: 0,
            u64_window_start_ms: 0,
            i_rows_added_window: 0,
            i_events_processed_window: 0,
        }
    }
}


fn read_json_array_of_strings_file_ascii(
    s_path: &str,
    i_max_file_bytes: usize,
) -> Result<Vec<String>, String> {
    if s_path.trim().is_empty() {
        return Err("ingest_path_empty".to_string());
    }

    let md = std::fs::metadata(s_path).map_err(|_| "ingest_file_metadata_error".to_string())?;
    let i_len = md.len() as usize;
    if i_len == 0 {
        return Err("ingest_file_empty".to_string());
    }
    if i_len > i_max_file_bytes {
        return Err("ingest_file_too_large".to_string());
    }

    let s_json = std::fs::read_to_string(s_path).map_err(|_| "ingest_file_read_error".to_string())?;
    if s_json.trim().is_empty() {
        return Err("ingest_file_read_empty".to_string());
    }

    let v_data: Vec<String> =
        serde_json::from_str(&s_json).map_err(|_| "ingest_json_parse_error".to_string())?;
    Ok(v_data)
}

fn append_tokenized_rows_ascii(
    llm: &Llm,
    v_dst_token_rows: &mut Vec<Vec<usize>>,
    v_new_rows: &[String],
    st_ing: &mut online_data_ingestion_state_ascii,
) {
    if v_new_rows.is_empty() {
        return;
    }

    let i_budget = st_ing
        .i_max_total_rows
        .saturating_sub(v_dst_token_rows.len());
    if i_budget == 0 {
        st_ing.i_total_rows_rejected = st_ing.i_total_rows_rejected.saturating_add(v_new_rows.len());
        return;
    }

    let i_take = v_new_rows
        .len()
        .min(st_ing.i_max_rows_per_event)
        .min(i_budget);

    for s in v_new_rows.iter().take(i_take) {
        if s.trim().is_empty() {
            st_ing.i_total_rows_rejected = st_ing.i_total_rows_rejected.saturating_add(1);
            continue;
        }

        let r_tok = llm.tokenize(s);
        let v_ids = match r_tok {
            Ok(v) => v,
            Err(_) => {
                st_ing.i_total_parse_errors = st_ing.i_total_parse_errors.saturating_add(1);
                continue;
            }
        };

        if v_ids.len() < 2 {
            st_ing.i_total_rows_rejected = st_ing.i_total_rows_rejected.saturating_add(1);
            continue;
        }

        v_dst_token_rows.push(v_ids);
        st_ing.i_total_rows_added = st_ing.i_total_rows_added.saturating_add(1);
    }
}
fn now_ms_ascii() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/* ------------------------------ Helpers for drift proxy ------------------------------ */

fn logits_distance_l2_ascii(a_a: &Array2<f32>, a_b: &Array2<f32>) -> f32 {
    if a_a.raw_dim() != a_b.raw_dim() || a_a.len() == 0 {
        return 0.0;
    }
    let mut d_sum: f32 = 0.0;
    for (x, y) in a_a.iter().zip(a_b.iter()) {
        let dx = math::sanitize_f32(*x);
        let dy = math::sanitize_f32(*y);
        let d = dx - dy;
        d_sum += d * d;
    }
    math::sanitize_f32(d_sum.sqrt())
}

fn logits_cosine_distance_ascii(a_a: &Array2<f32>, a_b: &Array2<f32>) -> f32 {
    if a_a.raw_dim() != a_b.raw_dim() || a_a.len() == 0 {
        return 0.0;
    }
    let v_a: Vec<f32> = a_a.iter().map(|&d| math::sanitize_f32(d)).collect();
    let v_b: Vec<f32> = a_b.iter().map(|&d| math::sanitize_f32(d)).collect();
    let d_cos = math::cosine_similarity_f32(&v_a, &v_b);
    math::sanitize_f32(1.0 - d_cos)
}

fn drain_training_data_events_non_blocking_ascii(
    llm: &Llm,
    opt_rx: &mut Option<Receiver<TrainingDataEventAscii>>,
    v_tokenized_data: &mut Vec<Vec<usize>>,
    st_ing: &mut online_data_ingestion_state_ascii,
    met_ing: &mut ingestion_metrics_ascii,
) {
    let rx = match opt_rx.as_mut() {
        Some(v) => v,
        None => return,
    };

    let u64_t0 = now_ms_ascii();
    if st_ing.u64_window_start_ms == 0 {
        st_ing.u64_window_start_ms = u64_t0;
    }

    // Queue proxy: drain until empty; count how many drain batches happen.
    let mut i_local_batches: usize = 0;
    let mut i_pending_probe: usize = 0;

    loop {
        // Approximate pending count by repeated try_recv attempts; not exact but a proxy.
        i_pending_probe = i_pending_probe.saturating_add(1);
        let ev = match rx.try_recv() {
            Ok(v) => v,
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                *opt_rx = None;
                break;
            }
        };

        i_local_batches = i_local_batches.saturating_add(1);

        st_ing.i_total_events_processed = st_ing.i_total_events_processed.saturating_add(1);
        st_ing.i_events_processed_window = st_ing.i_events_processed_window.saturating_add(1);

        match ev {
            TrainingDataEventAscii::shutdown => {
                *opt_rx = None;
                break;
            }
            TrainingDataEventAscii::add_training_rows { v_rows } => {
                let i_before = st_ing.i_total_rows_added;
                append_tokenized_rows_ascii(llm, v_tokenized_data, &v_rows, st_ing);
                let i_after = st_ing.i_total_rows_added;
                st_ing.i_rows_added_window = st_ing
                    .i_rows_added_window
                    .saturating_add(i_after.saturating_sub(i_before));
            }
            TrainingDataEventAscii::add_training_file_json_array { s_path } => {
                match read_json_array_of_strings_file_ascii(&s_path, st_ing.i_max_file_bytes) {
                    Ok(v_rows) => {
                        let i_before = st_ing.i_total_rows_added;
                        append_tokenized_rows_ascii(llm, v_tokenized_data, &v_rows, st_ing);
                        let i_after = st_ing.i_total_rows_added;
                        st_ing.i_rows_added_window = st_ing
                            .i_rows_added_window
                            .saturating_add(i_after.saturating_sub(i_before));
                    }
                    Err(_) => st_ing.i_total_parse_errors = st_ing.i_total_parse_errors.saturating_add(1),
                }
            }
        }
    }

    st_ing.i_pending_events_observed_peak = st_ing.i_pending_events_observed_peak.max(i_pending_probe);

    // Export ingestion metrics snapshot each drain call (windowed rates).
    let u64_t1 = now_ms_ascii();
    let d_dt = ((u64_t1.saturating_sub(st_ing.u64_window_start_ms)) as f32) / 1000.0;
    let d_dt_safe = d_dt.max(1e-6);

    met_ing.i_events_processed_total = st_ing.i_total_events_processed;
    met_ing.i_rows_added_total = st_ing.i_total_rows_added;
    met_ing.i_rows_rejected_total = st_ing.i_total_rows_rejected;
    met_ing.i_parse_errors_total = st_ing.i_total_parse_errors;

    met_ing.i_rows_added_window = st_ing.i_rows_added_window;
    met_ing.i_events_processed_window = st_ing.i_events_processed_window;
    met_ing.u64_window_start_ms = st_ing.u64_window_start_ms;
    met_ing.u64_window_end_ms = u64_t1;

    met_ing.d_rows_per_sec_window = (st_ing.i_rows_added_window as f32) / d_dt_safe;
    met_ing.d_events_per_sec_window = (st_ing.i_events_processed_window as f32) / d_dt_safe;

    met_ing.i_pending_events_observed_peak = met_ing.i_pending_events_observed_peak.max(st_ing.i_pending_events_observed_peak);
    met_ing.i_last_drain_batches = i_local_batches;

    // Reset window if it grows too large.
    if d_dt > 5.0 {
        st_ing.u64_window_start_ms = u64_t1;
        st_ing.i_rows_added_window = 0;
        st_ing.i_events_processed_window = 0;
    }
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
/* ------------------------------ Metrics structs ------------------------------ */

#[derive(Clone, Debug)]
pub struct ingestion_metrics_ascii {
    // Throughput and queue states.
    pub i_events_processed_total: usize,
    pub i_rows_added_total: usize,
    pub i_rows_rejected_total: usize,
    pub i_parse_errors_total: usize,

    // Derived rates and last window info.
    pub d_rows_per_sec_window: f32,
    pub d_events_per_sec_window: f32,
    pub i_rows_added_window: usize,
    pub i_events_processed_window: usize,
    pub u64_window_start_ms: u64,
    pub u64_window_end_ms: u64,

    // Queue proxy counters.
    pub i_pending_events_observed_peak: usize,
    pub i_last_drain_batches: usize,
}

impl ingestion_metrics_ascii {
    pub fn new() -> Self {
        Self {
            i_events_processed_total: 0,
            i_rows_added_total: 0,
            i_rows_rejected_total: 0,
            i_parse_errors_total: 0,
            d_rows_per_sec_window: 0.0,
            d_events_per_sec_window: 0.0,
            i_rows_added_window: 0,
            i_events_processed_window: 0,
            u64_window_start_ms: 0,
            u64_window_end_ms: 0,
            i_pending_events_observed_peak: 0,
            i_last_drain_batches: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct coverage_metrics_ascii {
    pub i_epoch_token_rows_start: usize,
    pub i_epoch_token_rows_end: usize,
    pub i_epoch_rows_used: usize,
    pub d_coverage_ratio_used_over_available: f32,
    pub d_new_data_ratio_in_available: f32,
    pub i_new_rows_added_during_epoch: usize,
}

impl coverage_metrics_ascii {
    pub fn new() -> Self {
        Self {
            i_epoch_token_rows_start: 0,
            i_epoch_token_rows_end: 0,
            i_epoch_rows_used: 0,
            d_coverage_ratio_used_over_available: 0.0,
            d_new_data_ratio_in_available: 0.0,
            i_new_rows_added_during_epoch: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct mask_participation_metrics_ascii {
    pub i_steps_observed: usize,
    pub i_active_branches_min: usize,
    pub i_active_branches_max: usize,
    pub d_active_branches_mean: f32,
    pub d_active_branches_m2: f32, // Welford variance helper.
    pub i_steps_at_min_active: usize,

    pub d_mask_sparsity_mean: f32, // fraction inactive.
    pub d_mask_sparsity_m2: f32,
}

impl mask_participation_metrics_ascii {
    pub fn new() -> Self {
        Self {
            i_steps_observed: 0,
            i_active_branches_min: usize::MAX,
            i_active_branches_max: 0,
            d_active_branches_mean: 0.0,
            d_active_branches_m2: 0.0,
            i_steps_at_min_active: 0,
            d_mask_sparsity_mean: 0.0,
            d_mask_sparsity_m2: 0.0,
        }
    }

    pub fn update(&mut self, i_active: usize, i_total: usize, i_min_active: usize) {
        if i_total == 0 {
            return;
        }

        let d_sparsity = 1.0 - (i_active as f32) / (i_total as f32).max(1.0);

        self.i_steps_observed = self.i_steps_observed.saturating_add(1);
        self.i_active_branches_min = self.i_active_branches_min.min(i_active);
        self.i_active_branches_max = self.i_active_branches_max.max(i_active);

        if i_active == i_min_active {
            self.i_steps_at_min_active = self.i_steps_at_min_active.saturating_add(1);
        }

        // Welford mean/variance for active branches.
        let d_x = i_active as f32;
        let d_n = self.i_steps_observed as f32;
        let d_delta = d_x - self.d_active_branches_mean;
        self.d_active_branches_mean += d_delta / d_n.max(1.0);
        let d_delta2 = d_x - self.d_active_branches_mean;
        self.d_active_branches_m2 += d_delta * d_delta2;

        // Welford for sparsity.
        let d_y = d_sparsity;
        let d_delta_s = d_y - self.d_mask_sparsity_mean;
        self.d_mask_sparsity_mean += d_delta_s / d_n.max(1.0);
        let d_delta_s2 = d_y - self.d_mask_sparsity_mean;
        self.d_mask_sparsity_m2 += d_delta_s * d_delta_s2;
    }

    pub fn active_branches_std(&self) -> f32 {
        if self.i_steps_observed < 2 {
            return 0.0;
        }
        let d_var = self.d_active_branches_m2 / ((self.i_steps_observed - 1) as f32).max(1.0);
        math::sanitize_f32(d_var).sqrt()
    }

    pub fn sparsity_std(&self) -> f32 {
        if self.i_steps_observed < 2 {
            return 0.0;
        }
        let d_var = self.d_mask_sparsity_m2 / ((self.i_steps_observed - 1) as f32).max(1.0);
        math::sanitize_f32(d_var).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct inverse_participation_scaling_metrics_ascii {
    // Proxy: compare grad norm with scaling vs without scaling for same mask and same row.
    pub i_samples: usize,
    pub d_grad_norm_ratio_mean: f32,
    pub d_grad_norm_ratio_m2: f32,
    pub d_grad_norm_scaled_mean: f32,
    pub d_grad_norm_unscaled_mean: f32,
}

impl inverse_participation_scaling_metrics_ascii {
    pub fn new() -> Self {
        Self {
            i_samples: 0,
            d_grad_norm_ratio_mean: 0.0,
            d_grad_norm_ratio_m2: 0.0,
            d_grad_norm_scaled_mean: 0.0,
            d_grad_norm_unscaled_mean: 0.0,
        }
    }

    pub fn update(&mut self, d_scaled: f32, d_unscaled: f32) {
        if !d_scaled.is_finite() || !d_unscaled.is_finite() || d_unscaled <= 1e-12 {
            return;
        }
        let d_ratio = d_scaled / d_unscaled;

        self.i_samples = self.i_samples.saturating_add(1);
        let d_n = self.i_samples as f32;

        // Ratio Welford.
        let d_delta = d_ratio - self.d_grad_norm_ratio_mean;
        self.d_grad_norm_ratio_mean += d_delta / d_n.max(1.0);
        let d_delta2 = d_ratio - self.d_grad_norm_ratio_mean;
        self.d_grad_norm_ratio_m2 += d_delta * d_delta2;

        // Means.
        self.d_grad_norm_scaled_mean += (d_scaled - self.d_grad_norm_scaled_mean) / d_n.max(1.0);
        self.d_grad_norm_unscaled_mean += (d_unscaled - self.d_grad_norm_unscaled_mean) / d_n.max(1.0);
    }

    pub fn ratio_std(&self) -> f32 {
        if self.i_samples < 2 {
            return 0.0;
        }
        let d_var = self.d_grad_norm_ratio_m2 / ((self.i_samples - 1) as f32).max(1.0);
        math::sanitize_f32(d_var).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct replay_metrics_ascii {
    pub i_replay_steps_total: usize,
    pub i_fresh_steps_total: usize,
    pub d_replay_p_last: f32,

    // Effect strength proxy:
    // delta_loss = loss_replay - loss_fresh (positive means replay is worse).
    pub i_pairs: usize,
    pub d_delta_loss_mean: f32,
    pub d_delta_loss_m2: f32,
}

impl replay_metrics_ascii {
    pub fn new() -> Self {
        Self {
            i_replay_steps_total: 0,
            i_fresh_steps_total: 0,
            d_replay_p_last: 0.0,
            i_pairs: 0,
            d_delta_loss_mean: 0.0,
            d_delta_loss_m2: 0.0,
        }
    }

    pub fn inc_fresh(&mut self) {
        self.i_fresh_steps_total = self.i_fresh_steps_total.saturating_add(1);
    }

    pub fn inc_replay(&mut self) {
        self.i_replay_steps_total = self.i_replay_steps_total.saturating_add(1);
    }

    pub fn update_delta_loss(&mut self, d_loss_fresh: f32, d_loss_replay: f32) {
        if !d_loss_fresh.is_finite() || !d_loss_replay.is_finite() {
            return;
        }
        let d_delta = d_loss_replay - d_loss_fresh;
        self.i_pairs = self.i_pairs.saturating_add(1);
        let d_n = self.i_pairs as f32;

        let d_d1 = d_delta - self.d_delta_loss_mean;
        self.d_delta_loss_mean += d_d1 / d_n.max(1.0);
        let d_d2 = d_delta - self.d_delta_loss_mean;
        self.d_delta_loss_m2 += d_d1 * d_d2;
    }

    pub fn delta_loss_std(&self) -> f32 {
        if self.i_pairs < 2 {
            return 0.0;
        }
        let d_var = self.d_delta_loss_m2 / ((self.i_pairs - 1) as f32).max(1.0);
        math::sanitize_f32(d_var).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct retention_metrics_ascii {
    // Fixed control sets evaluation.
    pub d_loss_control_old: f32,
    pub d_loss_control_new: f32,
    pub d_retention_delta_old: f32,
    pub d_retention_delta_new: f32,
    pub u64_last_eval_ms: u64,
}

impl retention_metrics_ascii {
    pub fn new() -> Self {
        Self {
            d_loss_control_old: 0.0,
            d_loss_control_new: 0.0,
            d_retention_delta_old: 0.0,
            d_retention_delta_new: 0.0,
            u64_last_eval_ms: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct branch_fairness_metrics_ascii {
    // Selection and dominance.
    pub v_selected_count: Vec<usize>,
    pub i_total_selections: usize,
    pub d_gini_selected: f32,
    pub d_top1_share_selected: f32,
    pub i_starvation_steps_max: usize,
}

impl branch_fairness_metrics_ascii {
    pub fn new(i_k: usize) -> Self {
        Self {
            v_selected_count: vec![0; i_k.max(1)],
            i_total_selections: 0,
            d_gini_selected: 0.0,
            d_top1_share_selected: 0.0,
            i_starvation_steps_max: 0,
        }
    }

    pub fn ensure_len(&mut self, i_k: usize) {
        if self.v_selected_count.len() != i_k {
            self.v_selected_count = vec![0; i_k.max(1)];
            self.i_total_selections = 0;
            self.d_gini_selected = 0.0;
            self.d_top1_share_selected = 0.0;
            self.i_starvation_steps_max = 0;
        }
    }

    pub fn on_select(&mut self, i_branch: usize) {
        if i_branch >= self.v_selected_count.len() {
            return;
        }
        self.v_selected_count[i_branch] = self.v_selected_count[i_branch].saturating_add(1);
        self.i_total_selections = self.i_total_selections.saturating_add(1);
    }

    pub fn recompute(&mut self) {
        if self.v_selected_count.is_empty() {
            return;
        }
        let d_sum: f32 = self.v_selected_count.iter().map(|&c| c as f32).sum();
        if d_sum <= 0.0 || !d_sum.is_finite() {
            self.d_gini_selected = 0.0;
            self.d_top1_share_selected = 0.0;
            return;
        }
        let v_p: Vec<f32> = self.v_selected_count.iter().map(|&c| (c as f32) / d_sum).collect();
        self.d_gini_selected = math::gini_coefficient_f32(&v_p);

        let mut v_sorted = v_p.clone();
        v_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        self.d_top1_share_selected = *v_sorted.get(0).unwrap_or(&0.0);
    }
}

#[derive(Clone, Debug)]
pub struct snapshot_metrics_ascii {
    pub i_snapshots_sent_total: usize,
    pub i_snapshots_applied_total: usize,

    // Latency and staleness.
    pub d_latency_ms_last: f32,
    pub d_latency_ms_mean: f32,
    pub d_latency_ms_m2: f32,
    pub i_latency_samples: usize,

    pub i_staleness_steps_last: usize,
    pub d_staleness_steps_mean: f32,
    pub d_staleness_steps_m2: f32,
    pub i_staleness_samples: usize,

    // Producer tracking.
    pub i_train_step_last_sent: usize,
    pub u64_last_send_ms: u64,
}

impl snapshot_metrics_ascii {
    pub fn new() -> Self {
        Self {
            i_snapshots_sent_total: 0,
            i_snapshots_applied_total: 0,
            d_latency_ms_last: 0.0,
            d_latency_ms_mean: 0.0,
            d_latency_ms_m2: 0.0,
            i_latency_samples: 0,
            i_staleness_steps_last: 0,
            d_staleness_steps_mean: 0.0,
            d_staleness_steps_m2: 0.0,
            i_staleness_samples: 0,
            i_train_step_last_sent: 0,
            u64_last_send_ms: 0,
        }
    }

    pub fn on_send(&mut self, i_train_step: usize, u64_now_ms: u64) {
        self.i_snapshots_sent_total = self.i_snapshots_sent_total.saturating_add(1);
        self.i_train_step_last_sent = i_train_step;
        self.u64_last_send_ms = u64_now_ms;
    }

    pub fn on_apply(&mut self, u64_apply_ms: u64, i_train_step_now: usize, i_snapshot_step: usize) {
        self.i_snapshots_applied_total = self.i_snapshots_applied_total.saturating_add(1);

        // Latency in ms between send and apply if send timestamp exists.
        if self.u64_last_send_ms > 0 && u64_apply_ms >= self.u64_last_send_ms {
            let d_lat = (u64_apply_ms - self.u64_last_send_ms) as f32;
            if d_lat.is_finite() {
                self.d_latency_ms_last = d_lat;
                self.i_latency_samples = self.i_latency_samples.saturating_add(1);
                let d_n = self.i_latency_samples as f32;
                let d_delta = d_lat - self.d_latency_ms_mean;
                self.d_latency_ms_mean += d_delta / d_n.max(1.0);
                let d_delta2 = d_lat - self.d_latency_ms_mean;
                self.d_latency_ms_m2 += d_delta * d_delta2;
            }
        }

        // Staleness in steps at time of apply.
        let i_stale = i_train_step_now.saturating_sub(i_snapshot_step);
        self.i_staleness_steps_last = i_stale;
        self.i_staleness_samples = self.i_staleness_samples.saturating_add(1);
        let d_n2 = self.i_staleness_samples as f32;
        let d_x = i_stale as f32;
        let d_delta_s = d_x - self.d_staleness_steps_mean;
        self.d_staleness_steps_mean += d_delta_s / d_n2.max(1.0);
        let d_delta_s2 = d_x - self.d_staleness_steps_mean;
        self.d_staleness_steps_m2 += d_delta_s * d_delta_s2;
    }

    pub fn latency_std(&self) -> f32 {
        if self.i_latency_samples < 2 {
            return 0.0;
        }
        let d_var = self.d_latency_ms_m2 / ((self.i_latency_samples - 1) as f32).max(1.0);
        math::sanitize_f32(d_var).sqrt()
    }

    pub fn staleness_std(&self) -> f32 {
        if self.i_staleness_samples < 2 {
            return 0.0;
        }
        let d_var = self.d_staleness_steps_m2 / ((self.i_staleness_samples - 1) as f32).max(1.0);
        math::sanitize_f32(d_var).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct expansion_metrics_ascii {
    pub i_expansion_events_total: usize,
    pub i_branches_before_last: usize,
    pub i_branches_after_last: usize,
    pub d_eta_injection_last: f32,
    pub d_sum_w_new_last: f32,
    pub u64_last_event_ms: u64,
}

impl expansion_metrics_ascii {
    pub fn new() -> Self {
        Self {
            i_expansion_events_total: 0,
            i_branches_before_last: 0,
            i_branches_after_last: 0,
            d_eta_injection_last: 0.0,
            d_sum_w_new_last: 0.0,
            u64_last_event_ms: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct drift_metrics_ascii {
    // Proxy distances between model logits before and after expansion.
    pub i_drift_samples_total: usize,
    pub d_logits_l2_mean: f32,
    pub d_logits_l2_m2: f32,
    pub d_logits_cos_dist_mean: f32,
    pub d_logits_cos_dist_m2: f32,
}

impl drift_metrics_ascii {
    pub fn new() -> Self {
        Self {
            i_drift_samples_total: 0,
            d_logits_l2_mean: 0.0,
            d_logits_l2_m2: 0.0,
            d_logits_cos_dist_mean: 0.0,
            d_logits_cos_dist_m2: 0.0,
        }
    }

    pub fn update(&mut self, d_l2: f32, d_cos_dist: f32) {
        if !d_l2.is_finite() || !d_cos_dist.is_finite() {
            return;
        }
        self.i_drift_samples_total = self.i_drift_samples_total.saturating_add(1);
        let d_n = self.i_drift_samples_total as f32;

        let d_dl = d_l2 - self.d_logits_l2_mean;
        self.d_logits_l2_mean += d_dl / d_n.max(1.0);
        let d_dl2 = d_l2 - self.d_logits_l2_mean;
        self.d_logits_l2_m2 += d_dl * d_dl2;

        let d_dc = d_cos_dist - self.d_logits_cos_dist_mean;
        self.d_logits_cos_dist_mean += d_dc / d_n.max(1.0);
        let d_dc2 = d_cos_dist - self.d_logits_cos_dist_mean;
        self.d_logits_cos_dist_m2 += d_dc * d_dc2;
    }

    pub fn l2_std(&self) -> f32 {
        if self.i_drift_samples_total < 2 {
            return 0.0;
        }
        let d_var = self.d_logits_l2_m2 / ((self.i_drift_samples_total - 1) as f32).max(1.0);
        math::sanitize_f32(d_var).sqrt()
    }

    pub fn cos_dist_std(&self) -> f32 {
        if self.i_drift_samples_total < 2 {
            return 0.0;
        }
        let d_var = self.d_logits_cos_dist_m2 / ((self.i_drift_samples_total - 1) as f32).max(1.0);
        math::sanitize_f32(d_var).sqrt()
    }
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

    pub i_skips_empty_act: usize,
    pub i_skips_empty_logits: usize,
    pub i_skips_pg_downcast_failed: usize,
    pub i_skips_pg_no_branches: usize,

    // ---- New metrics: keep ASCII names ----

    // 1) Ingestion throughput and queue proxy.
    pub d_ingest_rows_per_sec_window: f32,
    pub d_ingest_events_per_sec_window: f32,
    pub i_ingest_rows_added_total: usize,
    pub i_ingest_events_processed_total: usize,
    pub i_ingest_parse_errors_total: usize,
    pub i_ingest_rows_rejected_total: usize,
    pub i_ingest_pending_events_observed_peak: usize,

    // 2) Coverage ratio.
    pub d_coverage_ratio_used_over_available: f32,
    pub d_new_data_ratio_in_available: f32,
    pub i_new_rows_added_during_epoch: usize,
    pub i_epoch_token_rows_start: usize,
    pub i_epoch_token_rows_end: usize,

    // 3) Mask participation stats.
    pub d_active_branches_mean: f32,
    pub d_active_branches_std: f32,
    pub i_active_branches_min: usize,
    pub i_active_branches_max: usize,
    pub d_mask_sparsity_mean: f32,
    pub d_mask_sparsity_std: f32,
    pub d_steps_at_min_active_share: f32,

    // 4) Inverse participation scaling impact.
    pub d_grad_norm_ratio_scaled_over_unscaled_mean: f32,
    pub d_grad_norm_ratio_scaled_over_unscaled_std: f32,
    pub d_grad_norm_scaled_mean: f32,
    pub d_grad_norm_unscaled_mean: f32,

    // 5) Replay usage and effect.
    pub d_replay_share: f32,
    pub d_replay_p_last: f32,
    pub d_replay_delta_loss_mean: f32,
    pub d_replay_delta_loss_std: f32,

    // 6) Retention score on fixed control sets.
    pub d_loss_control_old: f32,
    pub d_loss_control_new: f32,
    pub d_retention_delta_old: f32,
    pub d_retention_delta_new: f32,

    // 7) Branch selection fairness.
    pub d_branch_select_gini: f32,
    pub d_branch_select_top1_share: f32,

    // 8) Snapshot latency and staleness.
    pub d_snapshot_latency_ms_last: f32,
    pub d_snapshot_latency_ms_mean: f32,
    pub d_snapshot_latency_ms_std: f32,
    pub i_snapshot_staleness_steps_last: usize,
    pub d_snapshot_staleness_steps_mean: f32,
    pub d_snapshot_staleness_steps_std: f32,
    pub i_snapshots_sent_total: usize,

    // 9) Expansion telemetry.
    pub i_expansion_events_total: usize,
    pub i_branches_before_last_expand: usize,
    pub i_branches_after_last_expand: usize,
    pub d_eta_injection_last: f32,
    pub d_sum_w_new_last: f32,

    // 10) Output drift bound proxy.
    pub d_expand_drift_logits_l2_mean: f32,
    pub d_expand_drift_logits_l2_std: f32,
    pub d_expand_drift_logits_cos_dist_mean: f32,
    pub d_expand_drift_logits_cos_dist_std: f32,

    // EMA state.
    pub b_ema_active: bool,
    pub i_ema_last_selected_branch: isize,
}
impl TrainingProgressEventAscii {
    pub fn new_basic(
        s_phase: &str,
        i_epoch_current: usize,
        i_epochs_total: usize,
        d_last_epoch_loss: f32,
        d_last_step_loss: f32,
        i_rows_used_last_epoch: usize,
        i_total_steps: usize,
    ) -> Self {
        Self {
            s_phase: s_phase.to_string(),
            i_epoch_current,
            i_epochs_total,
            d_last_epoch_loss,
            d_last_step_loss,
            i_rows_used_last_epoch,
            i_total_steps,

            i_skips_empty_act: 0,
            i_skips_empty_logits: 0,
            i_skips_pg_downcast_failed: 0,
            i_skips_pg_no_branches: 0,

            d_ingest_rows_per_sec_window: 0.0,
            d_ingest_events_per_sec_window: 0.0,
            i_ingest_rows_added_total: 0,
            i_ingest_events_processed_total: 0,
            i_ingest_parse_errors_total: 0,
            i_ingest_rows_rejected_total: 0,
            i_ingest_pending_events_observed_peak: 0,

            d_coverage_ratio_used_over_available: 0.0,
            d_new_data_ratio_in_available: 0.0,
            i_new_rows_added_during_epoch: 0,
            i_epoch_token_rows_start: 0,
            i_epoch_token_rows_end: 0,

            d_active_branches_mean: 0.0,
            d_active_branches_std: 0.0,
            i_active_branches_min: 0,
            i_active_branches_max: 0,
            d_mask_sparsity_mean: 0.0,
            d_mask_sparsity_std: 0.0,
            d_steps_at_min_active_share: 0.0,

            d_grad_norm_ratio_scaled_over_unscaled_mean: 0.0,
            d_grad_norm_ratio_scaled_over_unscaled_std: 0.0,
            d_grad_norm_scaled_mean: 0.0,
            d_grad_norm_unscaled_mean: 0.0,

            d_replay_share: 0.0,
            d_replay_p_last: 0.0,
            d_replay_delta_loss_mean: 0.0,
            d_replay_delta_loss_std: 0.0,

            d_loss_control_old: 0.0,
            d_loss_control_new: 0.0,
            d_retention_delta_old: 0.0,
            d_retention_delta_new: 0.0,

            d_branch_select_gini: 0.0,
            d_branch_select_top1_share: 0.0,

            d_snapshot_latency_ms_last: 0.0,
            d_snapshot_latency_ms_mean: 0.0,
            d_snapshot_latency_ms_std: 0.0,
            i_snapshot_staleness_steps_last: 0,
            d_snapshot_staleness_steps_mean: 0.0,
            d_snapshot_staleness_steps_std: 0.0,
            i_snapshots_sent_total: 0,

            i_expansion_events_total: 0,
            i_branches_before_last_expand: 0,
            i_branches_after_last_expand: 0,
            d_eta_injection_last: 0.0,
            d_sum_w_new_last: 0.0,

            d_expand_drift_logits_l2_mean: 0.0,
            d_expand_drift_logits_l2_std: 0.0,
            d_expand_drift_logits_cos_dist_mean: 0.0,
            d_expand_drift_logits_cos_dist_std: 0.0,

            b_ema_active: false,
            i_ema_last_selected_branch: -1,
        }
    }
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


#[derive(Clone, Debug)]
struct branch_loss_ema_state_ascii {
    // Exponential moving average per branch. None means uninitialized.
    v_ema_loss: Vec<Option<f32>>,
    // Last chosen branch (optional diagnostic).
    opt_last_selected_branch: Option<usize>,
    // EMA smoothing factor in (0, 1]. Higher means more reactive.
    d_alpha: f32,
}

impl branch_loss_ema_state_ascii {
    fn new(i_num_branches: usize, d_alpha: f32) -> Self {
        let d_a = if d_alpha.is_finite() { d_alpha.clamp(0.01, 1.0) } else { 0.2 };
        Self {
            v_ema_loss: vec![None; i_num_branches],
            opt_last_selected_branch: None,
            d_alpha: d_a,
        }
    }

    fn ensure_len(&mut self, i_num_branches: usize) {
        if self.v_ema_loss.len() == i_num_branches {
            return;
        }
        self.v_ema_loss = vec![None; i_num_branches];
        self.opt_last_selected_branch = None;
    }

    fn update_ema(&mut self, i_branch: usize, d_loss: f32) {
        if i_branch >= self.v_ema_loss.len() {
            return;
        }
        if !d_loss.is_finite() || d_loss < 0.0 {
            return;
        }

        let d_alpha = self.d_alpha;
        match self.v_ema_loss[i_branch] {
            None => self.v_ema_loss[i_branch] = Some(d_loss),
            Some(d_prev) => {
                let d_new = d_alpha * d_loss + (1.0 - d_alpha) * d_prev;
                self.v_ema_loss[i_branch] = Some(if d_new.is_finite() { d_new } else { d_prev });
            }
        }
    }

    fn get_score_or_fallback(&self, i_branch: usize, d_fallback: f32) -> f32 {
        if i_branch >= self.v_ema_loss.len() {
            return d_fallback;
        }
        match self.v_ema_loss[i_branch] {
            Some(v) if v.is_finite() => v,
            _ => d_fallback,
        }
    }
}

// Description: Patch - Add simple experience replay buffer for continual learning.
// History:
// - 2026-02-13: Add replay buffer to reduce catastrophic forgetting.
// Author: Marcus Schlieper

#[derive(Clone, Debug)]
struct replay_buffer_ascii {
    // Stores token id sequences (already truncated to MAX_SEQ_LEN).
    v_rows: Vec<Vec<usize>>,
    i_capacity: usize,
    // Deterministic RNG for sampling.
    rng: rand::rngs::StdRng,
}

impl replay_buffer_ascii {
    fn new(i_capacity: usize, u64_seed: u64) -> Self {
        let i_cap = if i_capacity > 0 { i_capacity.min(50000) } else { 0 };
        Self {
            v_rows: Vec::new(),
            i_capacity: i_cap,
            rng: rand::rngs::StdRng::seed_from_u64(u64_seed),
        }
    }

    fn is_enabled(&self) -> bool {
        self.i_capacity > 0
    }

    fn len(&self) -> usize {
        self.v_rows.len()
    }

    fn push_row(&mut self, v_row: &[usize]) {
        if !self.is_enabled() {
            return;
        }
        if v_row.len() < 2 {
            return;
        }
        // Copy and cap length defensively.
        let mut v = v_row.to_vec();
        if v.len() > crate::MAX_SEQ_LEN {
            v.truncate(crate::MAX_SEQ_LEN);
        }

        if self.v_rows.len() < self.i_capacity {
            self.v_rows.push(v);
            return;
        }

        // Ring behavior: overwrite a random slot to avoid bias toward old items.
        if !self.v_rows.is_empty() {
            let i_idx = self.rng.gen_range(0..self.v_rows.len());
            self.v_rows[i_idx] = v;
        }
    }

    fn sample_row(&mut self) -> Option<Vec<usize>> {
        if self.v_rows.is_empty() {
            return None;
        }
        let i_idx = self.rng.gen_range(0..self.v_rows.len());
        Some(self.v_rows[i_idx].clone())
    }
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

#[derive(Clone, Debug)]
pub enum training_phase_ascii {
    warmup,
    finetune,
    realtime,
}

#[derive(Clone, Debug)]
pub struct phase_strategy_config_ascii {
    pub e_phase: training_phase_ascii,

    // EMA routing control.
    pub b_enable_ema_branch_selection: bool,
    pub i_ema_warmup_steps: usize,

    // Replay control.
    pub b_enable_replay: bool,
    pub d_replay_p_start: f32,
    pub d_replay_p_max: f32,
    pub i_replay_ramp_steps: usize,

    // Expansion control.
    pub b_enable_autonomous_expansion: bool,
    pub i_expand_check_every_steps: usize,
    pub d_eta_injection: f32,

    // Safety limits.
    pub i_max_total_branches: usize,
}

impl phase_strategy_config_ascii {
    pub fn validate(&self) -> Result<(), String> {
        if self.i_expand_check_every_steps == 0 {
            return Err("phase_cfg_expand_check_every_steps_zero".to_string());
        }
        if self.i_max_total_branches == 0 {
            return Err("phase_cfg_max_total_branches_zero".to_string());
        }
        if !self.d_eta_injection.is_finite() || self.d_eta_injection < 0.0 || self.d_eta_injection > 0.5 {
            return Err("phase_cfg_eta_injection_invalid".to_string());
        }
        if !self.d_replay_p_start.is_finite() || !self.d_replay_p_max.is_finite() {
            return Err("phase_cfg_replay_p_non_finite".to_string());
        }
        if self.d_replay_p_start < 0.0 || self.d_replay_p_start > 1.0 {
            return Err("phase_cfg_replay_p_start_invalid".to_string());
        }
        if self.d_replay_p_max < 0.0 || self.d_replay_p_max > 1.0 {
            return Err("phase_cfg_replay_p_max_invalid".to_string());
        }
        if self.d_replay_p_start > self.d_replay_p_max {
            return Err("phase_cfg_replay_p_start_gt_max".to_string());
        }
        Ok(())
    }

    pub fn replay_p_at_step(&self, i_total_steps: usize) -> f32 {
        if !self.b_enable_replay {
            return 0.0;
        }
        if self.i_replay_ramp_steps == 0 {
            return self.d_replay_p_max;
        }
        let d_t = (i_total_steps as f32) / (self.i_replay_ramp_steps as f32).max(1.0);
        let d_a = d_t.clamp(0.0, 1.0);
        let d_p = self.d_replay_p_start + d_a * (self.d_replay_p_max - self.d_replay_p_start);
        if d_p.is_finite() { d_p.clamp(0.0, 1.0) } else { 0.0 }
    }

    pub fn ema_is_active(&self, i_total_steps: usize) -> bool {
        self.b_enable_ema_branch_selection && i_total_steps >= self.i_ema_warmup_steps
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
            if let Some(pg) = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
            {
                pg.set_training(b_training);
                continue;
            }
            if let Some(ts) = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<TransformerSequence>())
            {
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
            if let Some(pg) = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
            {
                pg.set_residual_dropout_p(self.d_residual_dropout_p);
                continue;
            }
            if let Some(ts) = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<TransformerSequence>())
            {
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
                    i_embedding_dim: EMBEDDING_DIM,
                    i_hidden_dim: HIDDEN_DIM,
                    i_num_heads: 4,
                });
                continue;
            }

            if s_t == "TransformerSequence" {
                let ts = layer
                    .as_any_mut()
                    .and_then(|a| a.downcast_mut::<TransformerSequence>())
                    .ok_or_else(|| "topology_export_downcast_transformer_sequence_failed".to_string())?;
                v_layers.push(ts.export_spec());
                continue;
            }

            if s_t == "ParallelBlockGroup" {
                let pg = layer
                    .as_any_mut()
                    .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                    .ok_or_else(|| "topology_export_downcast_parallel_block_group_failed".to_string())?;

                let mut v_branches: Vec<llm_layer_spec> = Vec::new();
                for br in pg.v_branches.iter_mut() {
                    let s_bt = br.layer_type();

                    if s_bt == "TransformerBlock" {
                        v_branches.push(llm_layer_spec::transformer_block {
                            i_embedding_dim: EMBEDDING_DIM,
                            i_hidden_dim: HIDDEN_DIM,
                            i_num_heads: 4,
                        });
                    } else if s_bt == "TransformerSequence" {
                        let tsb = br
                            .as_any_mut()
                            .and_then(|a| a.downcast_mut::<TransformerSequence>())
                            .ok_or_else(|| {
                                "topology_export_downcast_branch_transformer_sequence_failed".to_string()
                            })?;
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
                    i_embedding_dim: EMBEDDING_DIM,
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
        let topology = self.export_topology_spec()?;

        // Step 2: Materialize tokenizer checkpoint without keeping a borrow alive.
        let tokenizer_cp = {
            let tok = self
                .bpe_tokenizer
                .as_ref()
                .ok_or_else(|| "tokenizer_not_set".to_string())?;
            tok.to_checkpoint()
        };

        // Step 3: Collect parameters.
        let v_params = self.collect_all_parameters_flat();

        let cp = llm_checkpoint_v2::new(
            tokenizer_cp,
            topology,
            v_params,
            MAX_SEQ_LEN,
            EMBEDDING_DIM,
            HIDDEN_DIM,
        );

        let s_json = utils::checkpoint_to_json_ascii(&cp)?;
        utils::write_file_atomic_ascii(s_path, &s_json)?;
        Ok(())
    }

    pub fn load_checkpoint_llm_checkpoint_v2_rebuild(s_path: &str) -> Result<Llm, String> {
        if s_path.trim().is_empty() {
            return Err("checkpoint_path_empty".to_string());
        }

        let s_json =
            std::fs::read_to_string(s_path).map_err(|_| "checkpoint_read_error".to_string())?;
        let cp: llm_checkpoint_v2 = utils::checkpoint_from_json_ascii(&s_json)?;
        cp.validate()?;

        let bpe = BpeTokenizer::from_checkpoint(&cp.tokenizer)?;
        let vocab = bpe.vocab.clone();

        let network = build_network_from_topology(&cp.topology, &vocab)?;

        let mut llm = Llm::new(vocab, network);
        llm.set_bpe_tokenizer(bpe);

        llm.set_residual_dropout_p(0.1);
        llm.set_training(true);
        let _ = llm.set_sampling_config(0.9, 40, 0.95, 987654321);

        let i_expected: usize = llm
            .network
            .iter()
            .map(|l| l.get_parameters_flat().len())
            .sum();
        if i_expected != cp.v_params.len() {
            return Err(format!(
                "checkpoint_param_count_mismatch expected={} got={}",
                i_expected,
                cp.v_params.len()
            ));
        }

        llm.assign_all_parameters_flat(&cp.v_params)?;
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

        let dist =
            WeightedIndex::new(&v_weights).map_err(|_| "sampling_weighted_index_error".to_string())?;
        let i_pick = dist.sample(&mut self.rng_sampling);

        v_ids
            .get(i_pick)
            .copied()
            .ok_or_else(|| "sampling_pick_oob".to_string())
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
            let opt_pg = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
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
            let opt_pg = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
            if let Some(pg) = opt_pg {
                pg.set_fault_drop_branch_idx(None);
                pg.set_fault_injection_enabled(false);
            }
        }
    }

    pub fn set_outage_simulation_enabled(&mut self, b_enabled: bool) {
        self.b_outage_simulation_enabled = b_enabled;
        if !b_enabled {
            self.clear_predict_outage_for_all_parallel_groups_test_only();
        }
    }

    pub fn is_outage_simulation_enabled(&self) -> bool {
        self.b_outage_simulation_enabled
    }

    fn find_first_parallel_block_group_index_ascii(&self) -> Option<usize> {
        for (i_idx, layer) in self.network.iter().enumerate() {
            if layer.layer_type() == "ParallelBlockGroup" {
                return Some(i_idx);
            }
        }
        None
    }

    pub fn train_two_phase_with_progress_online_ascii(
        &mut self,
        v_data_phase1: Vec<&str>,
        i_epochs_phase1: usize,
        d_lr_phase1: f32,
        s_phase1: &str,
        opt_cl_phase1: Option<ContinuousLearningConfig>,
        cfg_phase1: phase_strategy_config_ascii,
        v_data_phase2: Vec<&str>,
        i_epochs_phase2: usize,
        d_lr_phase2: f32,
        s_phase2: &str,
        opt_cl_phase2: Option<ContinuousLearningConfig>,
        cfg_phase2: phase_strategy_config_ascii,
        b_cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
        tx_progress: Sender<TrainingProgressEventAscii>,
        i_snapshot_every_steps: usize,
        tx_snapshot: Option<Sender<Vec<f32>>>,
        rx_data: Receiver<TrainingDataEventAscii>,
    ) -> Result<(), String> {
        // History:
        // - 2026-02-14: Ensure receiver persists across both phases to prevent send failures.

        // Keep receiver alive for entire session; pass as Option to allow shutdown.
        let mut opt_rx: Option<Receiver<TrainingDataEventAscii>> = Some(rx_data);

        // Phase 1.
        self.train_with_progress_continuous_learning_online_ascii(
            v_data_phase1,
            i_epochs_phase1,
            d_lr_phase1,
            std::sync::Arc::clone(&b_cancel),
            tx_progress.clone(),
            s_phase1,
            opt_cl_phase1,
            i_snapshot_every_steps,
            tx_snapshot.as_ref().map(|t| t.clone()),
            cfg_phase1,
            &mut opt_rx,
        )?;

        if b_cancel.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        // Phase 2.
        self.train_with_progress_continuous_learning_online_ascii(
            v_data_phase2,
            i_epochs_phase2,
            d_lr_phase2,
            std::sync::Arc::clone(&b_cancel),
            tx_progress,
            s_phase2,
            opt_cl_phase2,
            i_snapshot_every_steps,
            tx_snapshot,
            cfg_phase2,
            &mut opt_rx,
        )?;

        Ok(())
    }


    pub fn train_with_progress_continuous_learning_online_ascii(
        &mut self,
        v_data: Vec<&str>,
        i_epochs: usize,
        d_lr: f32,
        b_cancel: std::sync::Arc<AtomicBool>,
        tx_progress: Sender<TrainingProgressEventAscii>,
        s_phase: &str,
        opt_cl: Option<ContinuousLearningConfig>,
        i_snapshot_every_steps: usize,
        tx_snapshot: Option<Sender<Vec<f32>>>,
        cfg_phase: phase_strategy_config_ascii,
        opt_data_rx: &mut Option<Receiver<TrainingDataEventAscii>>,
    ) -> Result<(), String> {
        // History:
        // - 2026-02-15: Add advanced validation metrics emission for continuous learning and expansion.

        cfg_phase.validate()?;
        self.set_training(true);

        if v_data.is_empty() || i_epochs == 0 {
            return Err("invalid_training_args".to_string());
        }
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return Err("invalid_learning_rate".to_string());
        }
        if s_phase.trim().is_empty() {
            return Err("invalid_phase".to_string());
        }
        if self.bpe_tokenizer.is_none() {
            return Err("tokenizer_not_set".to_string());
        }

        // Initial token cache (append-only afterwards).
        let mut v_tokenized_data: Vec<Vec<usize>> = v_data
            .iter()
            .map(|s| self.tokenize(s))
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .filter(|v| v.len() >= 2)
            .collect();

        if v_tokenized_data.is_empty() {
            return Err("no_tokenized_rows".to_string());
        }

        // Fixed control sets for retention (6).
        // NOTE: Use deterministic slices from initial dataset as old/new proxies.
        let v_control_old: Vec<Vec<usize>> = v_tokenized_data.iter().take(64.min(v_tokenized_data.len())).cloned().collect();
        let v_control_new: Vec<Vec<usize>> = v_tokenized_data
            .iter()
            .rev()
            .take(64.min(v_tokenized_data.len()))
            .cloned()
            .collect();

        // Find first ParallelBlockGroup.
        let mut opt_pg_idx: Option<usize> = None;
        for (i_idx, layer) in self.network.iter().enumerate() {
            if layer.layer_type() == "ParallelBlockGroup" {
                opt_pg_idx = Some(i_idx);
                break;
            }
        }

        let mut rng_mask = rand::rngs::StdRng::seed_from_u64(
            opt_cl
                .as_ref()
                .map(|c| c.u64_mask_seed)
                .unwrap_or(20260213),
        );

        // Metrics state.
        let mut met_ing = ingestion_metrics_ascii::new();
        let mut met_cov = coverage_metrics_ascii::new();
        let mut met_mask = mask_participation_metrics_ascii::new();
        let mut met_inv = inverse_participation_scaling_metrics_ascii::new();
        let mut met_replay = replay_metrics_ascii::new();
        let mut met_ret = retention_metrics_ascii::new();
        let mut met_fair = branch_fairness_metrics_ascii::new(1);
        let mut met_snap = snapshot_metrics_ascii::new();
        let mut met_expand = expansion_metrics_ascii::new();
        let mut met_drift = drift_metrics_ascii::new();

        // Ingestion state.
        let mut st_ing = online_data_ingestion_state_ascii::new_default();

        // Baseline retention evaluation at start of phase.
        let d_old0 = self.eval_control_set_loss_ascii(&v_control_old, 64).unwrap_or(0.0);
        let d_new0 = self.eval_control_set_loss_ascii(&v_control_new, 64).unwrap_or(0.0);
        met_ret.d_loss_control_old = d_old0;
        met_ret.d_loss_control_new = d_new0;
        met_ret.d_retention_delta_old = 0.0;
        met_ret.d_retention_delta_new = 0.0;
        met_ret.u64_last_eval_ms = now_ms_ascii();

        // EMA and replay internal state (existing).
        let mut st_branch_ema = branch_loss_ema_state_ascii::new(1, 0.2);
        let mut rb_replay = replay_buffer_ascii::new(if cfg_phase.b_enable_replay { 5000 } else { 0 }, 20260213);
        let i_replay_max_steps_per_row: usize = 1;

        let mut i_total_steps: usize = 0;
        let mut i_skips_empty_act: usize = 0;
        let mut i_skips_empty_logits: usize = 0;
        let mut i_skips_pg_downcast_failed: usize = 0;
        let mut i_skips_pg_no_branches: usize = 0;

        // Drift prompts (10).
        let v_drift_prompts: Vec<String> = vec![
            "User: diagnostic prompt 1".to_string(),
            "User: diagnostic prompt 2".to_string(),
            "User: diagnostic prompt 3".to_string(),
            "User: diagnostic prompt 4".to_string(),
        ];

        for i_epoch in 0..i_epochs {
            if b_cancel.load(AtomicOrdering::SeqCst) {
                return Ok(());
            }

            // (1) Drain ingestion and update ingestion metrics.
            drain_training_data_events_non_blocking_ascii(
                self,
                opt_data_rx,
                &mut v_tokenized_data,
                &mut st_ing,
                &mut met_ing,
            );

            // (2) Coverage baseline for epoch.
            met_cov.i_epoch_token_rows_start = v_tokenized_data.len();
            let i_epoch_len = v_tokenized_data.len();

            let mut d_total_loss_used: f32 = 0.0;
            let mut i_used_rows: usize = 0;
            let mut d_last_step_loss: f32 = 0.0;

            for i_row_idx in 0..i_epoch_len {
                if b_cancel.load(AtomicOrdering::SeqCst) {
                    return Ok(());
                }

                // Periodic drain for responsiveness.
                if (i_total_steps % 50) == 0 {
                    drain_training_data_events_non_blocking_ascii(
                        self,
                        opt_data_rx,
                        &mut v_tokenized_data,
                        &mut st_ing,
                        &mut met_ing,
                    );
                }

                let v_row = &v_tokenized_data[i_row_idx];

                let b_ema_active = cfg_phase.ema_is_active(i_total_steps);
                let d_replay_p = cfg_phase.replay_p_at_step(i_total_steps);
                met_replay.d_replay_p_last = d_replay_p;

                // (5) Fresh step counter.
                met_replay.inc_fresh();

                // Train fresh row (includes mask and optional EMA selection).
                let (b_ok, d_loss, opt_step_diag) = self.train_one_row_continuous_learning_with_metrics_ascii(
                    v_row,
                    d_lr,
                    opt_pg_idx,
                    &opt_cl,
                    &mut rng_mask,
                    &mut st_branch_ema,
                    &mut met_mask,
                    &mut met_inv,
                    &mut met_fair,
                    &mut i_skips_empty_act,
                    &mut i_skips_empty_logits,
                    &mut i_skips_pg_downcast_failed,
                    &mut i_skips_pg_no_branches,
                    b_ema_active,
                )?;

                if !b_ok {
                    continue;
                }

                d_last_step_loss = d_loss;
                d_total_loss_used += d_loss;
                i_used_rows = i_used_rows.saturating_add(1);
                i_total_steps = i_total_steps.saturating_add(1);

                // Replay buffer update and optional replay steps.
                rb_replay.push_row(v_row);
                if rb_replay.is_enabled() && rb_replay.len() > 0 && d_replay_p > 0.0 {
                    let d_u: f32 = rng_mask.gen_range(0.0..1.0);
                    if d_u < d_replay_p {
                        for _ in 0..i_replay_max_steps_per_row {
                            if let Some(v_rep) = rb_replay.sample_row() {
                                met_replay.inc_replay();
                                let (b_ok_r, d_loss_r, _) = self.train_one_row_continuous_learning_with_metrics_ascii(
                                    &v_rep,
                                    d_lr,
                                    opt_pg_idx,
                                    &opt_cl,
                                    &mut rng_mask,
                                    &mut st_branch_ema,
                                    &mut met_mask,
                                    &mut met_inv,
                                    &mut met_fair,
                                    &mut i_skips_empty_act,
                                    &mut i_skips_empty_logits,
                                    &mut i_skips_pg_downcast_failed,
                                    &mut i_skips_pg_no_branches,
                                    b_ema_active,
                                )?;
                                if b_ok_r {
                                    met_replay.update_delta_loss(d_last_step_loss, d_loss_r);
                                }
                            }
                        }
                    }
                }

                // (9) Autonomous expansion telemetry and (10) drift proxy around expansion.
                if cfg_phase.b_enable_autonomous_expansion
                    && cfg_phase.i_expand_check_every_steps > 0
                    && (i_total_steps % cfg_phase.i_expand_check_every_steps) == 0
                {
                    // Compute drift baseline logits.
                    let v_logits_before = self.compute_logits_for_prompts_ascii(&v_drift_prompts, 4).unwrap_or_else(|_| Vec::new());

                    let v_diag_inputs = self
                        .collect_parallel_block_group_inputs_for_diagnostics(&v_drift_prompts, 4)
                        .unwrap_or_else(|_| Vec::new());

                    let mut b_expanded: bool = false;
                    let mut i_before: usize = 0;
                    let mut i_after: usize = 0;

                    if !v_diag_inputs.is_empty() {
                        // Capture before branches.
                        if let Some(i_pg) = opt_pg_idx {
                            if let Some(pg) = self.network[i_pg].as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>()) {
                                i_before = pg.num_branches();
                            }
                        }

                        b_expanded = self.try_autonomous_expand_first_pg_ascii(&cfg_phase, &v_diag_inputs).unwrap_or(false);

                        if b_expanded {
                            if let Some(i_pg) = opt_pg_idx {
                                if let Some(pg) = self.network[i_pg].as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>()) {
                                    i_after = pg.num_branches();
                                    // Best effort: compute sum of weights added (approx via eta and new count).
                                    let d_eta = cfg_phase.d_eta_injection.clamp(0.0, 0.5);
                                    let i_add = i_after.saturating_sub(i_before);
                                    let d_sum_w_new = if i_add == 0 { 0.0 } else { d_eta };
                                    met_expand.i_expansion_events_total = met_expand.i_expansion_events_total.saturating_add(1);
                                    met_expand.i_branches_before_last = i_before;
                                    met_expand.i_branches_after_last = i_after;
                                    met_expand.d_eta_injection_last = d_eta;
                                    met_expand.d_sum_w_new_last = d_sum_w_new;
                                    met_expand.u64_last_event_ms = now_ms_ascii();
                                }
                            }

                            // Compute drift after.
                            let v_logits_after = self.compute_logits_for_prompts_ascii(&v_drift_prompts, 4).unwrap_or_else(|_| Vec::new());
                            let i_cmp = v_logits_before.len().min(v_logits_after.len());
                            for i in 0..i_cmp {
                                let d_l2 = logits_distance_l2_ascii(&v_logits_before[i], &v_logits_after[i]);
                                let d_cd = logits_cosine_distance_ascii(&v_logits_before[i], &v_logits_after[i]);
                                met_drift.update(d_l2, d_cd);
                            }
                        }
                    }
                }

                // (8) Snapshot send with timestamp and step tags.
                if i_snapshot_every_steps > 0 && (i_total_steps % i_snapshot_every_steps) == 0 {
                    if let Some(tx) = tx_snapshot.as_ref() {
                        let v_params = self.export_parameters_snapshot();
                        let u64_send = now_ms_ascii();
                        met_snap.on_send(i_total_steps, u64_send);

                        // NOTE: Snapshot payload remains Vec<f32> to keep compatibility with main.rs.
                        // Latency and staleness must be computed on the receiving side (main.rs),
                        // but sent counters and last-send timestamp remain useful on the training side.
                        let _ = tx.send(v_params);
                    }
                }

                // (6) Retention evaluation periodically.
                if (i_total_steps % 500) == 0 {
                    let d_old = self.eval_control_set_loss_ascii(&v_control_old, 64).unwrap_or(0.0);
                    let d_new = self.eval_control_set_loss_ascii(&v_control_new, 64).unwrap_or(0.0);
                    met_ret.d_retention_delta_old = d_old - d_old0;
                    met_ret.d_retention_delta_new = d_new - d_new0;
                    met_ret.d_loss_control_old = d_old;
                    met_ret.d_loss_control_new = d_new;
                    met_ret.u64_last_eval_ms = now_ms_ascii();
                }

                // (7) Recompute fairness summaries occasionally.
                if (i_total_steps % 100) == 0 {
                    met_fair.recompute();
                }

                // Progress event.
                if (i_total_steps % 25) == 0 {
                    let d_running_epoch_avg_loss: f32 = if i_used_rows == 0 {
                        0.0
                    } else {
                        d_total_loss_used / (i_used_rows as f32).max(1.0)
                    };

                    let mut ev = TrainingProgressEventAscii::new_basic(
                        s_phase,
                        i_epoch + 1,
                        i_epochs,
                        d_running_epoch_avg_loss,
                        d_last_step_loss,
                        0,
                        i_total_steps,
                    );

                    // Existing diagnostics.
                    ev.i_skips_empty_act = i_skips_empty_act;
                    ev.i_skips_empty_logits = i_skips_empty_logits;
                    ev.i_skips_pg_downcast_failed = i_skips_pg_downcast_failed;
                    ev.i_skips_pg_no_branches = i_skips_pg_no_branches;

                    // (1) Ingestion metrics.
                    ev.d_ingest_rows_per_sec_window = met_ing.d_rows_per_sec_window;
                    ev.d_ingest_events_per_sec_window = met_ing.d_events_per_sec_window;
                    ev.i_ingest_rows_added_total = met_ing.i_rows_added_total;
                    ev.i_ingest_events_processed_total = met_ing.i_events_processed_total;
                    ev.i_ingest_parse_errors_total = met_ing.i_parse_errors_total;
                    ev.i_ingest_rows_rejected_total = met_ing.i_rows_rejected_total;
                    ev.i_ingest_pending_events_observed_peak = met_ing.i_pending_events_observed_peak;

                    // (2) Coverage: computed end of epoch; include interim approximations.
                    ev.i_epoch_token_rows_start = met_cov.i_epoch_token_rows_start;
                    ev.i_epoch_token_rows_end = v_tokenized_data.len();
                    ev.i_new_rows_added_during_epoch =
                        v_tokenized_data.len().saturating_sub(met_cov.i_epoch_token_rows_start);
                    ev.d_coverage_ratio_used_over_available = if i_epoch_len == 0 {
                        0.0
                    } else {
                        (i_used_rows as f32) / (i_epoch_len as f32).max(1.0)
                    };
                    ev.d_new_data_ratio_in_available = if v_tokenized_data.len() == 0 {
                        0.0
                    } else {
                        (ev.i_new_rows_added_during_epoch as f32) / (v_tokenized_data.len() as f32).max(1.0)
                    };

                    // (3) Mask stats.
                    ev.d_active_branches_mean = met_mask.d_active_branches_mean;
                    ev.d_active_branches_std = met_mask.active_branches_std();
                    ev.i_active_branches_min = if met_mask.i_active_branches_min == usize::MAX { 0 } else { met_mask.i_active_branches_min };
                    ev.i_active_branches_max = met_mask.i_active_branches_max;
                    ev.d_mask_sparsity_mean = met_mask.d_mask_sparsity_mean;
                    ev.d_mask_sparsity_std = met_mask.sparsity_std();
                    ev.d_steps_at_min_active_share = if met_mask.i_steps_observed == 0 {
                        0.0
                    } else {
                        (met_mask.i_steps_at_min_active as f32) / (met_mask.i_steps_observed as f32).max(1.0)
                    };

                    // (4) Scaling impact.
                    ev.d_grad_norm_ratio_scaled_over_unscaled_mean = met_inv.d_grad_norm_ratio_mean;
                    ev.d_grad_norm_ratio_scaled_over_unscaled_std = met_inv.ratio_std();
                    ev.d_grad_norm_scaled_mean = met_inv.d_grad_norm_scaled_mean;
                    ev.d_grad_norm_unscaled_mean = met_inv.d_grad_norm_unscaled_mean;

                    // (5) Replay.
                    let i_total = met_replay.i_fresh_steps_total.saturating_add(met_replay.i_replay_steps_total);
                    ev.d_replay_share = if i_total == 0 {
                        0.0
                    } else {
                        (met_replay.i_replay_steps_total as f32) / (i_total as f32).max(1.0)
                    };
                    ev.d_replay_p_last = met_replay.d_replay_p_last;
                    ev.d_replay_delta_loss_mean = met_replay.d_delta_loss_mean;
                    ev.d_replay_delta_loss_std = met_replay.delta_loss_std();

                    // (6) Retention.
                    ev.d_loss_control_old = met_ret.d_loss_control_old;
                    ev.d_loss_control_new = met_ret.d_loss_control_new;
                    ev.d_retention_delta_old = met_ret.d_retention_delta_old;
                    ev.d_retention_delta_new = met_ret.d_retention_delta_new;

                    // (7) Fairness.
                    ev.d_branch_select_gini = met_fair.d_gini_selected;
                    ev.d_branch_select_top1_share = met_fair.d_top1_share_selected;

                    // (8) Snapshot send counters.
                    ev.i_snapshots_sent_total = met_snap.i_snapshots_sent_total;

                    // (9) Expansion.
                    ev.i_expansion_events_total = met_expand.i_expansion_events_total;
                    ev.i_branches_before_last_expand = met_expand.i_branches_before_last;
                    ev.i_branches_after_last_expand = met_expand.i_branches_after_last;
                    ev.d_eta_injection_last = met_expand.d_eta_injection_last;
                    ev.d_sum_w_new_last = met_expand.d_sum_w_new_last;

                    // (10) Drift.
                    ev.d_expand_drift_logits_l2_mean = met_drift.d_logits_l2_mean;
                    ev.d_expand_drift_logits_l2_std = met_drift.l2_std();
                    ev.d_expand_drift_logits_cos_dist_mean = met_drift.d_logits_cos_dist_mean;
                    ev.d_expand_drift_logits_cos_dist_std = met_drift.cos_dist_std();

                    // EMA.
                    ev.b_ema_active = b_ema_active;
                    ev.i_ema_last_selected_branch = st_branch_ema
                        .opt_last_selected_branch
                        .map(|i| i as isize)
                        .unwrap_or(-1);

                    let _ = tx_progress.send(ev);
                }
            }

            // End-of-epoch coverage finalize.
            met_cov.i_epoch_token_rows_end = v_tokenized_data.len();
            met_cov.i_epoch_rows_used = i_used_rows;
            met_cov.i_new_rows_added_during_epoch = met_cov
                .i_epoch_token_rows_end
                .saturating_sub(met_cov.i_epoch_token_rows_start);

            met_cov.d_coverage_ratio_used_over_available = if met_cov.i_epoch_token_rows_start == 0 {
                0.0
            } else {
                (i_used_rows as f32) / (met_cov.i_epoch_token_rows_start as f32).max(1.0)
            };

            met_cov.d_new_data_ratio_in_available = if met_cov.i_epoch_token_rows_end == 0 {
                0.0
            } else {
                (met_cov.i_new_rows_added_during_epoch as f32) / (met_cov.i_epoch_token_rows_end as f32).max(1.0)
            };

            // End-of-epoch progress.
            let d_avg_loss: f32 = if i_used_rows == 0 {
                0.0
            } else {
                d_total_loss_used / (i_used_rows as f32).max(1.0)
            };

            let mut ev = TrainingProgressEventAscii::new_basic(
                s_phase,
                i_epoch + 1,
                i_epochs,
                d_avg_loss,
                d_last_step_loss,
                i_used_rows,
                i_total_steps,
            );

            // Attach key metrics at epoch end as well.
            ev.i_rows_used_last_epoch = i_used_rows;
            ev.i_epoch_token_rows_start = met_cov.i_epoch_token_rows_start;
            ev.i_epoch_token_rows_end = met_cov.i_epoch_token_rows_end;
            ev.i_new_rows_added_during_epoch = met_cov.i_new_rows_added_during_epoch;
            ev.d_coverage_ratio_used_over_available = met_cov.d_coverage_ratio_used_over_available;
            ev.d_new_data_ratio_in_available = met_cov.d_new_data_ratio_in_available;

            ev.d_ingest_rows_per_sec_window = met_ing.d_rows_per_sec_window;
            ev.d_ingest_events_per_sec_window = met_ing.d_events_per_sec_window;
            ev.i_ingest_rows_added_total = met_ing.i_rows_added_total;
            ev.i_ingest_events_processed_total = met_ing.i_events_processed_total;
            ev.i_ingest_parse_errors_total = met_ing.i_parse_errors_total;
            ev.i_ingest_rows_rejected_total = met_ing.i_rows_rejected_total;
            ev.i_ingest_pending_events_observed_peak = met_ing.i_pending_events_observed_peak;

            ev.d_active_branches_mean = met_mask.d_active_branches_mean;
            ev.d_active_branches_std = met_mask.active_branches_std();
            ev.i_active_branches_min = if met_mask.i_active_branches_min == usize::MAX { 0 } else { met_mask.i_active_branches_min };
            ev.i_active_branches_max = met_mask.i_active_branches_max;
            ev.d_mask_sparsity_mean = met_mask.d_mask_sparsity_mean;
            ev.d_mask_sparsity_std = met_mask.sparsity_std();

            ev.d_grad_norm_ratio_scaled_over_unscaled_mean = met_inv.d_grad_norm_ratio_mean;
            ev.d_grad_norm_ratio_scaled_over_unscaled_std = met_inv.ratio_std();

            let i_total = met_replay.i_fresh_steps_total.saturating_add(met_replay.i_replay_steps_total);
            ev.d_replay_share = if i_total == 0 { 0.0 } else { (met_replay.i_replay_steps_total as f32) / (i_total as f32).max(1.0) };
            ev.d_replay_delta_loss_mean = met_replay.d_delta_loss_mean;
            ev.d_replay_delta_loss_std = met_replay.delta_loss_std();

            ev.d_loss_control_old = met_ret.d_loss_control_old;
            ev.d_loss_control_new = met_ret.d_loss_control_new;
            ev.d_retention_delta_old = met_ret.d_retention_delta_old;
            ev.d_retention_delta_new = met_ret.d_retention_delta_new;

            met_fair.recompute();
            ev.d_branch_select_gini = met_fair.d_gini_selected;
            ev.d_branch_select_top1_share = met_fair.d_top1_share_selected;

            ev.i_snapshots_sent_total = met_snap.i_snapshots_sent_total;

            ev.i_expansion_events_total = met_expand.i_expansion_events_total;
            ev.i_branches_before_last_expand = met_expand.i_branches_before_last;
            ev.i_branches_after_last_expand = met_expand.i_branches_after_last;
            ev.d_eta_injection_last = met_expand.d_eta_injection_last;
            ev.d_sum_w_new_last = met_expand.d_sum_w_new_last;

            ev.d_expand_drift_logits_l2_mean = met_drift.d_logits_l2_mean;
            ev.d_expand_drift_logits_l2_std = met_drift.l2_std();
            ev.d_expand_drift_logits_cos_dist_mean = met_drift.d_logits_cos_dist_mean;
            ev.d_expand_drift_logits_cos_dist_std = met_drift.cos_dist_std();

            ev.b_ema_active = cfg_phase.ema_is_active(i_total_steps);
            ev.i_ema_last_selected_branch = st_branch_ema
                .opt_last_selected_branch
                .map(|i| i as isize)
                .unwrap_or(-1);

            ev.i_skips_empty_act = i_skips_empty_act;
            ev.i_skips_empty_logits = i_skips_empty_logits;
            ev.i_skips_pg_downcast_failed = i_skips_pg_downcast_failed;
            ev.i_skips_pg_no_branches = i_skips_pg_no_branches;

            let _ = tx_progress.send(ev);
        }

        Ok(())
    }

    // Wrapper around existing train_one_row_continuous_learning_ascii to extract
    // mask stats, inverse scaling impact proxy, and fairness selection stats.
    fn train_one_row_continuous_learning_with_metrics_ascii(
        &mut self,
        v_row: &[usize],
        d_lr: f32,
        opt_pg_idx: Option<usize>,
        opt_cl: &Option<ContinuousLearningConfig>,
        rng_mask: &mut StdRng,
        st_branch_ema: &mut branch_loss_ema_state_ascii,
        met_mask: &mut mask_participation_metrics_ascii,
        met_inv: &mut inverse_participation_scaling_metrics_ascii,
        met_fair: &mut branch_fairness_metrics_ascii,
        i_skips_empty_act: &mut usize,
        i_skips_empty_logits: &mut usize,
        i_skips_pg_downcast_failed: &mut usize,
        i_skips_pg_no_branches: &mut usize,
        b_enable_ema_selection: bool,
    ) -> Result<(bool, f32, Option<(usize, usize, bool)>), String> {
        // History:
        // - 2026-02-15: Add metric taps for mask participation, scaling proxy, and fairness.

        if v_row.len() < 2 {
            return Ok((false, 0.0, None));
        }

        // If there is a PG, sample mask similarly to original function, but collect stats.
        if let Some(i_pg) = opt_pg_idx {
            let i_k = match self.network[i_pg]
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
            {
                Some(pg) => pg.num_branches(),
                None => {
                    *i_skips_pg_downcast_failed = i_skips_pg_downcast_failed.saturating_add(1);
                    return Ok((false, 0.0, None));
                }
            };

            if i_k == 0 {
                *i_skips_pg_no_branches = i_skips_pg_no_branches.saturating_add(1);
                return Ok((false, 0.0, None));
            }

            let cl_cfg = opt_cl.clone().unwrap_or_else(|| ContinuousLearningConfig::new_default_for_num_branches(i_k));
            cl_cfg.validate(i_k)?;

            let v_available_mask = Self::sample_availability_mask(
                rng_mask,
                &cl_cfg.v_branch_participation_p,
                cl_cfg.i_min_active_branches,
            );

            let i_active = v_available_mask.iter().filter(|&&b| b).count();
            met_mask.update(i_active, i_k, cl_cfg.i_min_active_branches);

            // EMA selection in original path: estimate selected branch for fairness.
            let mut i_selected_branch: Option<usize> = None;
            if b_enable_ema_selection {
                    // Compute activation before calling select_branch... to avoid double mutable borrow.
                    let a_act_before_pg = self.forward_prefix_to_pg_ascii(v_row, i_pg)?;

                    if let Ok(i_sel) = self.select_branch_min_ema_loss_ascii(
                        st_branch_ema,
                        i_pg,
                        &a_act_before_pg,
                        &v_row[1..],
                        &v_available_mask,
                    ) {
                        i_selected_branch = Some(i_sel);
                    }
                }

            met_fair.ensure_len(i_k);
            if let Some(i_sel) = i_selected_branch {
                met_fair.on_select(i_sel);
            }

            // (4) Inverse participation scaling impact proxy:
            // compute grad norms for scaled and unscaled for identical row and mask, without updating weights
            // by using "dry run" gradients; then perform actual training using existing function.
            if cl_cfg.b_scale_by_inverse_participation && (met_inv.i_samples < 5000) && (i_active > 0) {
                let (d_g_scaled, d_g_unscaled) = self.compute_grad_norm_scaled_unscaled_proxy_ascii(
                    v_row,
                    d_lr,
                    i_pg,
                    &cl_cfg,
                    &v_available_mask,
                    i_selected_branch,
                )?;
                met_inv.update(d_g_scaled, d_g_unscaled);
            }

            let (b_ok, d_loss) = self.train_one_row_continuous_learning_ascii(
                v_row,
                d_lr,
                opt_pg_idx,
                opt_cl,
                rng_mask,
                st_branch_ema,
                i_skips_empty_act,
                i_skips_empty_logits,
                i_skips_pg_downcast_failed,
                i_skips_pg_no_branches,
                b_enable_ema_selection,
            )?;

            Ok((b_ok, d_loss, Some((i_active, i_k, b_enable_ema_selection))))
        } else {
            let (b_ok, d_loss) = self.train_one_row_continuous_learning_ascii(
                v_row,
                d_lr,
                opt_pg_idx,
                opt_cl,
                rng_mask,
                st_branch_ema,
                i_skips_empty_act,
                i_skips_empty_logits,
                i_skips_pg_downcast_failed,
                i_skips_pg_no_branches,
                b_enable_ema_selection,
            )?;
            Ok((b_ok, d_loss, None))
        }
    }
    // Forward prefix helper to compute activations before PG for EMA selection proxy.
    fn forward_prefix_to_pg_ascii(&mut self, v_row: &[usize], i_pg: usize) -> Result<Array2<f32>, String> {
        if v_row.len() < 2 {
            return Err("prefix_row_too_short".to_string());
        }
        let v_input_ids = &v_row[..v_row.len() - 1];
        let a_token_input: Array2<f32> = Array2::from_shape_vec(
            (1, v_input_ids.len()),
            v_input_ids.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
        )
        .map_err(|_| "prefix_shape_error_token_input".to_string())?;

        let mut a_act = a_token_input;
        for i_l in 0..i_pg {
            a_act = self.network[i_l].forward(&a_act);
            if a_act.nrows() == 0 || a_act.ncols() == 0 {
                return Err("prefix_empty_act".to_string());
            }
        }
        Ok(a_act)
    }

    // Proxy grad norm computation: scaled vs unscaled, for same mask and same row.
    // This computes gradients at logits only and does not update any weights.
    fn compute_grad_norm_scaled_unscaled_proxy_ascii(
        &mut self,
        v_row: &[usize],
        _d_lr: f32,
        i_pg: usize,
        cl_cfg: &ContinuousLearningConfig,
        v_available_mask: &[bool],
        opt_selected_branch: Option<usize>,
    ) -> Result<(f32, f32), String> {
        if v_row.len() < 2 {
            return Ok((0.0, 0.0));
        }
        if i_pg >= self.network.len() {
            return Err("inv_proxy_pg_oob".to_string());
        }

        let v_input_ids = &v_row[..v_row.len() - 1];
        let v_target_ids = &v_row[1..];

        let a_token_input: Array2<f32> = Array2::from_shape_vec(
            (1, v_input_ids.len()),
            v_input_ids.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
        )
        .map_err(|_| "inv_proxy_shape_error_token_input".to_string())?;

        // Forward prefix to PG.
        let mut a_act = a_token_input;
        for i_l in 0..i_pg {
            a_act = self.network[i_l].forward(&a_act);
            if a_act.nrows() == 0 || a_act.ncols() == 0 {
                return Ok((0.0, 0.0));
            }
        }

        // Decide training mask (single selected or available).
        let i_k = v_available_mask.len();
        let v_train_mask: Vec<bool> = if let Some(i_sel) = opt_selected_branch {
            Self::make_single_branch_mask_ascii(i_k, i_sel)
        } else {
            v_available_mask.to_vec()
        };

        // Forward PG.
        let a_after_pg = {
            let pg = self.network[i_pg]
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                .ok_or_else(|| "inv_proxy_pg_downcast_failed".to_string())?;
            pg.forward_with_availability_mask(&a_act, &v_train_mask)
        };
        if a_after_pg.nrows() == 0 || a_after_pg.ncols() == 0 {
            return Ok((0.0, 0.0));
        }

        // Forward tail to logits.
        let a_logits = self.forward_from_layer_index_ascii(i_pg + 1, &a_after_pg)?;
        if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
            return Ok((0.0, 0.0));
        }

        let a_probs = math::softmax_rows(&a_logits);

        // Gradients at logits.
        let mut a_grads = math::compute_gradients_step(&a_probs, v_target_ids);
        math::sanitize_gradients_inplace(&mut a_grads);

        // Unscaled grad norm.
        let d_norm_unscaled = {
            let mut d_sum: f32 = 0.0;
            for &d in a_grads.iter() {
                d_sum += d * d;
            }
            math::sanitize_f32(d_sum.sqrt())
        };

        // Scaled grad norm: apply inverse participation scaling on logits grads as proxy.
        // NOTE: True scaling is applied at branch backward, but this proxy captures magnitude shift.
        let mut a_grads_scaled = a_grads.clone();
        if cl_cfg.b_scale_by_inverse_participation {
            // Apply average scale over active branches.
            let mut d_scale_sum: f32 = 0.0;
            let mut i_cnt: usize = 0;
            for (i_b, &b_on) in v_train_mask.iter().enumerate() {
                if !b_on {
                    continue;
                }
                let p = cl_cfg.v_branch_participation_p.get(i_b).copied().unwrap_or(1.0).max(1e-6);
                let d_inv = (1.0 / p).clamp(1.0, 5.0);
                d_scale_sum += d_inv;
                i_cnt = i_cnt.saturating_add(1);
            }
            let d_avg_scale = if i_cnt == 0 { 1.0 } else { d_scale_sum / (i_cnt as f32).max(1.0) };
            a_grads_scaled.mapv_inplace(|x| x * d_avg_scale);
        }

        let d_norm_scaled = {
            let mut d_sum: f32 = 0.0;
            for &d in a_grads_scaled.iter() {
                d_sum += d * d;
            }
            math::sanitize_f32(d_sum.sqrt())
        };

        Ok((d_norm_scaled, d_norm_unscaled))
    }
    fn try_autonomous_expand_first_pg_ascii(
        &mut self,
        cfg_phase: &phase_strategy_config_ascii,
        v_diag_inputs: &[Array2<f32>],
    ) -> Result<bool, String> {
        if !cfg_phase.b_enable_autonomous_expansion {
            return Ok(false);
        }

        let opt_i_pg = self.find_first_parallel_block_group_index_ascii();
        let i_pg = match opt_i_pg {
            Some(i) => i,
            None => return Ok(false),
        };

        // Downcast.
        let pg_num_branches = {
            let pg = self.network[i_pg]
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                .ok_or_else(|| "expand_pg_downcast_failed".to_string())?;
            pg.num_branches()
        };

        if pg_num_branches >= cfg_phase.i_max_total_branches {
            return Ok(false);
        }

        // Compute diagnostics on provided inputs.
        let m = {
            let pg = self.network[i_pg]
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                .ok_or_else(|| "expand_pg_downcast_failed".to_string())?;
            pg.compute_metrics_from_inputs(v_diag_inputs)?
        };

        // Simple, conservative trigger rules.
        let b_starved = m.d_path_starvation_index.is_finite() && m.d_path_starvation_index > 0.60;
        let b_collapsed = m.d_top1_share.is_finite() && m.d_top1_share > 0.70;
        let b_low_eff = m.d_effective_num_paths.is_finite() && m.d_effective_num_paths < 2.0;

        if !(b_starved || b_collapsed || b_low_eff) {
            return Ok(false);
        }

        // Expand by one new branch (safe default).
        // In this codebase, branches are TransformerSequence of 2 TransformerBlocks.
        let b_new_branch: Box<dyn Layer> = {
            let tb1 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
            let tb2 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
            let seq = TransformerSequence::new(vec![tb1, tb2])?;
            Box::new(seq)
        };

        {
            let pg = self.network[i_pg]
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                .ok_or_else(|| "expand_pg_downcast_failed".to_string())?;

            pg.add_branches_with_conservative_injection_ascii(vec![b_new_branch], cfg_phase.d_eta_injection)?;
        }

        Ok(true)
    }

    pub fn predict(&mut self, s_text: &str) -> Result<String, String> {
        let b_prev = self.b_training;
        self.set_training(false);

        if self.b_outage_simulation_enabled {
            self.set_predict_outage_for_all_parallel_groups_test_only();
        }

        let r = self
            .forward_generate(s_text)
            .map(|v_out_ids| self.decode_ids(&v_out_ids));

        if self.b_outage_simulation_enabled {
            self.clear_predict_outage_for_all_parallel_groups_test_only();
        }

        self.set_training(b_prev);
        r
    }

    fn forward_generate_with_stats(&mut self, s_text: &str) -> Result<(Vec<usize>, PredictStats), String> {
        let mut v_context = self.tokenize(s_text)?;
        let mut v_generated: Vec<usize> = Vec::new();

        if v_context.len() >= MAX_SEQ_LEN {
            return Ok((v_generated, PredictStats::empty()));
        }

        let opt_eos = self.vocab.encode(S_EOS);

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

            let a_probs = math::softmax_rows(&a_scaled);
            if a_probs.nrows() != 1 || a_probs.ncols() != i_vocab {
                return Err("sampling_probs_shape_invalid".to_string());
            }

            let mut v_p: Vec<f32> = Vec::with_capacity(i_vocab);
            for j in 0..i_vocab {
                v_p.push(math::clamp_prob_f32(a_probs[[0, j]]));
            }
            v_entropies.push(math::entropy_nat_f32(&v_p));
            v_margins.push(math::top1_top2_margin_f32(&v_p));

            let i_next = self.sample_next_token_from_logits(&a_last_logits)?;

            let d_p_sel = if i_next < i_vocab { a_probs[[0, i_next]] } else { 0.0 };
            v_selected_probs.push(math::clamp_prob_f32(d_p_sel));

            v_generated.push(i_next);
            v_context.push(i_next);

            if let Some(i_eos) = opt_eos {
                if i_next == i_eos {
                    break;
                }
            }
        }

        let d_avg_p_sel = math::mean_vec_f32(&v_selected_probs);
        let d_ppl = math::perplexity_from_selected_probs_f32(&v_selected_probs);
        let d_avg_h = math::mean_vec_f32(&v_entropies);
        let d_avg_margin = math::mean_vec_f32(&v_margins);

        let stats = PredictStats {
            d_avg_selected_token_prob: d_avg_p_sel,
            d_perplexity_selected: d_ppl,
            d_avg_next_token_entropy_nat: d_avg_h,
            d_avg_top1_top2_margin: d_avg_margin,
            i_steps: v_selected_probs.len(),
        };

        Ok((v_generated, stats))
    }

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

    fn sample_availability_mask(rng: &mut StdRng, v_p: &[f32], i_min_active: usize) -> Vec<bool> {
        let i_k = v_p.len();
        if i_k == 0 {
            return Vec::new();
        }

        let mut v_mask = vec![false; i_k];
        let mut i_active: usize = 0;

        for i in 0..i_k {
            let d_u: f32 = rng.gen_range(0.0..1.0);
            let d_pp = v_p[i].clamp(0.0, 1.0);
            if d_u < d_pp {
                v_mask[i] = true;
                i_active = i_active.saturating_add(1);
            }
        }

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

    fn forward_from_layer_index_ascii(
        &mut self,
        i_start_layer: usize,
        a_input: &Array2<f32>,
    ) -> Result<Array2<f32>, String> {
        if i_start_layer >= self.network.len() {
            return Err("forward_from_layer_index_start_oob".to_string());
        }
        if a_input.nrows() == 0 || a_input.ncols() == 0 {
            return Err("forward_from_layer_index_empty_input".to_string());
        }

        let mut a_act = a_input.clone();
        for i_l in i_start_layer..self.network.len() {
            a_act = self.network[i_l].forward(&a_act);
            if a_act.nrows() == 0 || a_act.ncols() == 0 {
                return Err("forward_from_layer_index_empty_act".to_string());
            }
        }

        Ok(a_act)
    }

    fn make_single_branch_mask_ascii(i_k: usize, i_branch: usize) -> Vec<bool> {
        let mut v = vec![false; i_k];
        if i_branch < i_k {
            v[i_branch] = true;
        }
        v
    }

    fn evaluate_branch_loss_one_example_ascii(
        &mut self,
        i_pg: usize,
        a_act_before_pg: &Array2<f32>,
        v_target_ids: &[usize],
        v_mask_single_branch: &[bool],
    ) -> Result<f32, String> {
        if i_pg >= self.network.len() {
            return Err("eval_branch_loss_pg_oob".to_string());
        }
        if a_act_before_pg.nrows() == 0 || a_act_before_pg.ncols() == 0 {
            return Err("eval_branch_loss_empty_pre_pg_act".to_string());
        }
        if v_target_ids.is_empty() {
            return Err("eval_branch_loss_empty_targets".to_string());
        }

        let a_after_pg = {
            let pg = self.network[i_pg]
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                .ok_or_else(|| "eval_branch_loss_pg_downcast_failed".to_string())?;
            pg.forward_with_availability_mask(a_act_before_pg, v_mask_single_branch)
        };

        if a_after_pg.nrows() == 0 || a_after_pg.ncols() == 0 {
            return Err("eval_branch_loss_empty_post_pg_act".to_string());
        }

        let a_logits = self.forward_from_layer_index_ascii(i_pg + 1, &a_after_pg)?;
        if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
            return Err("eval_branch_loss_empty_logits".to_string());
        }

        let a_probs = math::softmax_rows(&a_logits);
        let d_loss = math::cross_entropy_loss_step(&a_probs, v_target_ids);

        if !d_loss.is_finite() {
            return Err("eval_branch_loss_non_finite".to_string());
        }

        Ok(d_loss)
    }

    fn select_branch_min_ema_loss_ascii(
        &mut self,
        st_ema: &mut branch_loss_ema_state_ascii,
        i_pg: usize,
        a_act_before_pg: &Array2<f32>,
        v_target_ids: &[usize],
        v_available_mask: &[bool],
    ) -> Result<usize, String> {
        let i_k = v_available_mask.len();
        if i_k == 0 {
            return Err("select_branch_no_branches".to_string());
        }
        st_ema.ensure_len(i_k);

        for i_b in 0..i_k {
            if !v_available_mask[i_b] {
                continue;
            }
            let v_single = Self::make_single_branch_mask_ascii(i_k, i_b);
            let d_loss = match self.evaluate_branch_loss_one_example_ascii(
                i_pg,
                a_act_before_pg,
                v_target_ids,
                &v_single,
            ) {
                Ok(v) => v,
                Err(_) => continue,
            };
            st_ema.update_ema(i_b, d_loss);
        }

        let mut opt_best: Option<usize> = None;
        let mut d_best: f32 = f32::INFINITY;

        for i_b in 0..i_k {
            if !v_available_mask[i_b] {
                continue;
            }
            let d_score = st_ema.get_score_or_fallback(i_b, 1.0e9);
            if d_score < d_best {
                d_best = d_score;
                opt_best = Some(i_b);
            }
        }

        let i_sel = opt_best.ok_or_else(|| "select_branch_no_valid_candidate".to_string())?;
        st_ema.opt_last_selected_branch = Some(i_sel);
        Ok(i_sel)
    }

    fn train_one_row_continuous_learning_ascii(
        &mut self,
        v_row: &[usize],
        d_lr: f32,
        opt_pg_idx: Option<usize>,
        opt_cl: &Option<ContinuousLearningConfig>,
        rng_mask: &mut StdRng,
        st_branch_ema: &mut branch_loss_ema_state_ascii,
        // Diagnostics counters (shared with outer loop).
        i_skips_empty_act: &mut usize,
        i_skips_empty_logits: &mut usize,
        i_skips_pg_downcast_failed: &mut usize,
        i_skips_pg_no_branches: &mut usize,
        b_enable_ema_selection: bool,
    ) -> Result<(bool, f32), String> {
        if v_row.len() < 2 {
            return Ok((false, 0.0));
        }

        let v_input_ids = &v_row[..v_row.len() - 1];
        let v_target_ids = &v_row[1..];

        let mut a_input: Array2<f32> = Array2::zeros((1, v_input_ids.len()));
        let a_row = Array1::from_iter(v_input_ids.iter().map(|&x| x as f32));
        a_input.row_mut(0).assign(&a_row);

        let mut b_update_applied: bool = false;
        let mut d_step_loss: f32 = 0.0;

        if let Some(i_pg) = opt_pg_idx {
            // Forward prefix up to PG.
            let mut a_act = a_input;
            for i_l in 0..i_pg {
                a_act = self.network[i_l].forward(&a_act);
                if a_act.nrows() == 0 || a_act.ncols() == 0 {
                    *i_skips_empty_act = i_skips_empty_act.saturating_add(1);
                    return Ok((false, 0.0));
                }
            }

            // Branch count.
            let i_k = match self.network[i_pg]
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
            {
                Some(pg) => pg.num_branches(),
                None => {
                    *i_skips_pg_downcast_failed = i_skips_pg_downcast_failed.saturating_add(1);
                    return Ok((false, 0.0));
                }
            };

            if i_k == 0 {
                *i_skips_pg_no_branches = i_skips_pg_no_branches.saturating_add(1);
                return Ok((false, 0.0));
            }

            let cl_cfg = opt_cl.clone().unwrap_or_else(|| {
                ContinuousLearningConfig::new_default_for_num_branches(i_k)
            });
            cl_cfg.validate(i_k)?;

            // Availability mask.
            let v_available_mask = Self::sample_availability_mask(
                rng_mask,
                &cl_cfg.v_branch_participation_p,
                cl_cfg.i_min_active_branches,
            );

            if v_available_mask.len() != i_k {
                *i_skips_pg_no_branches = i_skips_pg_no_branches.saturating_add(1);
                return Ok((false, 0.0));
            }

            // EMA based selection within available set.
            let i_best_branch = if b_enable_ema_selection {
                match self.select_branch_min_ema_loss_ascii(
                    st_branch_ema,
                    i_pg,
                    &a_act,
                    v_target_ids,
                    &v_available_mask,
                ) {
                    Ok(i) => i,
                    Err(_) => usize::MAX,
                }
            } else {
                usize::MAX
            };

            let v_train_mask: Vec<bool> = if i_best_branch != usize::MAX && i_best_branch < i_k {
                Self::make_single_branch_mask_ascii(i_k, i_best_branch)
            } else {
                v_available_mask.clone()
            };

            let opt_scales: Option<Vec<f32>> = if cl_cfg.b_scale_by_inverse_participation {
                Some(
                    cl_cfg
                        .v_branch_participation_p
                        .iter()
                        .map(|&p| {
                            let d_inv = 1.0 / p.max(1e-6);
                            d_inv.clamp(1.0, 5.0)
                        })
                        .collect(),
                )
            } else {
                None
            };

            // Forward PG with mask.
            {
                let pg = self.network[i_pg]
                    .as_any_mut()
                    .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                    .ok_or_else(|| "pg_downcast_failed".to_string())?;

                a_act = pg.forward_with_availability_mask(&a_act, &v_train_mask);
            }

            if a_act.nrows() == 0 || a_act.ncols() == 0 {
                *i_skips_empty_act = i_skips_empty_act.saturating_add(1);
                return Ok((false, 0.0));
            }

            // Forward tail.
            for i_l in (i_pg + 1)..self.network.len() {
                a_act = self.network[i_l].forward(&a_act);
                if a_act.nrows() == 0 || a_act.ncols() == 0 {
                    *i_skips_empty_act = i_skips_empty_act.saturating_add(1);
                    return Ok((false, 0.0));
                }
            }

            let a_logits = a_act;
            if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
                *i_skips_empty_logits = i_skips_empty_logits.saturating_add(1);
                return Ok((false, 0.0));
            }

            // Loss and backward.
            let a_probs = math::softmax_rows(&a_logits);
            d_step_loss = math::cross_entropy_loss_step(&a_probs, v_target_ids);

            let mut a_grads = math::compute_gradients_step(&a_probs, v_target_ids);
            math::clip_gradients_global_norm(&mut a_grads, 5.0);

            for i_l in (i_pg + 1..self.network.len()).rev() {
                a_grads = self.network[i_l].backward(&a_grads, d_lr);
            }

            {
                let pg = self.network[i_pg]
                    .as_any_mut()
                    .and_then(|a| a.downcast_mut::<ParallelBlockGroup>())
                    .ok_or_else(|| "pg_downcast_failed".to_string())?;

                a_grads = pg.backward_with_availability_mask(
                    &a_grads,
                    d_lr,
                    &v_train_mask,
                    opt_scales.as_deref(),
                );
            }

            for i_l in (0..i_pg).rev() {
                a_grads = self.network[i_l].backward(&a_grads, d_lr);
            }

            b_update_applied = true;
        } else {
            // Fallback: full participation.
            let mut a_act = a_input;
            for layer in self.network.iter_mut() {
                a_act = layer.forward(&a_act);
            }

            let a_logits = a_act;
            if a_logits.nrows() == 0 || a_logits.ncols() == 0 {
                *i_skips_empty_logits = i_skips_empty_logits.saturating_add(1);
                return Ok((false, 0.0));
            }

            let a_probs = math::softmax_rows(&a_logits);
            d_step_loss = math::cross_entropy_loss_step(&a_probs, v_target_ids);

            let mut a_grads = math::compute_gradients_step(&a_probs, v_target_ids);
            math::clip_gradients_global_norm(&mut a_grads, 5.0);

            for layer in self.network.iter_mut().rev() {
                a_grads = layer.backward(&a_grads, d_lr);
            }

            b_update_applied = true;
        }

        Ok((b_update_applied, d_step_loss))
    }



// layer.rs
// Description: Patch - Integrate phase strategy and autonomous expansion into training loop.
// History:
// - 2026-02-14: Add phase oriented strategy with ramp up and autonomous expansion checks.
// Author: Marcus Schlieper

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
    cfg_phase: phase_strategy_config_ascii,
) -> Result<(), String> {
    // History:
    // - 2026-02-13: Add EMA stabilized branch selection and experience replay.
    // - 2026-02-14: Phase oriented strategy with replay ramp up and autonomous expansion.

    cfg_phase.validate()?;
    self.set_training(true);

    if v_data.is_empty() || i_epochs == 0 {
        return Err("invalid_training_args".to_string());
    }
    if !d_lr.is_finite() || d_lr <= 0.0 {
        return Err("invalid_learning_rate".to_string());
    }
    if s_phase.trim().is_empty() {
        return Err("invalid_phase".to_string());
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

    let mut opt_pg_idx: Option<usize> = None;
    for (i_idx, layer) in self.network.iter().enumerate() {
        if layer.layer_type() == "ParallelBlockGroup" {
            opt_pg_idx = Some(i_idx);
            break;
        }
    }

    let mut rng_mask = StdRng::seed_from_u64(
        opt_cl
            .as_ref()
            .map(|c| c.u64_mask_seed)
            .unwrap_or(20260213),
    );

    let mut st_branch_ema = branch_loss_ema_state_ascii::new(1, 0.2);

    // Replay buffer is enabled by cfg_phase, but probability is ramped up.
    let mut rb_replay = replay_buffer_ascii::new(if cfg_phase.b_enable_replay { 5000 } else { 0 }, 20260213);
    let i_replay_max_steps_per_row: usize = 1;

    let mut i_total_steps: usize = 0;

    let mut i_skips_empty_act: usize = 0;
    let mut i_skips_empty_logits: usize = 0;
    let mut i_skips_pg_downcast_failed: usize = 0;
    let mut i_skips_pg_no_branches: usize = 0;

    for i_epoch in 0..i_epochs {
        if b_cancel.load(AtomicOrdering::SeqCst) {
            return Ok(());
        }

        let mut d_total_loss_used: f32 = 0.0;
        let mut i_used_rows: usize = 0;
        let mut d_last_step_loss: f32 = 0.0;

        for v_row in v_tokenized_data.iter() {
            if b_cancel.load(AtomicOrdering::SeqCst) {
                return Ok(());
            }

            // Phase dependent toggles.
            let b_ema_active = cfg_phase.ema_is_active(i_total_steps);
            let d_replay_p = cfg_phase.replay_p_at_step(i_total_steps);

            // NOTE:
            // The current code uses select_branch_min_ema_loss_ascii inside train_one_row_continuous_learning_ascii.
            // For strict phase control, that function must receive a boolean to bypass EMA selection
            // and default to v_available_mask (no selective routing) while EMA is inactive.

            let (b_ok, d_loss) = self.train_one_row_continuous_learning_ascii(
                v_row,
                d_lr,
                opt_pg_idx,
                &opt_cl,
                &mut rng_mask,
                &mut st_branch_ema,
                &mut i_skips_empty_act,
                &mut i_skips_empty_logits,
                &mut i_skips_pg_downcast_failed,
                &mut i_skips_pg_no_branches,
                b_ema_active,
            )?;

            if !b_ok {
                continue;
            }

            d_last_step_loss = d_loss;
            d_total_loss_used += d_loss;
            i_used_rows = i_used_rows.saturating_add(1);
            i_total_steps = i_total_steps.saturating_add(1);

            let d_running_epoch_avg_loss: f32 = if i_used_rows == 0 {
                0.0
            } else {
                d_total_loss_used / (i_used_rows as f32).max(1.0)
            };

            // Replay update and optional replay steps.
            rb_replay.push_row(v_row);
            if rb_replay.is_enabled() && rb_replay.len() > 0 && d_replay_p > 0.0 {
                let d_u: f32 = rng_mask.gen_range(0.0..1.0);
                if d_u < d_replay_p {
                    for _ in 0..i_replay_max_steps_per_row {
                        if let Some(v_rep) = rb_replay.sample_row() {
                            let _ = self.train_one_row_continuous_learning_ascii(
                                &v_rep,
                                d_lr,
                                opt_pg_idx,
                                &opt_cl,
                                &mut rng_mask,
                                &mut st_branch_ema,
                                &mut i_skips_empty_act,
                                &mut i_skips_empty_logits,
                                &mut i_skips_pg_downcast_failed,
                                &mut i_skips_pg_no_branches,
                                b_ema_active,
                            )?;
                        }
                    }
                }
            }

            // Autonomous expansion check at fixed interval.
            if cfg_phase.b_enable_autonomous_expansion
                && cfg_phase.i_expand_check_every_steps > 0
                && (i_total_steps % cfg_phase.i_expand_check_every_steps) == 0
            {
                // Diagnostics inputs: reuse a small slice of already tokenized stream.
                // Build activations before PG using existing helper if available; otherwise use
                // collect_parallel_block_group_inputs_for_diagnostics with fixed prompts.
                let v_prompts: Vec<String> = vec![
                    "User: diagnostic prompt 1".to_string(),
                    "User: diagnostic prompt 2".to_string(),
                    "User: diagnostic prompt 3".to_string(),
                    "User: diagnostic prompt 4".to_string(),
                ];

                let v_diag_inputs = self
                    .collect_parallel_block_group_inputs_for_diagnostics(&v_prompts, 4)
                    .unwrap_or_else(|_| Vec::new());

                if !v_diag_inputs.is_empty() {
                    let _ = self.try_autonomous_expand_first_pg_ascii(&cfg_phase, &v_diag_inputs);
                }
            }

            // Progress event.
            if (i_total_steps % 25) == 0 {
                let mut ev = TrainingProgressEventAscii::new_basic(
                    s_phase,
                    i_epoch + 1,
                    i_epochs,
                    d_running_epoch_avg_loss,
                    d_last_step_loss,
                    0,
                    i_total_steps,
                );
                ev.i_skips_empty_act = i_skips_empty_act;
                ev.i_skips_empty_logits = i_skips_empty_logits;
                ev.i_skips_pg_downcast_failed = i_skips_pg_downcast_failed;
                ev.i_skips_pg_no_branches = i_skips_pg_no_branches;
                let _ = tx_progress.send(ev);
            }

            // Snapshot export for serving updates.
            if i_snapshot_every_steps > 0 && (i_total_steps % i_snapshot_every_steps) == 0 {
                if let Some(tx) = tx_snapshot.as_ref() {
                    let v_params = self.export_parameters_snapshot();
                    let _ = tx.send(v_params);
                }
            }

            // b_ema_active is currently not wired into train_one_row_continuous_learning_ascii.
            // This is intentionally not ignored: the bypass must be implemented next.
            let _ = b_ema_active;
        }

        let d_avg_loss: f32 = if i_used_rows == 0 {
            0.0
        } else {
            d_total_loss_used / (i_used_rows as f32).max(1.0)
        };

        let mut ev = TrainingProgressEventAscii::new_basic(
            s_phase,
            i_epoch + 1,
            i_epochs,
            d_avg_loss,
            d_last_step_loss,
            i_used_rows,
            i_total_steps,
        );
        ev.i_skips_empty_act = i_skips_empty_act;
        ev.i_skips_empty_logits = i_skips_empty_logits;
        ev.i_skips_pg_downcast_failed = i_skips_pg_downcast_failed;
        ev.i_skips_pg_no_branches = i_skips_pg_no_branches;
        let _ = tx_progress.send(ev);
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

                i_used_rows = i_used_rows.saturating_add(1);
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

    // Evaluate average loss on a fixed control set (token rows), without updating weights.
    // NOTE: This temporarily switches to eval mode and does forward-only.
    fn eval_control_set_loss_ascii(&mut self, v_token_rows: &[Vec<usize>], i_max_rows: usize) -> Result<f32, String> {
        if v_token_rows.is_empty() || i_max_rows == 0 {
            return Ok(0.0);
        }

        let b_prev = self.b_training;
        self.set_training(false);

        let mut d_sum: f32 = 0.0;
        let mut i_used: usize = 0;

        let i_take = i_max_rows.min(v_token_rows.len());
        for v_row in v_token_rows.iter().take(i_take) {
            if v_row.len() < 2 {
                continue;
            }
            let v_input_ids = &v_row[..v_row.len() - 1];
            let v_target_ids = &v_row[1..];

            let a_token_input: Array2<f32> = Array2::from_shape_vec(
                (1, v_input_ids.len()),
                v_input_ids.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            )
            .map_err(|_| "control_eval_shape_error".to_string())?;

            let mut a_act = a_token_input;
            for layer in self.network.iter_mut() {
                a_act = layer.forward(&a_act);
                if a_act.nrows() == 0 || a_act.ncols() == 0 {
                    a_act = Array2::zeros((0, 0));
                    break;
                }
            }

            if a_act.nrows() == 0 || a_act.ncols() == 0 {
                continue;
            }

            let a_probs = math::softmax_rows(&a_act);
            let d_loss = math::cross_entropy_loss_step(&a_probs, v_target_ids);
            if d_loss.is_finite() {
                d_sum += d_loss;
                i_used = i_used.saturating_add(1);
            }
        }

        self.set_training(b_prev);

        if i_used == 0 {
            Ok(0.0)
        } else {
            Ok(d_sum / (i_used as f32).max(1.0))
        }
    }
    // Compute drift proxy for a set of prompts by comparing last-step logits.
    // This is used around expansion events to quantify functional continuity.
    fn compute_logits_for_prompts_ascii(&mut self, v_prompts: &[String], i_max_samples: usize) -> Result<Vec<Array2<f32>>, String> {
        if v_prompts.is_empty() || i_max_samples == 0 {
            return Err("drift_prompts_invalid".to_string());
        }

        let b_prev = self.b_training;
        self.set_training(false);

        let mut v_out: Vec<Array2<f32>> = Vec::new();
        let i_take = i_max_samples.min(v_prompts.len());

        for s in v_prompts.iter().take(i_take) {
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
            .map_err(|_| "drift_shape_error_token_input".to_string())?;

            let mut a_act = a_token_input;
            for layer in self.network.iter_mut() {
                a_act = layer.forward(&a_act);
                if a_act.nrows() == 0 || a_act.ncols() == 0 {
                    a_act = Array2::zeros((0, 0));
                    break;
                }
            }
            if a_act.nrows() == 0 || a_act.ncols() == 0 {
                continue;
            }

            // Take last row logits.
            let a_last = a_act
                .row(a_act.nrows().saturating_sub(1))
                .to_owned()
                .insert_axis(Axis(0));
            v_out.push(a_last);
        }

        self.set_training(b_prev);

        if v_out.is_empty() {
            return Err("drift_no_valid_logits".to_string());
        }

        Ok(v_out)
    }
}

