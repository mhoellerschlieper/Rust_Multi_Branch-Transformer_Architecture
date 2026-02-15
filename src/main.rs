// main.rs
// Description: Binary entry point with interactive menu loop.
//              Builds model from tokenizer, supports checkpoint save and load, and provides
//              parallel training and serving via separate model instances.
//
//              Extensions:
//              - Background training thread with live metrics
//              - Parallel ask during training via separate serving model instance
//              - Snapshot based parameter hot updates from training to serving
//              - Continuous learning support for partial path availability and incremental updates
//              - Online data ingestion: add training data files while training is running
//
// History:
// - 2026-02-01: Add menu loop and checkpoint save and load.
// - 2026-02-07: Add MBT parallel block group layer to support multi branch topology.
// - 2026-02-08: Add predict_with_stats and post predict metrics.
// - 2026-02-13: Add background training, live metrics, cooperative stop.
// - 2026-02-13: Add serving model with snapshot updates for true parallel ask during training.
// - 2026-02-14: Add online ingestion channel and robust menu wiring for command b.
// - 2026-02-15: Add help function for menu and metrics documentation (ASCII).
// Author: Marcus Schlieper (ExpChat.ai)

#![allow(warnings)]

mod layer;
mod math;
mod tokenizer;
mod train;
mod utils;

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

use crate::layer::{
    ContinuousLearningConfig, Embeddings, Layer, Llm, OutputProjection, ParallelBlockGroup,
    PredictStats, TrainingDataEventAscii, TrainingProgressEventAscii, TransformerBlock,
    TransformerSequence, phase_strategy_config_ascii, training_phase_ascii,
};
use crate::tokenizer::{BpeTokenizer, BpeTokenizerConfig};
use crate::train::{Dataset, DatasetType};

pub const MAX_SEQ_LEN: usize = 80;
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;

fn read_line_ascii_trimmed() -> Result<String, String> {
    let mut s_input = String::new();
    std::io::stdin()
        .read_line(&mut s_input)
        .map_err(|_| "input_read_error".to_string())?;
    Ok(s_input.trim().to_string())
}

// Build a fresh model whose dimensions match the tokenizer vocab.
fn build_llm_from_tokenizer(bpe: crate::tokenizer::BpeTokenizer) -> Llm {
    let vocab = bpe.vocab.clone();

    let embeddings = Embeddings::new(vocab.clone());
    let block1 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);

    // MTB stage: parallel branches inside one logical layer position, now as sequences.
    let block2_1 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_2 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_3 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_4 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_5 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_6 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_7 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);
    let block2_8 = TransformerBlock::new(crate::EMBEDDING_DIM, crate::HIDDEN_DIM);

    let seq_2_1 =
        TransformerSequence::new(vec![block2_1, block2_2]).expect("transformer_sequence_new_failed");
    let seq_2_2 =
        TransformerSequence::new(vec![block2_3, block2_4]).expect("transformer_sequence_new_failed");
    let seq_2_3 =
        TransformerSequence::new(vec![block2_5, block2_6]).expect("transformer_sequence_new_failed");
    let seq_2_4 =
        TransformerSequence::new(vec![block2_7, block2_8]).expect("transformer_sequence_new_failed");

    let parallel_block2 = ParallelBlockGroup::new(vec![
        Box::new(seq_2_1) as Box<dyn Layer>,
        Box::new(seq_2_2) as Box<dyn Layer>,
        Box::new(seq_2_3) as Box<dyn Layer>,
        Box::new(seq_2_4) as Box<dyn Layer>,
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

    let _ = llm.set_sampling_config(0.9, 40, 0.95, 987654321);
    llm
}

fn topology_to_ascii_lines(llm: &mut Llm) -> Vec<String> {
    let mut v_out: Vec<String> = Vec::new();

    v_out.push("=== Model Topology (ASCII) ===".to_string());
    v_out.push(format!(
        "max_seq_len={}, embedding_dim={}, hidden_dim={}",
        crate::MAX_SEQ_LEN,
        crate::EMBEDDING_DIM,
        crate::HIDDEN_DIM
    ));
    v_out.push(format!("total_parameters={}", llm.total_parameters()));
    v_out.push("".to_string());

    for (i_idx, layer) in llm.network.iter_mut().enumerate() {
        let s_t = layer.layer_type().to_string();

        if s_t == "ParallelBlockGroup" {
            let opt_pg = layer.as_any_mut().and_then(|a| a.downcast_mut::<ParallelBlockGroup>());
            if let Some(pg) = opt_pg {
                v_out.push(format!(
                    "[{}] ParallelBlockGroup branches={}",
                    i_idx,
                    pg.num_branches()
                ));
                let v_branch_types = pg.branch_layer_types_ascii();
                for (i_b, s_bt) in v_branch_types.iter().enumerate() {
                    v_out.push(format!("  - branch[{}] {}", i_b, s_bt));
                }
                continue;
            }

            v_out.push(format!("[{}] ParallelBlockGroup (downcast_failed)", i_idx));
            continue;
        }

        v_out.push(format!(
            "[{}] {} parameters={}",
            i_idx,
            s_t,
            layer.parameters()
        ));
    }

    v_out
}

fn print_metrics_ascii(llm: &mut Llm) {
    println!();
    println!("=== Metrics (MTB diagnostics) ===");
    llm.run_post_load_mtb_diagnostics_ascii();
}

#[derive(Clone, Debug)]
struct predict_metrics_ascii {
    d_duration_ms: f64,
    i_generated_tokens: usize,
    d_tokens_per_sec: f64,
    i_input_tokens: usize,
    i_total_tokens: usize,
    i_output_chars: usize,
    d_avg_chars_per_token_out: f64,
    d_output_chars_per_sec: f64,
    d_effective_context_utilization: f64,
    d_avg_selected_token_prob: f64,
    d_perplexity_selected: f64,
    d_avg_next_token_entropy_nat: f64,
    d_avg_top1_top2_margin: f64,
    i_pred_stats_steps: usize,
}

#[derive(Clone, Debug)]
struct training_metrics_snapshot_ascii {
    b_running: bool,
    b_cancel_requested: bool,
    s_phase: String,
    i_epoch_current: usize,
    i_epochs_total: usize,
    d_last_epoch_loss: f32,
    d_last_step_loss: f32,
    i_rows_used_last_epoch: usize,
    i_total_steps: usize,
    s_last_error: String,

    i_skips_empty_act: usize,
    i_skips_empty_logits: usize,
    i_skips_pg_downcast_failed: usize,
    i_skips_pg_no_branches: usize,

    // ---- New metrics fields (mirrors TrainingProgressEventAscii) ----

    // 1) Ingestion
    d_ingest_rows_per_sec_window: f32,
    d_ingest_events_per_sec_window: f32,
    i_ingest_rows_added_total: usize,
    i_ingest_events_processed_total: usize,
    i_ingest_parse_errors_total: usize,
    i_ingest_rows_rejected_total: usize,
    i_ingest_pending_events_observed_peak: usize,

    // 2) Coverage
    d_coverage_ratio_used_over_available: f32,
    d_new_data_ratio_in_available: f32,
    i_new_rows_added_during_epoch: usize,
    i_epoch_token_rows_start: usize,
    i_epoch_token_rows_end: usize,

    // 3) Mask stats
    d_active_branches_mean: f32,
    d_active_branches_std: f32,
    i_active_branches_min: usize,
    i_active_branches_max: usize,
    d_mask_sparsity_mean: f32,
    d_mask_sparsity_std: f32,
    d_steps_at_min_active_share: f32,

    // 4) Scaling proxy
    d_grad_norm_ratio_scaled_over_unscaled_mean: f32,
    d_grad_norm_ratio_scaled_over_unscaled_std: f32,
    d_grad_norm_scaled_mean: f32,
    d_grad_norm_unscaled_mean: f32,

    // 5) Replay
    d_replay_share: f32,
    d_replay_p_last: f32,
    d_replay_delta_loss_mean: f32,
    d_replay_delta_loss_std: f32,

    // 6) Retention
    d_loss_control_old: f32,
    d_loss_control_new: f32,
    d_retention_delta_old: f32,
    d_retention_delta_new: f32,

    // 7) Fairness
    d_branch_select_gini: f32,
    d_branch_select_top1_share: f32,

    // 8) Snapshot (training side only in this patch)
    i_snapshots_sent_total: usize,

    // 9) Expansion telemetry
    i_expansion_events_total: usize,
    i_branches_before_last_expand: usize,
    i_branches_after_last_expand: usize,
    d_eta_injection_last: f32,
    d_sum_w_new_last: f32,

    // 10) Drift
    d_expand_drift_logits_l2_mean: f32,
    d_expand_drift_logits_l2_std: f32,
    d_expand_drift_logits_cos_dist_mean: f32,
    d_expand_drift_logits_cos_dist_std: f32,

    // EMA
    b_ema_active: bool,
    i_ema_last_selected_branch: isize,
}

impl training_metrics_snapshot_ascii {
    fn new_idle() -> Self {
        Self {
            b_running: false,
            b_cancel_requested: false,
            s_phase: "idle".to_string(),
            i_epoch_current: 0,
            i_epochs_total: 0,
            d_last_epoch_loss: 0.0,
            d_last_step_loss: 0.0,
            i_rows_used_last_epoch: 0,
            i_total_steps: 0,
            s_last_error: "".to_string(),

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

fn clamp_f64(d_x: f64, d_min: f64, d_max: f64) -> f64 {
    if !d_x.is_finite() {
        return d_min;
    }
    if d_x < d_min {
        d_min
    } else if d_x > d_max {
        d_max
    } else {
        d_x
    }
}

fn compute_predict_metrics_ascii(
    llm: &Llm,
    s_prompt: &str,
    s_output: &str,
    d_duration_ms: f64,
    opt_stats: Option<&PredictStats>,
) -> predict_metrics_ascii {
    let i_input_tokens = llm.tokenize(s_prompt).map(|v| v.len()).unwrap_or(0);
    let i_output_tokens = llm.tokenize(s_output).map(|v| v.len()).unwrap_or(0);
    let i_total_tokens = i_input_tokens.saturating_add(i_output_tokens);

    let d_sec = (d_duration_ms / 1000.0).max(1e-9);
    let d_tokens_per_sec = (i_output_tokens as f64) / d_sec;

    let i_output_chars = s_output.len();
    let d_avg_chars_per_token_out = if i_output_tokens == 0 {
        0.0
    } else {
        (i_output_chars as f64) / (i_output_tokens as f64)
    };
    let d_output_chars_per_sec = (i_output_chars as f64) / d_sec;

    let d_effective_context_utilization =
        (i_total_tokens as f64) / (crate::MAX_SEQ_LEN as f64).max(1.0);

    let (d_avg_p, d_ppl, d_h, d_margin, i_steps) = match opt_stats {
        Some(st) => (
            st.d_avg_selected_token_prob as f64,
            st.d_perplexity_selected as f64,
            st.d_avg_next_token_entropy_nat as f64,
            st.d_avg_top1_top2_margin as f64,
            st.i_steps,
        ),
        None => (0.0, 0.0, 0.0, 0.0, 0),
    };

    predict_metrics_ascii {
        d_duration_ms: clamp_f64(d_duration_ms, 0.0, 1.0e12),
        i_generated_tokens: i_output_tokens,
        d_tokens_per_sec: clamp_f64(d_tokens_per_sec, 0.0, 1.0e12),

        i_input_tokens,
        i_total_tokens,
        i_output_chars,
        d_avg_chars_per_token_out: clamp_f64(d_avg_chars_per_token_out, 0.0, 1.0e9),
        d_output_chars_per_sec: clamp_f64(d_output_chars_per_sec, 0.0, 1.0e12),
        d_effective_context_utilization: clamp_f64(d_effective_context_utilization, 0.0, 1.0),

        d_avg_selected_token_prob: clamp_f64(d_avg_p, 0.0, 1.0),
        d_perplexity_selected: clamp_f64(d_ppl, 0.0, 1.0e12),
        d_avg_next_token_entropy_nat: clamp_f64(d_h, 0.0, 1.0e12),
        d_avg_top1_top2_margin: clamp_f64(d_margin, 0.0, 1.0),
        i_pred_stats_steps: i_steps,
    }
}

fn print_predict_metrics_ascii(m: &predict_metrics_ascii) {
    println!();
    println!("=== Predict Metrics ===");
    println!("duration_ms: {:.3}", m.d_duration_ms);
    println!("generated_tokens: {}", m.i_generated_tokens);
    println!("tokens_per_sec: {:.3}", m.d_tokens_per_sec);

    println!("input_tokens: {}", m.i_input_tokens);
    println!("total_tokens: {}", m.i_total_tokens);
    println!("output_chars: {}", m.i_output_chars);
    println!("avg_chars_per_token_out: {:.3}", m.d_avg_chars_per_token_out);
    println!("output_chars_per_sec: {:.3}", m.d_output_chars_per_sec);
    println!(
        "effective_context_utilization: {:.6}",
        m.d_effective_context_utilization
    );

    println!("avg_selected_token_prob: {:.6}", m.d_avg_selected_token_prob);
    println!("perplexity_selected: {:.6}", m.d_perplexity_selected);
    println!(
        "avg_next_token_entropy_nat: {:.6}",
        m.d_avg_next_token_entropy_nat
    );
    println!("avg_top1_top2_margin: {:.6}", m.d_avg_top1_top2_margin);
    println!("pred_stats_steps: {}", m.i_pred_stats_steps);
}

fn drain_training_progress_non_blocking(
    opt_rx: &mut Option<std::sync::mpsc::Receiver<crate::layer::TrainingProgressEventAscii>>,
    metrics_shared: &std::sync::Arc<std::sync::Mutex<training_metrics_snapshot_ascii>>,
    b_cancel_train: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::mpsc;

    let rx = match opt_rx.as_mut() {
        Some(r) => r,
        None => return,
    };

    loop {
        let ev = match rx.try_recv() {
            Ok(v) => v,
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                *opt_rx = None;
                break;
            }
        };

        let mut m = match metrics_shared.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        m.b_running = true;
        m.b_cancel_requested = b_cancel_train.load(std::sync::atomic::Ordering::SeqCst);
        m.s_phase = ev.s_phase;
        m.i_epoch_current = ev.i_epoch_current;
        m.i_epochs_total = ev.i_epochs_total;
        m.d_last_epoch_loss = ev.d_last_epoch_loss;
        m.d_last_step_loss = ev.d_last_step_loss;
        m.i_rows_used_last_epoch = ev.i_rows_used_last_epoch;
        m.i_total_steps = ev.i_total_steps;

        m.i_skips_empty_act = ev.i_skips_empty_act;
        m.i_skips_empty_logits = ev.i_skips_empty_logits;
        m.i_skips_pg_downcast_failed = ev.i_skips_pg_downcast_failed;
        m.i_skips_pg_no_branches = ev.i_skips_pg_no_branches;

        // New metrics copy.
        m.d_ingest_rows_per_sec_window = ev.d_ingest_rows_per_sec_window;
        m.d_ingest_events_per_sec_window = ev.d_ingest_events_per_sec_window;
        m.i_ingest_rows_added_total = ev.i_ingest_rows_added_total;
        m.i_ingest_events_processed_total = ev.i_ingest_events_processed_total;
        m.i_ingest_parse_errors_total = ev.i_ingest_parse_errors_total;
        m.i_ingest_rows_rejected_total = ev.i_ingest_rows_rejected_total;
        m.i_ingest_pending_events_observed_peak = ev.i_ingest_pending_events_observed_peak;

        m.d_coverage_ratio_used_over_available = ev.d_coverage_ratio_used_over_available;
        m.d_new_data_ratio_in_available = ev.d_new_data_ratio_in_available;
        m.i_new_rows_added_during_epoch = ev.i_new_rows_added_during_epoch;
        m.i_epoch_token_rows_start = ev.i_epoch_token_rows_start;
        m.i_epoch_token_rows_end = ev.i_epoch_token_rows_end;

        m.d_active_branches_mean = ev.d_active_branches_mean;
        m.d_active_branches_std = ev.d_active_branches_std;
        m.i_active_branches_min = ev.i_active_branches_min;
        m.i_active_branches_max = ev.i_active_branches_max;
        m.d_mask_sparsity_mean = ev.d_mask_sparsity_mean;
        m.d_mask_sparsity_std = ev.d_mask_sparsity_std;
        m.d_steps_at_min_active_share = ev.d_steps_at_min_active_share;

        m.d_grad_norm_ratio_scaled_over_unscaled_mean = ev.d_grad_norm_ratio_scaled_over_unscaled_mean;
        m.d_grad_norm_ratio_scaled_over_unscaled_std = ev.d_grad_norm_ratio_scaled_over_unscaled_std;
        m.d_grad_norm_scaled_mean = ev.d_grad_norm_scaled_mean;
        m.d_grad_norm_unscaled_mean = ev.d_grad_norm_unscaled_mean;

        m.d_replay_share = ev.d_replay_share;
        m.d_replay_p_last = ev.d_replay_p_last;
        m.d_replay_delta_loss_mean = ev.d_replay_delta_loss_mean;
        m.d_replay_delta_loss_std = ev.d_replay_delta_loss_std;

        m.d_loss_control_old = ev.d_loss_control_old;
        m.d_loss_control_new = ev.d_loss_control_new;
        m.d_retention_delta_old = ev.d_retention_delta_old;
        m.d_retention_delta_new = ev.d_retention_delta_new;

        m.d_branch_select_gini = ev.d_branch_select_gini;
        m.d_branch_select_top1_share = ev.d_branch_select_top1_share;

        m.i_snapshots_sent_total = ev.i_snapshots_sent_total;

        m.i_expansion_events_total = ev.i_expansion_events_total;
        m.i_branches_before_last_expand = ev.i_branches_before_last_expand;
        m.i_branches_after_last_expand = ev.i_branches_after_last_expand;
        m.d_eta_injection_last = ev.d_eta_injection_last;
        m.d_sum_w_new_last = ev.d_sum_w_new_last;

        m.d_expand_drift_logits_l2_mean = ev.d_expand_drift_logits_l2_mean;
        m.d_expand_drift_logits_l2_std = ev.d_expand_drift_logits_l2_std;
        m.d_expand_drift_logits_cos_dist_mean = ev.d_expand_drift_logits_cos_dist_mean;
        m.d_expand_drift_logits_cos_dist_std = ev.d_expand_drift_logits_cos_dist_std;

        m.b_ema_active = ev.b_ema_active;
        m.i_ema_last_selected_branch = ev.i_ema_last_selected_branch;
    }
}
fn print_training_metrics_snapshot_ascii(m: &training_metrics_snapshot_ascii) {
    println!();
    println!("=== Training Metrics ===");
    println!("running: {}", m.b_running);
    println!("cancel_requested: {}", m.b_cancel_requested);
    println!("phase: {}", m.s_phase);
    println!("epoch: {} / {}", m.i_epoch_current, m.i_epochs_total);
    println!("last_epoch_loss: {:.6}", m.d_last_epoch_loss);
    println!("last_step_loss: {:.6}", m.d_last_step_loss);
    println!("rows_used_last_epoch: {}", m.i_rows_used_last_epoch);
    println!("total_steps: {}", m.i_total_steps);

    println!("skips_empty_act: {}", m.i_skips_empty_act);
    println!("skips_empty_logits: {}", m.i_skips_empty_logits);
    println!("skips_pg_downcast_failed: {}", m.i_skips_pg_downcast_failed);
    println!("skips_pg_no_branches: {}", m.i_skips_pg_no_branches);

    println!();
    println!("--- Continuous Learning Metrics ---");

    // 1) Ingestion throughput and queue proxy.
    println!("ingest_rows_per_sec_window: {:.3}", m.d_ingest_rows_per_sec_window);
    println!("ingest_events_per_sec_window: {:.3}", m.d_ingest_events_per_sec_window);
    println!("ingest_rows_added_total: {}", m.i_ingest_rows_added_total);
    println!("ingest_events_processed_total: {}", m.i_ingest_events_processed_total);
    println!("ingest_parse_errors_total: {}", m.i_ingest_parse_errors_total);
    println!("ingest_rows_rejected_total: {}", m.i_ingest_rows_rejected_total);
    println!(
        "ingest_pending_events_observed_peak: {}",
        m.i_ingest_pending_events_observed_peak
    );

    // 2) Coverage ratio.
    println!(
        "coverage_ratio_used_over_available: {:.6}",
        m.d_coverage_ratio_used_over_available
    );
    println!(
        "new_data_ratio_in_available: {:.6}",
        m.d_new_data_ratio_in_available
    );
    println!("new_rows_added_during_epoch: {}", m.i_new_rows_added_during_epoch);
    println!("epoch_token_rows_start: {}", m.i_epoch_token_rows_start);
    println!("epoch_token_rows_end: {}", m.i_epoch_token_rows_end);

    // 3) Mask participation.
    println!("active_branches_mean: {:.6}", m.d_active_branches_mean);
    println!("active_branches_std: {:.6}", m.d_active_branches_std);
    println!("active_branches_min: {}", m.i_active_branches_min);
    println!("active_branches_max: {}", m.i_active_branches_max);
    println!("mask_sparsity_mean: {:.6}", m.d_mask_sparsity_mean);
    println!("mask_sparsity_std: {:.6}", m.d_mask_sparsity_std);
    println!("steps_at_min_active_share: {:.6}", m.d_steps_at_min_active_share);

    // 4) Inverse participation scaling impact.
    println!(
        "grad_norm_ratio_scaled_over_unscaled_mean: {:.6}",
        m.d_grad_norm_ratio_scaled_over_unscaled_mean
    );
    println!(
        "grad_norm_ratio_scaled_over_unscaled_std: {:.6}",
        m.d_grad_norm_ratio_scaled_over_unscaled_std
    );
    println!("grad_norm_scaled_mean: {:.6}", m.d_grad_norm_scaled_mean);
    println!("grad_norm_unscaled_mean: {:.6}", m.d_grad_norm_unscaled_mean);

    // 5) Replay usage and effect.
    println!("replay_share: {:.6}", m.d_replay_share);
    println!("replay_p_last: {:.6}", m.d_replay_p_last);
    println!("replay_delta_loss_mean: {:.6}", m.d_replay_delta_loss_mean);
    println!("replay_delta_loss_std: {:.6}", m.d_replay_delta_loss_std);

    // 6) Retention / forgetting.
    println!("loss_control_old: {:.6}", m.d_loss_control_old);
    println!("loss_control_new: {:.6}", m.d_loss_control_new);
    println!("retention_delta_old: {:.6}", m.d_retention_delta_old);
    println!("retention_delta_new: {:.6}", m.d_retention_delta_new);

    // 7) Fairness.
    println!("branch_select_gini: {:.6}", m.d_branch_select_gini);
    println!("branch_select_top1_share: {:.6}", m.d_branch_select_top1_share);

    // 8) Snapshot counters (latency and staleness require receiver side instrumentation).
    println!("snapshots_sent_total: {}", m.i_snapshots_sent_total);

    // 9) Expansion telemetry.
    println!("expansion_events_total: {}", m.i_expansion_events_total);
    println!(
        "branches_before_last_expand: {}",
        m.i_branches_before_last_expand
    );
    println!(
        "branches_after_last_expand: {}",
        m.i_branches_after_last_expand
    );
    println!("eta_injection_last: {:.6}", m.d_eta_injection_last);
    println!("sum_w_new_last: {:.6}", m.d_sum_w_new_last);

    // 10) Drift proxy.
    println!(
        "expand_drift_logits_l2_mean: {:.6}",
        m.d_expand_drift_logits_l2_mean
    );
    println!(
        "expand_drift_logits_l2_std: {:.6}",
        m.d_expand_drift_logits_l2_std
    );
    println!(
        "expand_drift_logits_cos_dist_mean: {:.6}",
        m.d_expand_drift_logits_cos_dist_mean
    );
    println!(
        "expand_drift_logits_cos_dist_std: {:.6}",
        m.d_expand_drift_logits_cos_dist_std
    );

    // EMA.
    println!("ema_active: {}", m.b_ema_active);
    println!("ema_last_selected_branch: {}", m.i_ema_last_selected_branch);

    if !m.s_last_error.is_empty() {
        println!("last_error: {}", m.s_last_error);
    }
}

// Help function: explains all menu items and all metrics in ASCII only.
fn print_help_ascii() {
    println!();
    println!("=== Help (ASCII) ===");
    println!();
    println!("Menu commands:");
    println!("  t  Train (background, continuous learning)");
    println!("     - Starts background training on llm_train.");
    println!("     - Serving (ask) continues on llm_serve and receives snapshot updates.");
    println!("     - Training uses continuous learning mask logic (partial branch availability),");
    println!("       optional EMA based branch selection, replay, and optional autonomous expansion.");
    println!();
    println!("  b  Training metrics");
    println!("     - Prints last training progress snapshot received from training thread.");
    println!("     - Includes base loss metrics, diagnostic skip counters, and advanced validation metrics.");
    println!();
    println!("  s  Stop training");
    println!("     - Requests cooperative cancellation and joins training thread.");
    println!("     - Also signals shutdown to online ingestion receiver when present.");
    println!();
    println!("  n  Add new training data file (online ingestion)");
    println!("     - Requires running training thread.");
    println!("     - Expects a JSON file containing an array of strings (training examples).");
    println!("     - Data are tokenized and appended to the active training pool (append only).");
    println!();
    println!("  l  Load checkpoint (serve model)");
    println!("     - Loads checkpoint file and rebuilds topology for llm_serve only.");
    println!("     - Training model instance is not replaced by this action.");
    println!();
    println!("  w  Save checkpoint (serve model)");
    println!("     - Saves tokenizer, topology spec, and all parameters from llm_serve.");
    println!();
    println!("  a  Ask (serve model, parallel to training)");
    println!("     - Interactive inference loop on llm_serve.");
    println!("     - Reports prediction metrics (throughput, entropy, margin, perplexity proxy).");
    println!();
    println!("  o  Toggle outage simulation (serve model, test only)");
    println!("     - Enables fault injection in ParallelBlockGroup during predict.");
    println!("     - Drops a randomly chosen branch per predict call for robustness testing.");
    println!();
    println!("  y  Topology (ASCII, serve model)");
    println!("     - Prints layer list and for ParallelBlockGroup also branch layer types.");
    println!();
    println!("  x  Metrics (MTB diagnostics, serve model)");
    println!("     - Runs post load diagnostics on ParallelBlockGroup to quantify path usage and diversity.");
    println!();
    println!("  h  Help");
    println!("     - Prints this help text.");
    println!();
    println!("  e  Exit");
    println!("     - Requests ingestion shutdown, cancels training if running, joins thread, exits program.");
    println!();
    println!("Training metrics (printed by command b):");
    println!("  Base progress fields:");
    println!("    phase: current phase name (e.g. pretraining, instruction_tuning)");
    println!("    epoch: current epoch and total epochs");
    println!("    last_epoch_loss: mean loss of last completed epoch (running mean during epoch in step events)");
    println!("    last_step_loss: last observed step loss (cross entropy)");
    println!("    rows_used_last_epoch: number of training rows that produced a valid update");
    println!("    total_steps: number of successful update steps across epochs");
    println!();
    println!("  Diagnostic skip counters:");
    println!("    skips_empty_act: forward produced empty activations, row skipped");
    println!("    skips_empty_logits: forward produced empty logits, row skipped");
    println!("    skips_pg_downcast_failed: ParallelBlockGroup downcast failed, row skipped");
    println!("    skips_pg_no_branches: ParallelBlockGroup had zero branches, row skipped");
    println!();
    println!("  Advanced validation metrics (continuous learning and expandable width):");
    println!("  (1) Ingestion throughput and queue proxies:");
    println!("    ingest_rows_per_sec_window: windowed rate of accepted rows added to token pool");
    println!("    ingest_events_per_sec_window: windowed rate of processed ingestion events");
    println!("    ingest_rows_added_total: total accepted rows added since start of phase");
    println!("    ingest_events_processed_total: total processed ingestion events since start of phase");
    println!("    ingest_parse_errors_total: total parse or tokenization errors during ingestion");
    println!("    ingest_rows_rejected_total: total rows rejected (empty, too short, over budget)");
    println!("    ingest_pending_events_observed_peak: coarse proxy for pending events observed during drains");
    println!();
    println!("  (2) Coverage ratio (effective data coverage per epoch):");
    println!("    epoch_token_rows_start: token rows available at epoch start (snapshot length)");
    println!("    epoch_token_rows_end: token rows available at epoch end (after ingestion)");
    println!("    new_rows_added_during_epoch: epoch_token_rows_end - epoch_token_rows_start");
    println!("    coverage_ratio_used_over_available: used_rows / epoch_token_rows_start (approx)");
    println!("    new_data_ratio_in_available: new_rows_added_during_epoch / epoch_token_rows_end");
    println!();
    println!("  (3) Availability mask statistics (participation and sparsity):");
    println!("    active_branches_mean/std/min/max: statistics of active branches per step");
    println!("    mask_sparsity_mean/std: fraction of inactive branches per step");
    println!("    steps_at_min_active_share: share of steps where active branches equal min_active");
    println!();
    println!("  (4) Inverse participation scaling impact (unbiasedness proxy):");
    println!("    grad_norm_ratio_scaled_over_unscaled_mean/std: proxy ratio for gradient magnitude");
    println!("    grad_norm_scaled_mean/unscaled_mean: proxy means of compared gradient norms");
    println!("    Note: this implementation uses a lightweight proxy and does not clone full weights.");
    println!();
    println!("  (5) Replay usage and replay effect strength:");
    println!("    replay_share: replay_steps / (fresh_steps + replay_steps)");
    println!("    replay_p_last: last replay probability used by phase strategy");
    println!("    replay_delta_loss_mean/std: mean/std of (loss_replay - loss_fresh) pairs");
    println!();
    println!("  (6) Forgetting indicator (retention score) on fixed control sets:");
    println!("    loss_control_old/new: forward only loss on fixed control slices");
    println!("    retention_delta_old/new: loss_now - loss_baseline (positive indicates forgetting)");
    println!("    Note: control sets are deterministic slices of initial token pool in this implementation.");
    println!();
    println!("  (7) Branch selection fairness and dominance:");
    println!("    branch_select_gini: Gini coefficient of EMA selection frequencies");
    println!("    branch_select_top1_share: max selection share across branches");
    println!("    Interpretation: higher values indicate dominance and possible path starvation.");
    println!();
    println!("  (8) Snapshot telemetry (train to serve):");
    println!("    snapshots_sent_total: count of parameter snapshots sent from training");
    println!("    Note: latency and staleness require snapshot metadata in payload;");
    println!("          receiver side must carry send_time_ms and train_step for full measurement.");
    println!();
    println!("  (9) Expansion events and injection telemetry:");
    println!("    expansion_events_total: number of width expansion operations performed");
    println!("    branches_before_last_expand / branches_after_last_expand: last expansion size");
    println!("    eta_injection_last: conservative injection parameter used by phase strategy");
    println!("    sum_w_new_last: approximate total weight mass assigned to new branches (best effort)");
    println!();
    println!("  (10) Functional continuity on expansion (output drift proxy):");
    println!("    expand_drift_logits_l2_mean/std: L2 distance of last step logits before vs after expansion");
    println!("    expand_drift_logits_cos_dist_mean/std: cosine distance of last step logits before vs after");
    println!("    Interpretation: smaller values indicate more stable behavior under expansion.");
    println!();
    println!("  EMA state:");
    println!("    ema_active: whether EMA based branch selection is active at current step");
    println!("    ema_last_selected_branch: last selected branch index (or -1 if not applicable)");
    println!();
    println!("Prediction metrics (printed after each ask in interactive mode):");
    println!("  duration_ms: wall clock time for predict");
    println!("  generated_tokens: number of output tokens generated (tokenizer based)");
    println!("  tokens_per_sec: throughput based on generated_tokens and duration");
    println!("  effective_context_utilization: (input_tokens + output_tokens) / MAX_SEQ_LEN");
    println!("  avg_selected_token_prob: mean probability of selected token across generation steps");
    println!("  perplexity_selected: exp(mean(-ln(p_selected))) proxy from selected token probabilities");
    println!("  avg_next_token_entropy_nat: mean entropy of next token distribution per step (nats)");
    println!("  avg_top1_top2_margin: mean difference between top1 and top2 probabilities per step");
    println!();
    
}

fn drain_snapshot_updates_non_blocking(
    opt_rx: &mut Option<std::sync::mpsc::Receiver<Vec<f32>>>,
    llm_serve: &std::sync::Arc<std::sync::Mutex<crate::layer::Llm>>,
) {
    use std::sync::mpsc;

    let rx = match opt_rx.as_mut() {
        Some(r) => r,
        None => return,
    };

    let mut opt_last: Option<Vec<f32>> = None;

    loop {
        match rx.try_recv() {
            Ok(v) => opt_last = Some(v),
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                *opt_rx = None;
                break;
            }
        }
    }

    if let Some(v_params) = opt_last {
        let mut llm = match llm_serve.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let _ = llm.import_parameters_snapshot(&v_params);

        // Minimal: no further metrics possible without snapshot metadata.
        // Full implementation requires SnapshotPacketAscii as described above.
    }
}

fn main() {
    let mut s_checkpoint_path: String = "../../checkpoints/llm_checkpoint.json".to_string();

    let dataset = Dataset::new(
        "../../data/data_to_pretrain.json",
        "../../data/data_to_train.json",
        DatasetType::JSON,
    );

    // Initial tokenizer training.
    let mut v_corpus: Vec<String> = Vec::new();
    v_corpus.extend(dataset.pretraining_data.clone());
    v_corpus.extend(dataset.chat_training_data.clone());

    let mut config = BpeTokenizerConfig::default();
    config.i_vocab_target = 2000;
    config.i_min_pair_count = 2;

    let bpe = match BpeTokenizer::train_from_corpus_with_config(&v_corpus, config) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("Tokenizer training failed: {}", e);
            return;
        }
    };

    // Two models: one for training (exclusive), one for serving (parallel asks).
    let llm_train: Arc<Mutex<Llm>> = Arc::new(Mutex::new(build_llm_from_tokenizer(bpe.clone())));
    let llm_serve: Arc<Mutex<Llm>> = Arc::new(Mutex::new(build_llm_from_tokenizer(bpe)));

    // Background training control and metrics.
    let b_cancel_train: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let metrics_shared: Arc<Mutex<training_metrics_snapshot_ascii>> =
        Arc::new(Mutex::new(training_metrics_snapshot_ascii::new_idle()));

    // Receivers in main thread.
    let mut opt_progress_rx: Option<mpsc::Receiver<TrainingProgressEventAscii>> = None;
    let mut opt_snapshot_rx: Option<mpsc::Receiver<Vec<f32>>> = None;

    // Online ingestion sender (main to training thread).
    let mut opt_data_tx: Option<mpsc::Sender<TrainingDataEventAscii>> = None;

    // Training thread handle.
    let mut opt_train_handle: Option<thread::JoinHandle<()>> = None;

    {
        let llm = llm_serve.lock().expect("llm_mutex_poisoned");
        println!("\n=== MODEL INFORMATION ===");
        println!("Network architecture: {}", llm.network_description());
        println!(
            "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
            MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
        );
        println!("Total parameters: {}", llm.total_parameters());
    }

    loop {
        // Keep live state fresh.
        drain_training_progress_non_blocking(&mut opt_progress_rx, &metrics_shared, &b_cancel_train);
        drain_snapshot_updates_non_blocking(&mut opt_snapshot_rx, &llm_serve);

        println!("\n--- Menu Mode ---");
        println!("Commands:");
        println!("  t Train (background, continuous learning)");
        println!("  b Training metrics");
        println!("  s Stop training");
        println!("  n Add new training data file (online ingestion)");
        println!("  l Load checkpoint (serve model)");
        println!("  w Save checkpoint (serve model)");
        println!("  a Ask (serve model, parallel to training)");
        println!("  o Toggle outage simulation (serve model, test only)");
        println!("  y Topology (ASCII, serve model)");
        println!("  x Metrics (MTB diagnostics, serve model)");
        println!("  h Help");
        println!("  e Exit");

        print!("\nEnter command: ");
        let _ = std::io::stdout().flush();

        let s_cmd = match read_line_ascii_trimmed() {
            Ok(s) => s,
            Err(e) => {
                println!("Input error: {}", e);
                continue;
            }
        };

        // Drain again after input.
        drain_training_progress_non_blocking(&mut opt_progress_rx, &metrics_shared, &b_cancel_train);
        drain_snapshot_updates_non_blocking(&mut opt_snapshot_rx, &llm_serve);

        let s_cmd_lc = s_cmd.to_lowercase();


        if s_cmd_lc == "e" {
            if let Some(tx) = opt_data_tx.as_ref() {
                let _ = tx.send(TrainingDataEventAscii::shutdown);
            }

            if let Some(h) = opt_train_handle.take() {
                b_cancel_train.store(true, Ordering::SeqCst);
                let _ = h.join();
            }

            println!("Exit.");
            break;
        }
        if s_cmd_lc == "h" {
            print_help_ascii();
            continue;
        }

        if s_cmd_lc == "t" {
            let b_already_running = {
                let m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_running
            };
            if b_already_running {
                println!("Training already running.");
                continue;
            }

            b_cancel_train.store(false, Ordering::SeqCst);
            {
                let mut m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                *m = training_metrics_snapshot_ascii::new_idle();
                m.b_running = true;
                m.s_phase = "starting".to_string();
            }

            // Progress channel.
            let (tx_progress, rx_progress) = mpsc::channel::<TrainingProgressEventAscii>();
            opt_progress_rx = Some(rx_progress);

            // Snapshot channel for serving updates.
            let (tx_snapshot, rx_snapshot) = mpsc::channel::<Vec<f32>>();
            opt_snapshot_rx = Some(rx_snapshot);

            // Online data ingestion channel.
            let (tx_data, rx_data) = mpsc::channel::<TrainingDataEventAscii>();
            opt_data_tx = Some(tx_data.clone());

            let llm_for_train = Arc::clone(&llm_train);
            let metrics_for_train = Arc::clone(&metrics_shared);
            let cancel_for_train = Arc::clone(&b_cancel_train);

            let v_pretraining_examples: Vec<String> = dataset.pretraining_data.clone();
            let v_chat_training_examples: Vec<String> = dataset.chat_training_data.clone();

            opt_train_handle = Some(thread::spawn(move || {
                let r_run = (|| -> Result<(), String> {
                    let i_snapshot_every_steps: usize = 200;

                    let i_epochs_total_pretrain = 30;
                    let i_epochs_total_train = 5000;

                    let cl_cfg_pre = ContinuousLearningConfig {
                        v_branch_participation_p: vec![0.75, 0.75, 0.75, 0.75],
                        i_min_active_branches: 2,
                        b_scale_by_inverse_participation: true,
                        u64_mask_seed: 20260213,
                    };

                    let cl_cfg_tune = ContinuousLearningConfig {
                        v_branch_participation_p: vec![0.60, 0.70, 0.80, 0.65],
                        i_min_active_branches: 2,
                        b_scale_by_inverse_participation: true,
                        u64_mask_seed: 20260214,
                    };

                    let cfg_phase_pre = phase_strategy_config_ascii {
                        e_phase: training_phase_ascii::realtime,
                        b_enable_ema_branch_selection: true,
                        i_ema_warmup_steps: 500,

                        b_enable_replay: true,
                        d_replay_p_start: 0.0,
                        d_replay_p_max: 0.25,
                        i_replay_ramp_steps: 2000,

                        b_enable_autonomous_expansion: true,
                        i_expand_check_every_steps: 500,
                        d_eta_injection: 0.05,

                        i_max_total_branches: 16,
                    };

                    let cfg_phase_tune = phase_strategy_config_ascii {
                        e_phase: training_phase_ascii::realtime,
                        b_enable_ema_branch_selection: true,
                        i_ema_warmup_steps: 500,

                        b_enable_replay: true,
                        d_replay_p_start: 0.0,
                        d_replay_p_max: 0.25,
                        i_replay_ramp_steps: 2000,

                        b_enable_autonomous_expansion: true,
                        i_expand_check_every_steps: 500,
                        d_eta_injection: 0.05,

                        i_max_total_branches: 16,
                    };

                    // Update phase in shared metrics.
                    {
                        let mut m = metrics_for_train
                            .lock()
                            .map_err(|_| "metrics_lock_failed".to_string())?;
                        m.s_phase = "pretraining".to_string();
                        m.i_epoch_current = 0;
                        m.i_epochs_total = i_epochs_total_pretrain;
                        m.s_last_error = "".to_string();
                    }

                    // Training orchestration: if available, keep receiver alive across phases.
                    // NOTE: This method must exist in layer.rs for the robust solution.
                    {
                        let mut llm = llm_for_train.lock().map_err(|_| "llm_lock_failed".to_string())?;

                        llm.train_two_phase_with_progress_online_ascii(
                            v_pretraining_examples.iter().map(|s| s.as_str()).collect(),
                            i_epochs_total_pretrain,
                            0.0005,
                            "pretraining",
                            Some(cl_cfg_pre),
                            cfg_phase_pre,
                            v_chat_training_examples.iter().map(|s| s.as_str()).collect(),
                            i_epochs_total_train,
                            0.0001,
                            "instruction_tuning",
                            Some(cl_cfg_tune),
                            cfg_phase_tune,
                            Arc::clone(&cancel_for_train),
                            tx_progress.clone(),
                            i_snapshot_every_steps,
                            Some(tx_snapshot.clone()),
                            rx_data,
                        )?;
                    }

                    Ok(())
                })();

                drop(tx_progress);
                drop(tx_snapshot);

                let mut m = match metrics_for_train.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                m.b_cancel_requested = cancel_for_train.load(Ordering::SeqCst);
                m.b_running = false;

                match r_run {
                    Ok(()) => {
                        if m.b_cancel_requested {
                            m.s_phase = "canceled".to_string();
                        } else {
                            m.s_phase = "done".to_string();
                        }
                    }
                    Err(e) => {
                        m.s_phase = "error".to_string();
                        m.s_last_error = e;
                    }
                }
            }));

            println!("Training started in background. Serving continues on llm_serve.");
            continue;
        }

        if s_cmd_lc == "b" {
            drain_training_progress_non_blocking(&mut opt_progress_rx, &metrics_shared, &b_cancel_train);
            let m = metrics_shared
                .lock()
                .expect("metrics_mutex_poisoned")
                .clone();
            print_training_metrics_snapshot_ascii(&m);
            continue;
        }

        if s_cmd_lc == "s" {
            let b_running = {
                let m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_running
            };
            if !b_running {
                println!("Training not running.");
                if let Some(h) = opt_train_handle.take() {
                    let _ = h.join();
                }
                opt_data_tx = None;
                continue;
            }

            b_cancel_train.store(true, Ordering::SeqCst);
            {
                let mut m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_cancel_requested = true;
                m.s_phase = "cancel_requested".to_string();
            }

            if let Some(tx) = opt_data_tx.as_ref() {
                let _ = tx.send(TrainingDataEventAscii::shutdown);
            }

            if let Some(h) = opt_train_handle.take() {
                let _ = h.join();
            }

            opt_data_tx = None;
            println!("Training stop requested and thread joined.");
            continue;
        }

        if s_cmd_lc == "n" {
            let b_running = {
                let m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_running
            };
            if !b_running {
                println!("Training not running. Online ingestion requires a running training thread.");
                continue;
            }

            let tx = match opt_data_tx.as_ref() {
                Some(v) => v,
                None => {
                    println!("Online ingestion channel not available.");
                    continue;
                }
            };

            print!("Enter path to JSON training file (array of strings): ");
            let _ = std::io::stdout().flush();

            let s_path = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if s_path.trim().is_empty() {
                println!("Empty path.");
                continue;
            }

            match tx.send(TrainingDataEventAscii::add_training_file_json_array { s_path: s_path.clone() }) {
                Ok(()) => println!("Queued training data file for ingestion: {}", s_path),
                Err(_) => println!("Failed to send ingestion event (receiver not alive)."),
            }

            continue;
        }

        if s_cmd_lc == "w" {
            print!("Enter checkpoint path or press Enter for default: ");
            let _ = std::io::stdout().flush();

            let s_path = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if !s_path.is_empty() {
                s_checkpoint_path = s_path;
            }

            let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
            match llm.save_checkpoint_llm_checkpoint_v2(&s_checkpoint_path) {
                Ok(()) => println!("Saved checkpoint: {}", s_checkpoint_path),
                Err(e) => println!("Save failed: {}", e),
            }

            continue;
        }

        if s_cmd_lc == "l" {
            print!("Enter checkpoint path or press Enter for default: ");
            let _ = std::io::stdout().flush();

            let s_path = match read_line_ascii_trimmed() {
                Ok(s) => s,
                Err(e) => {
                    println!("Input error: {}", e);
                    continue;
                }
            };

            if !s_path.is_empty() {
                s_checkpoint_path = s_path;
            }

            match Llm::load_checkpoint_llm_checkpoint_v2_rebuild(&s_checkpoint_path) {
                Ok(llm_loaded) => {
                    let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
                    *llm = llm_loaded;
                    println!("Loaded checkpoint into serve model: {}", s_checkpoint_path);
                }
                Err(e) => println!("Load failed: {}", e),
            }

            continue;
        }

        if s_cmd_lc == "o" {
            let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
            let b_new = !llm.is_outage_simulation_enabled();
            llm.set_outage_simulation_enabled(b_new);
            println!(
                "Outage simulation: {}",
                if b_new { "enabled" } else { "disabled" }
            );
            continue;
        }

        if s_cmd_lc == "y" {
            let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
            let v_lines = topology_to_ascii_lines(&mut llm);
            println!();
            for s_line in v_lines {
                println!("{}", s_line);
            }
            continue;
        }

        if s_cmd_lc == "x" {
            let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
            print_metrics_ascii(&mut llm);
            continue;
        }

        if s_cmd_lc == "a" {
            println!("Interactive mode. Type 'done' to exit.");
            loop {
                // While in interactive mode, keep snapshots applied.
                drain_snapshot_updates_non_blocking(&mut opt_snapshot_rx, &llm_serve);

                print!("Enter prompt: ");
                let _ = std::io::stdout().flush();

                let s_user = match read_line_ascii_trimmed() {
                    Ok(s) => s,
                    Err(e) => {
                        println!("Input error: {}", e);
                        continue;
                    }
                };

                if s_user.is_empty() {
                    println!("Empty prompt.");
                    continue;
                }
                if s_user.eq_ignore_ascii_case("done") {
                    break;
                }

                let s_formatted = format!("User: {}", s_user);

                let t0 = Instant::now();
                let r_predict = {
                    let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
                    llm.predict_with_stats(&s_formatted)
                };
                let d_ms = t0.elapsed().as_secs_f64() * 1000.0;

                match r_predict {
                    Ok((s_out, st)) => {
                        println!("Model output: {}", s_out);
                        let llm = llm_serve.lock().expect("llm_mutex_poisoned");
                        let m =
                            compute_predict_metrics_ascii(&llm, &s_formatted, &s_out, d_ms, Some(&st));
                        print_predict_metrics_ascii(&m);
                    }
                    Err(e) => {
                        println!("Model output error: {}", e);
                        let llm = llm_serve.lock().expect("llm_mutex_poisoned");
                        let m = compute_predict_metrics_ascii(&llm, &s_formatted, "", d_ms, None);
                        print_predict_metrics_ascii(&m);
                    }
                }
            }
            continue;
        }

        println!("Unknown command.");
    }
}
