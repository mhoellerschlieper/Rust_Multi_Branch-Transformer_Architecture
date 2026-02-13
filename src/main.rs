// main.rs
// Description: Binary entry point with menu loop. Builds model from tokenizer,
//              supports checkpoint save and load.
//
//              Extension:
//              - Background training thread with live metrics
//              - Parallel "ask" during training via separate serving model instance
//              - Continuous learning: partial path availability and incremental updates
//
// History:
// - 2026-02-01: Add menu loop and checkpoint save and load.
// - 2026-02-07: Add MTB parallel block group layer to support multi branch topology.
// - 2026-02-08: Add predict_with_stats and post predict metrics.
// - 2026-02-08: Add command z to compute mean metrics with outage simulation off and on.
// - 2026-02-13: Add background training, live metrics, cooperative stop.
// - 2026-02-13: Add serving model with snapshot updates for true parallel ask during training.
// Author: Marcus Schlieper

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
    ContinuousLearningConfig, Embeddings, Layer, Llm, OutputProjection, ParallelBlockGroup, PredictStats,
    TrainingProgressEventAscii, TransformerBlock, TransformerSequence,
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

    let seq_2_1 = TransformerSequence::new(vec![block2_1, block2_2])
        .expect("transformer_sequence_new_failed");
    let seq_2_2 = TransformerSequence::new(vec![block2_3, block2_4])
        .expect("transformer_sequence_new_failed");
    let seq_2_3 = TransformerSequence::new(vec![block2_5, block2_6])
        .expect("transformer_sequence_new_failed");
    let seq_2_4 = TransformerSequence::new(vec![block2_7, block2_8])
        .expect("transformer_sequence_new_failed");

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
        }
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
    if !m.s_last_error.is_empty() {
        println!("last_error: {}", m.s_last_error);
    }
}

fn drain_training_progress_non_blocking(
    opt_rx: &mut Option<mpsc::Receiver<TrainingProgressEventAscii>>,
    metrics_shared: &Arc<Mutex<training_metrics_snapshot_ascii>>,
    b_cancel_train: &Arc<AtomicBool>,
) {
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
        m.b_cancel_requested = b_cancel_train.load(Ordering::SeqCst);

        m.s_phase = ev.s_phase;
        m.i_epoch_current = ev.i_epoch_current;
        m.i_epochs_total = ev.i_epochs_total;
        m.d_last_epoch_loss = ev.d_last_epoch_loss;
        m.d_last_step_loss = ev.d_last_step_loss;
        m.i_rows_used_last_epoch = ev.i_rows_used_last_epoch;
        m.i_total_steps = ev.i_total_steps;
    }
}

fn drain_snapshot_updates_non_blocking(
    opt_rx: &mut Option<mpsc::Receiver<Vec<f32>>>,
    llm_serve: &Arc<Mutex<Llm>>,
) {
    let rx = match opt_rx.as_mut() {
        Some(r) => r,
        None => return,
    };

    // Always keep only the latest snapshot to minimize work.
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

    // Training thread handle.
    let mut opt_train_handle: Option<std::thread::JoinHandle<()>> = None;

    {
        let mut llm = llm_serve.lock().expect("llm_mutex_poisoned");
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
        println!("  l Load checkpoint (serve model)");
        println!("  w Save checkpoint (serve model)");
        println!("  a Ask (serve model, parallel to training)");
        println!("  o Toggle outage simulation (serve model, test only)");
        println!("  y Topology (ASCII, serve model)");
        println!("  x Metrics (MTB diagnostics, serve model)");
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
            if let Some(h) = opt_train_handle.take() {
                b_cancel_train.store(true, Ordering::SeqCst);
                let _ = h.join();
            }
            println!("Exit.");
            break;
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

            let llm_for_train = Arc::clone(&llm_train);
            let metrics_for_train = Arc::clone(&metrics_shared);
            let cancel_for_train = Arc::clone(&b_cancel_train);

            let v_pretraining_examples: Vec<String> = dataset.pretraining_data.clone();
            let v_chat_training_examples: Vec<String> = dataset.chat_training_data.clone();

            opt_train_handle = Some(thread::spawn(move || {
                let r_run = (|| -> Result<(), String> {
                    let i_snapshot_every_steps: usize = 200;

                    // Phase 1: pretraining.
                    {
                        let mut m = metrics_for_train
                            .lock()
                            .map_err(|_| "metrics_lock_failed".to_string())?;
                        m.s_phase = "pretraining".to_string();
                        m.i_epoch_current = 0;
                        m.i_epochs_total = 30;
                        m.s_last_error = "".to_string();
                    }

                    {
                        let mut llm = llm_for_train.lock().map_err(|_| "llm_lock_failed".to_string())?;

                        // Continuous learning config for a 4-branch ParallelBlockGroup.
                        // p_i is per-branch availability; min participation stabilizes behavior.
                        let cl_cfg = ContinuousLearningConfig {
                            v_branch_participation_p: vec![0.75, 0.75, 0.75, 0.75],
                            i_min_active_branches: 2,
                            b_scale_by_inverse_participation: true,
                            u64_mask_seed: 20260213,
                        };

                        llm.train_with_progress_continuous_learning_ascii(
                            v_pretraining_examples.iter().map(|s| s.as_str()).collect(),
                            30,
                            0.0005,
                            Arc::clone(&cancel_for_train),
                            tx_progress.clone(),
                            "pretraining",
                            Some(cl_cfg),
                            i_snapshot_every_steps,
                            Some(tx_snapshot.clone()),
                        )?;
                    }

                    if cancel_for_train.load(Ordering::SeqCst) {
                        return Ok(());
                    }

                    // Phase 2: instruction tuning.
                    {
                        let mut m = metrics_for_train
                            .lock()
                            .map_err(|_| "metrics_lock_failed".to_string())?;
                        m.s_phase = "instruction_tuning".to_string();
                        m.i_epoch_current = 0;
                        m.i_epochs_total = 50;
                        m.s_last_error = "".to_string();
                    }

                    {
                        let mut llm = llm_for_train.lock().map_err(|_| "llm_lock_failed".to_string())?;

                        let cl_cfg = ContinuousLearningConfig {
                            v_branch_participation_p: vec![0.60, 0.70, 0.80, 0.65],
                            i_min_active_branches: 2,
                            b_scale_by_inverse_participation: true,
                            u64_mask_seed: 20260214,
                        };

                        llm.train_with_progress_continuous_learning_ascii(
                            v_chat_training_examples.iter().map(|s| s.as_str()).collect(),
                            50,
                            0.0001,
                            Arc::clone(&cancel_for_train),
                            tx_progress.clone(),
                            "instruction_tuning",
                            Some(cl_cfg),
                            i_snapshot_every_steps,
                            Some(tx_snapshot.clone()),
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
            let m = metrics_shared.lock().expect("metrics_mutex_poisoned").clone();
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
                continue;
            }

            b_cancel_train.store(true, Ordering::SeqCst);
            {
                let mut m = metrics_shared.lock().expect("metrics_mutex_poisoned");
                m.b_cancel_requested = true;
                m.s_phase = "cancel_requested".to_string();
            }

            if let Some(h) = opt_train_handle.take() {
                let _ = h.join();
            }

            println!("Training stop requested and thread joined.");
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
                        let m = compute_predict_metrics_ascii(&llm, &s_formatted, &s_out, d_ms, Some(&st));
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
