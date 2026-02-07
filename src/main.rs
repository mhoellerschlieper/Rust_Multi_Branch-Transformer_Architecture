// main.rs
// Description: Binary entry point with menu loop. Builds model from tokenizer,
//              supports checkpoint save and load.
// History:
// - 2026-02-01: Add menu loop and checkpoint save and load.
// - 2026-02-01: Fix checkpoint load by rebuilding model from checkpoint tokenizer vocab.
// - 2026-02-07: Add MTB parallel block group layer to support multi branch topology.
// - 2026-02-07: Add TransformerSequence and generalize ParallelBlockGroup to accept Layer branches.
// Author: Marcus Schlieper

mod layer;
mod math;
mod tokenizer;
mod train;
mod utils;

use std::io::Write;

use crate::layer::{
    Embeddings, Llm, OutputProjection, TransformerBlock, ParallelBlockGroup, TransformerSequence, Layer
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

    let _ = llm.set_sampling_config(0.9, 40, 0.95, 987654321);
    llm
}

// - Ask questions repeatedly
// - Stop when user types "done" (case-insensitive)
// - Store Q/A pairs and print a short summary at the end
fn run_interview_until_done_ascii() -> Result<(), String> {
    let mut v_qa: Vec<(String, String)> = Vec::new();
    let mut i_turn: usize = 0;

    println!("");
    println!("=== Interview Mode ===");
    println!("Enter done to finish.");
    println!("");

    loop {
        i_turn = i_turn.saturating_add(1);

        // Single question per message, repeated until done.
        // The question is intentionally stable and ASCII only.
        let s_question = format!(
            "Question {}: Describe the required core functionality and constraints:",
            i_turn
        );

        println!("{}", s_question);
        print!("Answer: ");
        let _ = std::io::stdout().flush();

        let s_answer = read_line_ascii_trimmed()?;
        if s_answer.to_lowercase() == "done" {
            break;
        }

        if s_answer.is_empty() {
            println!("Answer empty. Please provide text or enter done.");
            continue;
        }

        v_qa.push((s_question, s_answer));
        println!("");
    }

    println!("");
    println!("=== Interview Summary ===");
    println!("Collected items: {}", v_qa.len());
    for (i, (q, a)) in v_qa.iter().enumerate() {
        println!("");
        println!("Item {}:", i + 1);
        println!("Q: {}", q);
        println!("A: {}", a);
    }

    Ok(())
}


fn main() {
    let s_prompt = String::from("User: How do mountains form?");

    let mut s_checkpoint_path: String = "../../checkpoints/llm_checkpoint.json".to_string();

    let dataset = Dataset::new(
        "../../data/data_to_pretrain.json",
        "../../data/data_to_train.json",
        DatasetType::JSON,
    );

    // NOTE: For now, keep initial tokenizer training to allow immediate usage.
    // The important fix is that loading rebuilds the model to match checkpoint vocab.
    let mut v_corpus: Vec<String> = Vec::new();
    v_corpus.extend(dataset.pretraining_data.clone());
    v_corpus.extend(dataset.chat_training_data.clone());

    let mut config = BpeTokenizerConfig::default();
    config.i_vocab_target = 2000;
    config.i_min_pair_count = 2;
    // config.u64_seed can be changed for experimentation, but remains stored in checkpoint.

    let bpe = match BpeTokenizer::train_from_corpus_with_config(&v_corpus, config) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("Tokenizer training failed: {}", e);
            return;
        }
    };

    let mut llm = build_llm_from_tokenizer(bpe);

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!(
        "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("Total parameters: {}", llm.total_parameters());

    

    loop {
        println!("\n--- Menu Mode ---");
        println!("Commands:");
        println!("  t Train");
        println!("  l Load checkpoint");
        println!("  s Save checkpoint");
        println!("  a Ask");
        println!("  e Exit");
        print!("\nEnter command: ");
        let _ = std::io::stdout().flush();

        let s_cmd = match read_line_ascii_trimmed() {
            Ok(s) => s.to_lowercase(),
            Err(e) => {
                println!("Input error: {}", e);
                continue;
            }
        };

        if s_cmd == "e" {
            println!("Exit.");
            break;
        }

        if s_cmd == "t" {
            let v_pretraining_examples: Vec<&str> = dataset
                .pretraining_data
                .iter()
                .map(|s| s.as_str())
                .collect();

            let v_chat_training_examples: Vec<&str> = dataset
                .chat_training_data
                .iter()
                .map(|s| s.as_str())
                .collect();

            println!("\n=== PRE-TRAINING MODEL ===");
            println!(
                "Pre-training on {} examples for {} epochs with learning rate {}",
                dataset.pretraining_data.len(),
                100,
                0.0005
            );

            if let Err(e) = llm.train(v_pretraining_examples, 30, 0.0005) {
                eprintln!("Training failed: {}", e);
                continue;
            }

            println!("\n=== INSTRUCTION TUNING ===");
            println!(
                "Instruction tuning on {} examples for {} epochs with learning rate {}",
                dataset.chat_training_data.len(),
                200,
                0.0001
            );

            if let Err(e) = llm.train(v_chat_training_examples, 50, 0.0001) {
                eprintln!("Training failed: {}", e);
                continue;
            }

            continue;
        }

        if s_cmd == "s" {
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

            match llm.save_checkpoint(&s_checkpoint_path) {
                Ok(()) => println!("Saved checkpoint: {}", s_checkpoint_path),
                Err(e) => println!("Save failed: {}", e),
            }

            continue;
        }

        if s_cmd == "l" {
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

            // IMPORTANT: Rebuild model to match checkpoint tokenizer and vocab size.
            match Llm::load_checkpoint_rebuild(&s_checkpoint_path) {
                Ok(llm_loaded) => {
                    llm = llm_loaded;
                    println!("Loaded checkpoint: {}", s_checkpoint_path);
                }
                Err(e) => println!("Load failed: {}", e),
            }

            continue;
        }

        if s_cmd == "a" {
             println!("Interactive mode. Type 'done' to exit.");
             loop {
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
                match llm.predict(&s_formatted) {
                    Ok(s_out) => println!("Model output: {}", s_out),
                    Err(e) => println!("Model output error: {}", e),
                }
             }

            continue;
        }

        println!("Unknown command.");
    }
}
