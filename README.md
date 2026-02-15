# Multi-Branch-Transformer (MBT) Architecture

Implementation and reference documentation of a **Multi-Branch Transformer Architecture (MBT)** in **Rust**. The focus is on **intra-layer parallelism (width)** with **explicit aggregation**, complemented by a **BPE tokenizer pipeline**, **training/inference**, and **reproducible checkpoints** (tokenizer + parameters) as a closed, analyzable system. The architecture is described to serve as a foundation for **distributed execution** (including P2P topologies) as well as for **fault-tolerant and continuously extensible** transformer systems.

<img width="392" height="995" alt="grafik" src="https://github.com/user-attachments/assets/bc7ae7ee-03fd-4391-addf-42393690a981" />



## Contents
- [Motivation and Objectives](#motivation-and-objectives)
- [Core Idea: Multi-Branch Transformer (MBT)](#core-idea-multi-branch-transformer-mbt)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Build and Run](#build-and-run)
- [Usage (CLI)](#usage-cli)
- [Training](#training)
- [Inference](#inference)
- [Checkpoints and Reproducibility](#checkpoints-and-reproducibility)
- [Benchmark with and without outage simulation](#Benchmark)
- [Metrics](#Metrics)
- [Topology(Example)](#Topology)
- [Distributed Execution and Fault Tolerance (Concept)](#distributed-execution-and-fault-tolerance-concept)
- [Security and Robustness (System Perspective)](#security-and-robustness-system-perspective)
- [Distinction from MoE / Switch / Multi-Path](#distinction-from-moe--switch--multi-path)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [References (APA)](#references-apa)
- [License](#license)
- [Contact](#contact)

<img width="1090" height="713" alt="grafik" src="https://github.com/user-attachments/assets/0e7718d4-0aed-4685-b4bf-e8694df42a1a" />


In real inference deployments, large transformer models are often not primarily limited by compute operations, but by **memory footprint**, **memory bandwidth** and **communication and synchronization costs** in distributed environments. Classical partitioning along **depth** does reduce memory requirements per node, but it still enforces a **sequential token-processing chain**, leaving potential parallelism gains structurally underutilized&mdash;particularly in **heterogeneous** and **volatile** execution environments.

This repository addresses this situation via a **multi-branch topology** in which parallelism is organized as a **width structure within a layer**: multiple transformer blocks or block sequences are executed **concurrently**, and their path outputs are subsequently fused through an **aggregation stage**, which simultaneously provides a natural **partitioning and orchestration unit** for distributed resources.

---

## Core Idea: Multi-Branch Transformer (MBT)

<img width="1226" height="120" alt="grafik" src="https://github.com/user-attachments/assets/811bf50f-780e-4d40-abf4-2083bc8ae2fb" />


This aggregation functions as a central system component because it structurally enables **fusion**, **weighting**, **failure handling** (masking/renormalization), and **governance rules** against path impoverishment and weight collapse.

---
Result Metrics:

<img width="1228" height="745" alt="grafik" src="https://github.com/user-attachments/assets/2d224c94-1aac-449e-804b-0a957b6774b9" />

---

## Features

- **Multi-Branch Transformer Layer** (width parallelism with aggregation)
- **BPE tokenizer** with persisted configuration for deterministic reconstruction
- **Training loop** for autoregressive next-token training (pretraining and instruction tuning as variants)
- **Inference** (greedy decoding; depending on status optionally temperature / top-k / top-p)
- **Checkpointing**: saving and loading tokenizer + model parameters
- **Load with Rebuild**: reconstructing the model based on the vocabulary size stored in the checkpoint to prevent shape mismatches
- **Robustness mechanisms** (including validations, parameter-length checks, atomic writes; depending on implementation status)

---

## Architecture Overview

The codebase follows a &ldquo;self-contained&rdquo; approach in which tokenization, model, training, inference, and persistence are integrated into a traceable workflow. Typical components include:

- **Tokenizer**: BPE training, encode/decode, persisted tokenizer configuration
- **Model core**: embeddings, self-attention, feed-forward, layer norm, transformer blocks
- **MBT extension**: parallel branches per layer and aggregation logic
- **Optimization**: e.g., Adam; optional gradient clipping
- **Persistence**: checkpoint format with versioning/magic value and atomic writing

Note: Concrete module and file names depend on the current state of the repository; the README describes the target design consistently with the provided textual foundations.

---

## Installation

Requirements:
- Rust (stable)
- Cargo

Optional (recommended for development):
- `rustfmt`
- `clippy`

---

## Build and Run

Build (release):
- `cargo build --release`

Run:
- `cargo run --release`

---

## Usage (CLI)

The project is typically CLI-oriented and provides (depending on status) the following flows:
- start training (pretraining / instruction tuning)
- save checkpoint
- load checkpoint (including rebuild)
- enter a prompt and generate a response

Concrete commands, flags, and menu entries should be checked in `main.rs` or the respective CLI definition.

---

## Training

Training follows the autoregressive next-token scheme: input tokens and target tokens are created by shifting the sequence by 1, the loss is computed via cross-entropy, and gradients are propagated by backpropagation. In the MBT variant, it is additionally relevant that the **width paths** are trained fairly and stably to prevent **path impoverishment**, because otherwise redundant paths do not provide functional substitutability in the event of failure.

Depending on configuration, the following aspects are central:
- sequence-length limitation (e.g., `MAX_SEQ_LEN`)
- (optional) gradient clipping to stabilize small, non-optimized implementations (Pascanu et al., 2013)
- (optional) mini-batching/gradient accumulation (roadmap, if not yet implemented)

<img width="1121" height="1117" alt="grafik" src="https://github.com/user-attachments/assets/df02f74c-dafe-48ca-a839-e480d31d2a1b" />

---

## Inference

Inference in the simplest mode uses **greedy decoding**: the most likely next token is iteratively selected until EOS is reached or the maximum sequence length applies. Sampling methods such as temperature / top-k / top-p may be added or already present depending on implementation status; in the literature they are considered practically relevant for text quality (Holtzman et al., 2020).

---

## Checkpoints and Reproducibility

### Why &ldquo;Load with Rebuild&rdquo; Is Required

The output projection typically has shape \([d_{\text{emb}}, |V|]\), where \(|V|\) depends directly on the tokenizer vocabulary size. If a tokenizer with a different vocabulary size is used when loading, shape mismatches occur.

Therefore, when loading, the system (conceptually) implements the following steps:
1. load and validate checkpoint (magic/version)
2. reconstruct tokenizer from checkpoint
3. **rebuild** the model based on \(|V|\) from the checkpoint
4. load the parameter vector and check length/shape

### Atomic Writes

When saving, an atomic write strategy (temporary file + rename) is used to avoid inconsistent checkpoints in the event of interruption or system faults.

---
## Topology
<img width="508" height="256" alt="grafik" src="https://github.com/user-attachments/assets/c478ee5b-5f2d-43aa-b994-b6246ac1fd5f" />

---
## Benchmark 
Benchmark with and without outage simulation
<img width="738" height="846" alt="grafik" src="https://github.com/user-attachments/assets/f2ab349c-0ca2-413d-9b10-3e9a1f47b100" />

---

## Distributed Execution and Fault Tolerance (Concept)

The multi-branch structure is modeled such that **one path** can serve as a **partition unit** mapped to different nodes, while an aggregation instance fuses the path outputs. For fault tolerance, a masking variable \(m_i^{(l)} \in \{0,1\}\) is used; weights are renormalized in the event of failure:

\[
\tilde{\alpha}_i^{(l)} = \frac{m_i^{(l)} \alpha_i^{(l)}}{\sum_{j=1}^{K} m_j^{(l)} \alpha_j^{(l)}} \quad (\text{if denominator} &gt; 0),
\quad
\tilde{h}^{(l+1)} = \sum_{i=1}^{K} \tilde{\alpha}_i^{(l)} z_i^{(l)}.
\]

Thus, the layer function remains well-defined as long as at least one path is available. In P2P settings, quorum/timeout policies and anti-weight-collapse rules are methodologically central to limit tail latency and single-point-of-failure effects.

---
## Result Metrics
The repository exposes two complementary metric families: 
(i) MTB diagnostics that characterize a ParallelBlockGroup as a width-ensemble object, and 
(ii) continuous learning and operations metrics that quantify online ingestion, mask sparsity, replay usage, retention stability, and expansion-induced drift. 
These metrics are deliberately designed to connect implementation behavior with system-level goals such as fault tolerance, non-collapse of width, and continuous extensibility.

### A) MTB Diagnostics (CLI: x)
MTB diagnostics estimate whether width actually behaves as a set of substitutable or complementary paths, rather than collapsing into a single dominant route.

**path_starvation_index (derived from normalized entropy of branch-selection probabilities)**
Interpretation: values near 0 indicate relatively uniform usage; values near 1 indicate strong concentration and potential starvation.
Practical heuristic: > 0.60 indicates severe concentration and a high risk that unused paths fail to learn useful functions.
effective_num_paths ≈ exp(H)
Interpretation: the entropy-derived “effective” count of meaningfully participating paths, bounded by 1..K.
Practical heuristic: values < 2.0 in a multi-branch layer suggest functional collapse into near-single-path behavior.

**gini_concentration and top1_share**
Interpretation: alternative concentration measures; increasing values indicate dominance and impoverishment of width.
Practical heuristic: top1_share > 0.70 indicates strong dominance; mitigation may require fairness controls, replay, or expansion.

### diversity_cosine_distance_mean and branch_correlation_mean
Interpretation: measure geometric similarity of branch outputs; low diversity (high correlation) suggests redundant branches.
Practical heuristic: sustained high correlation indicates that width may not contribute to robustness under outages, because branch outputs are not functionally distinct.

### margin_top1_top2
Interpretation: stability of the internal branch scoring distribution; high margins can indicate deterministic routing pressure toward a single path.

### output_energy_cv
Interpretation: coefficient of variation of output energy across branches; extreme variability can indicate unstable scaling or mismatch in branch capacity.

Taken together, these metrics provide a measurable proxy for whether MBT is achieving the intended “parallel width with substitutability” property, rather than degenerating into an expensive single-route architecture.

## B) Training and Continuous Learning Metrics (CLI: b)
When background training is enabled, the system emits structured progress snapshots that aim to make “continuous, partial-availability learning” operationally observable.

### B.1 Ingestion Throughput and Data Health
**ingest_rows_per_sec_window, ingest_events_per_sec_window**
Interpretation: whether online ingestion is progressing; persistent zeros under active ingestion requests indicate stalled drains or missing receiver wiring.

**ingest_parse_errors_total, ingest_rows_rejected_total**
Interpretation: pipeline correctness and data quality; elevated rejection ratios indicate that “continuous learning” may be limited by input quality rather than model capacity.

**ingest_pending_events_observed_peak**
Interpretation: a coarse backlog indicator; sustained growth suggests backpressure and delayed adaptation.

### B.2 Coverage (Effective Use of Available Data)
**coverage_ratio_used_over_available, new_data_ratio_in_available**
Interpretation: whether the epoch uses most available rows and how non-stationary the stream is; low coverage can indicate skip-pathologies or frequent invalid rows.
### B.3 Availability Masks and Participation
**active_branches_mean/std/min/max, mask_sparsity_mean/std, steps_at_min_active_share**
Interpretation: whether training operates in a sparse regime (high variance but lower compute) or a dense regime (lower variance but higher cost).
For MBT specifically, high sparsity increases the importance of unbiasedness controls (inverse participation scaling) and replay.
### B.4 Inverse Participation Scaling (Unbiasedness Proxy)
**grad_norm_ratio_scaled_over_unscaled_mean/std**
Interpretation: whether inverse participation scaling strongly amplifies gradients; large amplification can require lower learning rates or stronger clipping to preserve stability.
### B.5 Replay and Retention
**replay_share, replay_delta_loss_mean/std**
Interpretation: whether replay is used at meaningful rates and whether replay examples are becoming “harder” (potential drift away from older knowledge) or “too easy” (possible redundancy).

**loss_control_old/new, retention_delta_old/new**
Interpretation: a forgetting proxy on deterministic control slices; persistent positive deltas indicate degradation of prior behavior and motivate increased replay, lower learning rates, or stricter governance.

### B.6 Fairness of Width Utilization
**branch_select_gini, branch_select_top1_share**
Interpretation: whether EMA-based selection leads to dominance; high concentration implies that width capacity is not used effectively and that robustness under branch failure is compromised.
### B.7 Expansion and Drift (Continuous Extensibility)
**expansion_events_total, eta_injection_last, sum_w_new_last**
Interpretation: whether the system expands width and how aggressively it injects new branches into the aggregation; aggressive injection increases the probability of behavioral drift.

**expand_drift_logits_l2_mean/std, expand_drift_logits_cos_dist_mean/std**
Interpretation: functional continuity under expansion; large drift suggests that expansion changes the model’s effective function too abruptly and should be mitigated by lower injection rates or stricter expansion triggers.


---
## Security and Robustness (System Perspective)

In open or semi-open distributed environments, parallel paths increase the attack surface (Byzantine outputs, straggling/DoS, update poisoning). An MBT system therefore typically requires:
- integrity of model artifacts (hashes/signatures/versioning)
- quorum/timeout policies against straggler tail latency
- norm and weight controls in the aggregation
- (optional) admission control for new paths under &ldquo;continuous expandable width&rdquo;

A blockchain can conceptually serve as a governance and audit layer, but it is not intended as the model&rsquo;s execution environment; rather, it acts as a root of trust for identity, artifact hashes, and update approvals.

---

## Distinction from MoE / Switch / Multi-Path

- **MoE/Switch**: width is primarily realized via **sparse, token-wise routing** to a small number of experts; the goal is parameter scaling with limited compute per token (Shazeer et al., 2017; Fedus et al., 2022).
- **Multi-Path (residual interpretation)**: describes multi-path behavior more analytically than as an explicit, orchestratable parallel structure (Veit et al., 2016).
- **MBT**: defines multi-pathness as **simultaneously active paths per layer** with **explicit aggregation**, thereby systematically addressing distributability, robustness, and continuous extensibility.

---

## Roadmap

Possible next steps (depending on current status):
- **Efficient inference**: KV cache, batching, masking, mixed precision
- **Distribution runtime**: branch discovery, scheduling, quorum-based aggregation, straggler management
- **Robust aggregation**: trimmed mean / median-of-means, reputation weights, anti-impoverishment governance
- **Continuous learning governance**: update validation, rollback, poisoning detection
- **Tests**: tokenizer determinism, softmax stability, checkpoint round-trip, golden tests

---

## Citation

If content from the project is cited, a reference to this repository and to the sources mentioned in the project context is recommended (see below).

---

## References (APA)

Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Le, Q. V., Mao, M. Z., Ranzato, M., Senior, A., Tucker, P., Yang, K., &amp; Ng, A. Y. (2012). Large scale distributed deep networks. In *Advances in Neural Information Processing Systems*.

Fedus, W., Zoph, B., &amp; Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research, 23*(120), 1&ndash;39.

Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). *Deep learning*. MIT Press.

Holtzman, A., Buys, J., Du, L., Forbes, M., &amp; Choi, Y. (2020). The curious case of neural text degeneration. In *International Conference on Learning Representations*.

Pascanu, R., Mikolov, T., &amp; Bengio, Y. (2013). On the difficulty of training recurrent neural networks. In *International Conference on Machine Learning*.

Schlieper, M. (2026): Multi-Branch Transformer (MBT): Distributed Transformer Blocks and Topologies in Large Language Models as a Deep-Width-Learning Approach

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., &amp; Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*.

Veit, A., Wilber, M. J., &amp; Belongie, S. (2016). Residual networks behave like ensembles of relatively shallow networks. In *Advances in Neural Information Processing Systems*.

---

## License

See `LICENSE` in the repository.

---

## Contact

- Contact: mschlieper@expchat.ai
- Related implementations/references (project environment):
  - Rust Distributed GPT Node: https://github.com/mhoellerschlieper/Rust-Distributed-GPT-Node
  - LLM Rust: https://github.com/mhoellerschlieper/LLM_Rust



















