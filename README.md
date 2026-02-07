# Rust_Multi_Branch-Transformer_Architecture

Implementierung und Referenzdokumentation einer **Multi-Branch-Transformer-Architektur (MBT)** in **Rust**. Der Schwerpunkt liegt auf **intra-layer Parallelität (Width)** mit **expliziter Aggregation**, ergänzt um eine **BPE-Tokenizer-Pipeline**, **Training/Inference** sowie **reproduzierbare Checkpoints** (Tokenizer + Parameter) als geschlossenes, analysierbares System. Die Architektur ist so beschrieben, dass sie als Grundlage für **verteilte Ausführung** (u. a. P2P-Topologien) sowie für **fehlertolerante und kontinuierlich erweiterbare** Transformer-Systeme dient.

## Inhalt
- [Motivation und Zielsetzung](#motivation-und-zielsetzung)
- [Kernidee: Multi-Branch Transformer (MBT)](#kernidee-multi-branch-transformer-mbt)
- [Features](#features)
- [Architekturüberblick](#architekturüberblick)
- [Installation](#installation)
- [Build und Run](#build-und-run)
- [Nutzung (CLI)](#nutzung-cli)
- [Training](#training)
- [Inference](#inference)
- [Checkpoints und Reproduzierbarkeit](#checkpoints-und-reproduzierbarkeit)
- [Verteilte Ausführung und Fault Tolerance (Konzept)](#verteilte-ausführung-und-fault-tolerance-konzept)
- [Sicherheit und Robustheit (Systemperspektive)](#sicherheit-und-robustheit-systemperspektive)
- [Abgrenzung zu MoE / Switch / Multi-Path](#abgrenzung-zu-moe--switch--multi-path)
- [Roadmap](#roadmap)
- [Zitation](#zitation)
- [Quellen (APA)](#quellen-apa)
- [Lizenz](#lizenz)
- [Kontakt](#kontakt)

---

## Motivation und Zielsetzung

Große Transformer-Modelle sind in realen Inferenz-Deployments häufig nicht primär durch Rechenoperationen limitiert, sondern durch **Speicherbedarf**, **Speicherbandbreite**, **KV-Cache-Management** sowie **Kommunikations- und Synchronisationskosten** in verteilten Umgebungen. Klassische Partitionierungen entlang der **Tiefe** reduzieren zwar den Speicherbedarf pro Node, erzwingen jedoch weiterhin eine **sequenzielle Token-Verarbeitungskette** und lassen damit potenzielle Parallelitätsgewinne strukturell ungenutzt, insbesondere in **heterogenen** und **volatilen** Ausführungsumgebungen.

Das Repository adressiert diese Lage durch eine **Multi-Branch-Topologie**, bei der Parallelität als **Breitenstruktur innerhalb eines Layers** organisiert wird: Mehrere Transformer-Blöcke bzw. Block-Sequenzen werden **zeitgleich** ausgeführt und ihre Pfadausgaben werden anschließend über eine **Aggregationsstufe** fusioniert, was zugleich eine natürliche **Partitionierungs- und Orchestrierungseinheit** für verteilte Ressourcen bereitstellt.

---

## Kernidee: Multi-Branch Transformer (MBT)

Ein MBT-Layer enthält eine Menge paralleler Pfade \( \{TB_{l,1}, \dots, TB_{l,K}\} \), die denselben Layer-Input verarbeiten und Pfadausgaben \( z_i^{(l)} \) erzeugen; die Layerausgabe entsteht als gewichtete Aggregation

\[
h^{(l+1)} = \sum_{i=1}^{K} \alpha_i^{(l)} \, z_i^{(l)}, \quad \alpha_i^{(l)} \ge 0, \quad \sum_{i=1}^{K} \alpha_i^{(l)} = 1.
\]

Diese Aggregation fungiert als zentrale Systemkomponente, weil sie zugleich **Fusion**, **Gewichtung**, **Ausfallbehandlung** (Maskierung/Renormalisierung) sowie **Governance-Regeln** gegen Pfad-Verarmung und Gewichtskollaps strukturell ermöglicht.

---

## Features

- **Multi-Branch Transformer Layer** (Width-Parallelität mit Aggregation)
- **BPE Tokenizer** mit persistierter Konfiguration für deterministische Rekonstruktion
- **Training Loop** für autoregressives Next-Token-Training (Pretraining und Instruction-Tuning als Varianten)
- **Inference** (greedy decoding; je nach Stand optional Temperature / top-k / top-p)
- **Checkpointing**: Speichern und Laden von Tokenizer + Modellparametern
- **Load with Rebuild**: Rekonstruktion des Modells anhand der im Checkpoint gespeicherten Vokabulargröße zur Vermeidung von Shape-Mismatches
- **Robustheitsmechanismen** (u. a. Validierungen, Parameterlängenprüfungen, atomare Writes; abhängig vom Implementationsstand)

---

## Architekturüberblick

Die Codebasis folgt einem „self-contained“-Ansatz, bei dem Tokenisierung, Modell, Training, Inferenz und Persistenz in einem nachvollziehbaren Workflow integriert sind. Typische Komponenten sind:

- **Tokenizer**: BPE-Training, Encode/Decode, persistierte Tokenizer-Konfiguration
- **Model Core**: Embeddings, Self-Attention, Feed-Forward, LayerNorm, Transformer-Blöcke
- **MBT-Erweiterung**: parallele Branches pro Layer und Aggregationslogik
- **Optimierung**: z. B. Adam; optional Gradient Clipping
- **Persistenz**: Checkpoint-Format mit Versionierung/Magic-Value und atomarem Schreiben

Hinweis: Die konkreten Modul- und Dateinamen sind abhängig vom aktuellen Repository-Stand; die README beschreibt das Zielbild konsistent mit den bereitgestellten Textgrundlagen.

---

## Installation

Voraussetzungen:
- Rust (stable)
- Cargo

Optional (empfohlen für Entwicklung):
- `rustfmt`
- `clippy`

---

## Build und Run

Build (Release):
- `cargo build --release`

Run:
- `cargo run --release`

---

## Nutzung (CLI)

Das Projekt ist typischerweise CLI-orientiert und stellt (je nach Stand) folgende Flows bereit:
- Training starten (Pretraining / Instruction-Tuning)
- Checkpoint speichern
- Checkpoint laden (inkl. Rebuild)
- Prompt eingeben und Antwort generieren

Die konkreten Kommandos, Flags und Menüeinträge sind in `main.rs` bzw. der jeweiligen CLI-Definition zu prüfen.

---

## Training

Das Training folgt dem autoregressiven Next-Token-Schema: Eingabetokens und Zieltokens entstehen durch eine um 1 verschobene Sequenz, der Loss wird über Cross-Entropy berechnet, und die Gradienten werden per Backpropagation propagiert. In der MBT-Variante ist zusätzlich relevant, dass die **Breitenpfade** fair und stabil trainiert werden, um **Pfad-Verarmung** zu vermeiden, weil andernfalls redundante Pfade im Ausfallfall keine funktionale Ersatzfähigkeit besitzen.

Je nach Konfiguration sind folgende Aspekte zentral:
- Sequenzlängenbegrenzung (z. B. `MAX_SEQ_LEN`)
- (Optional) Gradient Clipping zur Stabilisierung kleiner, nicht optimierter Implementationen (Pascanu et al., 2013)
- (Optional) Mini-Batching/Gradient Accumulation (Roadmap, falls noch nicht umgesetzt)

---

## Inference

Inference erfolgt im einfachsten Modus über **greedy decoding**: das wahrscheinlichste Folgetoken wird iterativ ausgewählt, bis EOS erreicht ist oder die maximale Sequenzlänge greift. Sampling-Verfahren wie Temperature / top-k / top-p können je nach Implementationsstand ergänzt oder bereits vorhanden sein; in der Literatur gelten sie als praxisrelevant für Textqualität (Holtzman et al., 2020).

---

## Checkpoints und Reproduzierbarkeit

### Warum „Load with Rebuild“ erforderlich ist

Die Output-Projektion besitzt typischerweise die Form \([d_{\text{emb}}, |V|]\), wobei \(|V|\) direkt von der Tokenizer-Vokabulargröße abhängt. Wird beim Laden ein Tokenizer mit anderer Vokabulargröße verwendet, entstehen Shape-Mismatches.

Daher implementiert das System beim Laden (konzeptionell) folgende Schritte:
1. Checkpoint laden und validieren (Magic/Version)
2. Tokenizer aus Checkpoint rekonstruieren
3. Modell **neu aufbauen** (Rebuild) anhand \(|V|\) aus dem Checkpoint
4. Parametervektor einspielen und Länge/Form prüfen

### Atomare Writes

Beim Speichern wird eine atomare Write-Strategie (Temp-Datei + Rename) eingesetzt, um inkonsistente Checkpoints bei Abbruch oder Systemstörungen zu vermeiden.

---

## Verteilte Ausführung und Fault Tolerance (Konzept)

Die Multi-Branch-Struktur ist so modelliert, dass **ein Pfad** als **Partitionseinheit** auf unterschiedliche Nodes gelegt werden kann, während eine Aggregationsinstanz die Pfadausgaben fusioniert. Für Ausfallsicherheit wird eine Maskierung \(m_i^{(l)} \in \{0,1\}\) verwendet; die Gewichte werden im Ausfallfall renormalisiert:

\[
\tilde{\alpha}_i^{(l)} = \frac{m_i^{(l)} \alpha_i^{(l)}}{\sum_{j=1}^{K} m_j^{(l)} \alpha_j^{(l)}} \quad (\text{sofern Nenner} > 0),
\quad
\tilde{h}^{(l+1)} = \sum_{i=1}^{K} \tilde{\alpha}_i^{(l)} z_i^{(l)}.
\]

Damit bleibt die Layerfunktion wohldefiniert, solange mindestens ein Pfad verfügbar ist. In P2P-Settings sind Quorum-/Timeout-Policies sowie Anti-Weight-Collapse-Regeln methodisch zentral, um Tail-Latency und Single-Point-of-Failure-Effekte zu begrenzen.

---

## Sicherheit und Robustheit (Systemperspektive)

In offenen oder teiloffenen verteilten Umgebungen erhöhen parallele Pfade die Angriffsfläche (Byzantinische Outputs, Straggling/DoS, Update-Poisoning). Ein MBT-System benötigt daher typischerweise:
- Integrität von Modellartefakten (Hashes/Signaturen/Versionierung)
- Quorum-/Timeout-Policies gegen Straggler-Tail-Latency
- Norm- und Gewichtskontrollen in der Aggregation
- (Optional) Admission Control für neue Pfade bei „Continuous Expandable Width“

Eine Blockchain kann konzeptionell als Governance- und Audit-Schicht dienen, ist jedoch nicht als Ausführungsumgebung des Modells intendiert, sondern als Root-of-Trust für Identität, Artefakt-Hashes und Update-Freigaben.

---

## Abgrenzung zu MoE / Switch / Multi-Path

- **MoE/Switch**: Breite wird primär über **sparsames, tokenweises Routing** auf wenige Expert*innen realisiert; Ziel ist Parameter-Skalierung bei begrenztem Compute pro Token (Shazeer et al., 2017; Fedus et al., 2022).
- **Multi-Path (Residual-Interpretation)**: beschreibt Mehrpfadigkeit eher analytisch als explizite, orchestrierbare Parallelstruktur (Veit et al., 2016).
- **MBT**: definiert Mehrpfadigkeit als **gleichzeitig aktive Pfade pro Layer** mit **expliziter Aggregation**, wodurch Verteilbarkeit, Robustheit und kontinuierliche Erweiterbarkeit systematisch adressiert werden.

---

## Roadmap

Mögliche nächste Schritte (abhängig vom aktuellen Stand):
- **Effiziente Inferenz**: KV-Cache, Batching, Maskierung, Mixed Precision
- **Verteilungsruntime**: Branch-Discovery, Scheduling, Quorum-basierte Aggregation, Straggler-Management
- **Robuste Aggregation**: trimmed mean / median-of-means, Reputationsgewichte, Anti-Verarmungs-Governance
- **Continuous Learning Governance**: Update-Validierung, Rollback, Poisoning-Detektion
- **Tests**: Tokenizer-Determinismus, Softmax-Stabilität, Checkpoint-Roundtrip, Golden-Tests

---

## Zitation

Wenn Inhalte aus dem Projekt zitiert werden, wird eine Referenz auf dieses Repository sowie auf die im Projektkontext genannten Quellen empfohlen (siehe unten).

---

## Quellen (APA)

Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Le, Q. V., Mao, M. Z., Ranzato, M., Senior, A., Tucker, P., Yang, K., & Ng, A. Y. (2012). Large scale distributed deep networks. In *Advances in Neural Information Processing Systems*.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research, 23*(120), 1–39.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The curious case of neural text degeneration. In *International Conference on Learning Representations*.

Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. In *International Conference on Machine Learning*.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*.

Veit, A., Wilber, M. J., & Belongie, S. (2016). Residual networks behave like ensembles of relatively shallow networks. In *Advances in Neural Information Processing Systems*.

---

## Lizenz

Siehe `LICENSE` im Repository.

---

## Kontakt

- Kontakt: mschlieper@expchat.ai
- Verwandte Implementationen/Referenzen (Projektumfeld):
  - Rust Distributed GPT Node: https://github.com/mhoellerschlieper/Rust-Distributed-GPT-Node
  - LLM Rust: https://github.com/mhoellerschlieper/LLM_Rust


