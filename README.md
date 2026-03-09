# MTech Thesis — Biaffine Dependency Parser

> **Full implementation of Dozat & Manning (2017) "Deep Biaffine Attention for Neural Dependency Parsing" with two novel innovations, trained and evaluated on the Universal Dependencies English EWT corpus.**

---

## Table of Contents

1. [Overview](#overview)
2. [Paper Reproduced](#paper-reproduced)
3. [Project Structure](#project-structure)
4. [Architecture Details](#architecture-details)
5. [Innovations](#innovations)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Configuration](#configuration)
10. [Implementation Notes](#implementation-notes)
11. [References](#references)

---

## Overview

This repository is the complete implementation for an MTech thesis based on the paper **"Deep Biaffine Attention for Neural Dependency Parsing"** (Dozat & Manning, 2017). The project:

- Implements the original paper from scratch in PyTorch (no pre-built parser frameworks used)
- Adds **two novel innovations** on top of the base model
- Trains and evaluates all three systems on the UD English EWT treebank
- Provides a clean, modular codebase suitable for research and extension

---

## Paper Reproduced

**Deep Biaffine Attention for Neural Dependency Parsing**
Timothy Dozat and Christopher D. Manning
ICLR 2017 — arXiv:1611.01734

**Core idea:** Instead of scoring arc candidates with a shallow MLP, use a biaffine (bilinear + linear) function that directly models both the probability that a word is a good head and the compatibility between a specific (dependent, head) pair. This outperforms prior attention mechanisms by factoring the scoring cleanly.

---

## Project Structure

```
biaffine_parser/
├── configs/
│   ├── base.yaml           # Baseline BiLSTM — exact paper hyperparameters
│   ├── base_char.yaml      # Baseline + CharLSTM (Innovation 1)
│   └── bert.yaml           # BERT encoder (Innovation 2)
│
├── data/
│   ├── conllu.py           # CoNLL-U reader and writer
│   ├── vocab.py            # Vocabulary builder (word / tag / rel / char)
│   └── dataset.py          # PyTorch Dataset and DataLoader
│
├── modules/
│   ├── biaffine.py         # Core biaffine attention layer
│   ├── bilstm.py           # BiLSTM with variational dropout
│   ├── mlp.py              # Dimension-reducing MLP
│   └── char_lstm.py        # Character-level BiLSTM (Innovation 1)
│
├── model/
│   ├── base_parser.py      # Full BiLSTM biaffine parser (~12.4M params)
│   └── bert_parser.py      # BERT-based parser (~111M params, Innovation 2)
│
├── utils/
│   ├── mst.py              # Chu-Liu / Edmonds MST decoder
│   ├── metric.py           # UAS / LAS evaluation metrics
│   └── scheduler.py        # Annealed Adam optimizer
│
├── train.py                # Training loop (supports all 3 model types)
├── evaluate.py             # Evaluation script (auto-detects model type)
└── requirements.txt
```

---

## Architecture Details

### 1. Input Representation

Each word is represented by concatenating:

| Component | Dimension | Notes |
|-----------|-----------|-------|
| Word embedding | 100 | Randomly initialized; GloVe/fastText supported |
| POS tag embedding | 100 | Universal POS tags from CoNLL-U |
| Char embedding (optional) | 100 | Innovation 1 — see below |

**Input dropout (Section 4.2.4):** Word and tag embeddings are dropped independently per-token (not per-dimension) with p=0.33. This is different from standard dropout and is implemented as a per-token Bernoulli mask broadcast over the embedding dimension.

### 2. BiLSTM Encoder

- 3-layer stacked Bidirectional LSTM
- 400 hidden units per direction → 800-dimensional output
- **Variational dropout (Gal & Ghahramani 2015):** One dropout mask is sampled once per forward pass and applied identically at every timestep. This is crucial for performance and is implemented from scratch as `VarDropout` in `modules/bilstm.py`
- Dropout rate: 0.33

### 3. Dimension-Reducing MLPs

Four separate MLP projections (Equations 4–5 of the paper):

| MLP | Output Dim | Purpose |
|-----|-----------|---------|
| arc-dep MLP | 500 | Word's representation as a dependent |
| arc-head MLP | 500 | Word's representation as a head |
| label-dep MLP | 100 | Label representation as a dependent |
| label-head MLP | 100 | Label representation as a head |

Each MLP: `Linear → LeakyReLU(0.1) → Dropout(0.33)` with Xavier uniform initialization.

### 4. Biaffine Classifier

The core of the paper — implemented in `modules/biaffine.py`.

**Arc scoring** (Equations 2 and 6):

```
score(i, j) = h_dep_i^T  W  h_head_j  +  h_head_j^T u
              ^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^
              bilinear term (which pair)   linear term (j is a good head)
```

**Label scoring** (Equation 3):

```
score(i, j, l) = h_dep_i^T  W_l  h_head_j  +  (h_dep_i ⊕ h_head_j)^T u_l  +  b_l
```

Implementation uses `torch.einsum('bxi,oij,byj->boxy', x, W, y)` for efficiency.

- Arc biaffine output: `[batch, dep, head]`
- Label biaffine output: `[batch, dep, head, n_rels]`

### 5. Loss Function

- **Arc loss:** CrossEntropy over all non-root, non-padding tokens; gold head index as target
- **Label loss:** CrossEntropy at the gold head position; gold relation as target
- Total loss = arc_loss + label_loss
- Root token (index 0) is excluded from loss computation

### 6. Optimizer — Annealed Adam

From Section 4.2.5 of the paper:

- β₁ = 0.9, β₂ = **0.9** (not the PyTorch default of 0.999 — this matters significantly)
- Initial lr = 2e-3
- Decay: `lr_t = lr_0 × 0.75^(t / 5000)` applied every step

### 7. Decoding

- **Training / fast eval:** Greedy argmax per dependent word
- **Test evaluation:** **Chu-Liu/Edmonds MST algorithm** via `networkx.maximum_spanning_arborescence` — guarantees a valid dependency tree (no cycles, exactly one root)

---

## Innovations

### Innovation 1 — Character-Level BiLSTM Embeddings

**File:** `modules/char_lstm.py`, enabled via `use_char: true` in config

The original paper uses only word-form and POS-tag embeddings. This innovation adds a character-level BiLSTM that produces a character-aware embedding for each word:

1. Each word's characters are embedded into 50-dimensional vectors
2. A BiLSTM processes the character sequence
3. The final forward and backward hidden states are concatenated → 100-dimensional character embedding
4. This is concatenated with the word and POS embeddings before the BiLSTM encoder

**Why it helps:**
- OOV words — the model can still produce a meaningful representation from characters
- Morphological patterns — prefixes/suffixes carry syntactic information (e.g., "-ing", "-ed", "-tion")
- Named entities, numbers, hyphenated words

**Result:** +0.51% LAS over baseline, converges faster (epoch 25 vs 30)

---

### Innovation 2 — BERT Encoder

**File:** `model/bert_parser.py`, config: `configs/bert.yaml`

Replaces the entire BiLSTM encoder with a pre-trained BERT model (`bert-base-uncased`, 110M parameters). The biaffine scoring head (MLPs + biaffine classifiers) remains unchanged.

**Key engineering challenges:**

1. **Subword tokenization:** BERT tokenizes words into subword pieces (e.g., "running" → ["run", "##ning"]). The dependency parser needs one representation per word. Solution: average pooling of all subword representations per word, implemented via a `word_map` tensor of shape `[batch, seq_len, max_subwords_per_word]`.

2. **Differential learning rates:** BERT's pre-trained weights are fine-tuned with a much smaller learning rate to prevent catastrophic forgetting:
   - BERT encoder layers: lr = 2e-5
   - Task head (MLPs + biaffine): lr = 1e-3

3. **Memory:** BERT requires significantly more memory → batch size reduced from 32 to 16.

**BERT model:** `bert-base-uncased` (12 transformer layers, 768 hidden dim, 12 attention heads, 110M parameters)

**Result:** +5.21% LAS over baseline — the largest single gain in the project

---

## Results

All experiments trained and evaluated on **Universal Dependencies English EWT**:

| Split | Sentences | Tokens |
|-------|-----------|--------|
| Train | 12,544 | 204,586 |
| Dev | 2,001 | 25,148 |
| Test | 2,077 | 25,148 |

### Test Set Results (MST Decoding)

| Model | Params | Best Epoch | Dev LAS | Test UAS | Test LAS |
|-------|--------|-----------|---------|----------|----------|
| Baseline BiLSTM (Dozat & Manning 2017) | 12.4M | 30 | 87.43% | 89.44% | 87.41% |
| + CharLSTM [Innovation 1] | 12.8M | 25 | 88.36% | 89.68% | 87.92% |
| BERT [Innovation 2] | 111M | 17 | 92.71% | **94.49%** | **92.62%** |

### Improvement Summary

| Innovation | UAS Gain | LAS Gain |
|------------|---------|---------|
| CharLSTM over Baseline | +0.24% | +0.51% |
| BERT over Baseline | +5.05% | **+5.21%** |

### Analysis

- **CharLSTM** provides a consistent improvement at low cost (+0.4M parameters). Converges faster because character embeddings help represent rare and OOV words from the very start of training.
- **BERT** delivers a massive +5.21% LAS improvement because its contextual representations encode rich syntactic information learned from 3.3 billion words of pre-training. It also converges in fewer epochs (17 vs 30) despite having 9× more parameters.

---

## Installation

```bash
# Clone the repository
git clone git@github.com:Sansgithub22/mtechthesis3biaffine-parser.git
cd mtechthesis3biaffine-parser

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch >= 2.0.0, HuggingFace Transformers >= 4.35.0, networkx >= 3.0

### Download Data

```bash
mkdir -p data
wget -O data/en_ewt-ud-train.conllu \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
wget -O data/en_ewt-ud-dev.conllu \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu
wget -O data/en_ewt-ud-test.conllu \
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu
```

---

## Usage

### Training

```bash
# Experiment 1: Baseline BiLSTM (Dozat & Manning 2017)
python train.py --config configs/base.yaml

# Experiment 2: BiLSTM + CharLSTM (Innovation 1)
python train.py --config configs/base_char.yaml

# Experiment 3: BERT parser (Innovation 2)
python train.py --config configs/bert.yaml
```

Training prints per-epoch stats:

```
 Epoch   Train Loss  Train UAS  Train LAS    Dev UAS    Dev LAS
-----------------------------------------------------------------
     1      4.2831    72.13%    63.45%    76.29%    68.12%  [45s]
     2      3.1047    81.55%    73.82%    82.41%    75.63%  [44s]
   ...
    17      0.4812    96.23%    94.81%    93.85%    92.71%  [47s]
         *** New best LAS: 92.71% — model saved ***
```

Checkpoints saved to `checkpoints/<model_name>/best_model.pt` and `checkpoints/<model_name>/vocab.pkl`.

### Evaluation

```bash
# Evaluate on test set (model type auto-detected from checkpoint)
python evaluate.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --test_file  data/en_ewt-ud-test.conllu \
    --vocab      checkpoints/baseline/vocab.pkl \
    --use_mst

# Save predictions to CoNLL-U file (for use with official eval tools)
python evaluate.py \
    --checkpoint checkpoints/bert/best_model.pt \
    --test_file  data/en_ewt-ud-test.conllu \
    --vocab      checkpoints/bert/vocab.pkl \
    --use_mst \
    --output     predictions.conllu
```

Output:

```
==================================================
  TEST RESULTS
==================================================
  UAS : 94.49%
  LAS : 92.62%
  Tokens evaluated: 25,148
==================================================
```

---

## Configuration

### Hyperparameter Reference

| Parameter | Baseline | CharLSTM | BERT | Description |
|-----------|---------|---------|------|-------------|
| `word_embed_dim` | 100 | 100 | — | Word embedding size |
| `tag_embed_dim` | 100 | 100 | — | POS tag embedding size |
| `use_char` | false | **true** | — | Enable CharLSTM |
| `char_embed_dim` | — | 50 | — | Char embedding size |
| `char_out_dim` | — | 100 | — | CharLSTM output size |
| `lstm_hidden` | 400 | 400 | — | BiLSTM units per direction |
| `lstm_layers` | 3 | 3 | — | Number of BiLSTM layers |
| `lstm_dropout` | 0.33 | 0.33 | — | Variational dropout |
| `bert_model` | — | — | bert-base-uncased | HuggingFace model name |
| `bert_dropout` | — | — | 0.1 | BERT output dropout |
| `arc_mlp_dim` | 500 | 500 | 500 | Arc MLP hidden size |
| `label_mlp_dim` | 100 | 100 | 100 | Label MLP hidden size |
| `mlp_dropout` | 0.33 | 0.33 | 0.33 | MLP dropout |
| `embed_dropout` | 0.33 | 0.33 | — | Input embedding dropout |
| `lr` | 0.002 | 0.002 | — | Adam learning rate |
| `bert_lr` | — | — | 2e-5 | BERT encoder LR |
| `head_lr` | — | — | 1e-3 | Task head LR |
| `decay_factor` | 0.75 | 0.75 | 0.75 | LR annealing factor |
| `decay_steps` | 5000 | 5000 | 5000 | Steps between annealing |
| `batch_size` | 32 | 32 | 16 | Training batch size |
| `max_epochs` | 30 | 30 | 50 | Maximum training epochs |
| `patience` | 10 | 10 | 5 | Early stopping patience |

---

## Implementation Notes

### Critical Details (Common Mistakes Avoided)

1. **Variational dropout** — Standard PyTorch LSTM dropout drops different neurons each timestep. The paper requires the **same mask at every timestep**. Implemented as `VarDropout` in `modules/bilstm.py` which samples one mask per forward call and broadcasts it over the sequence length.

2. **β₂ = 0.9 for Adam** — The paper explicitly uses β₂ = 0.9, not the PyTorch default of 0.999. This makes the optimizer more responsive to recent gradients and significantly affects convergence speed and final accuracy.

3. **Per-token embedding dropout** — The paper drops entire word/tag embeddings per-token (not individual dimensions). Implemented in `BiaffineParser._independent_embed_dropout()` as a Bernoulli mask of shape `[batch, seq_len, 1]` broadcast over the embedding dimension.

4. **Root token excluded from loss and metrics** — Token 0 is the virtual ROOT node. It is never a dependent, so it is excluded from cross-entropy loss computation and from UAS/LAS metric accumulation.

5. **MST only at test time** — Chu-Liu/Edmonds is O(n²) and slow. Greedy decoding is used during training for speed; MST decoding is used at evaluation time for correctness (valid tree guarantee).

6. **BERT subword alignment** — The `word_map` tensor (`[batch, seq_len, max_sub]`) maps each CoNLL-U word to its BERT subword token indices. Padding entries are -1, which are clamped to 0 before gathering and then masked out before averaging.

### Module Summary

| File | Key Class / Function | What it does |
|------|---------------------|-------------|
| `modules/biaffine.py` | `Biaffine` | `einsum('bxi,oij,byj->boxy', x, W, y)` |
| `modules/bilstm.py` | `VarDropout`, `BiLSTMEncoder` | Variational dropout + packed BiLSTM |
| `modules/mlp.py` | `MLP` | Linear → LeakyReLU(0.1) → Dropout |
| `modules/char_lstm.py` | `CharLSTM` | Flatten words → BiLSTM → concat final states |
| `model/base_parser.py` | `BiaffineParser` | Full BiLSTM pipeline, 12.4M params |
| `model/bert_parser.py` | `BERTBiaffineParser` | BERT + subword pooling + task head |
| `data/conllu.py` | `read_conllu`, `write_conllu` | CoNLL-U I/O (skips MWT and empty nodes) |
| `data/vocab.py` | `Vocab` | PAD=0, UNK=1, ROOT=2 specials |
| `data/dataset.py` | `DependencyDataset`, `collate_fn` | Indexing + padding |
| `utils/mst.py` | `batch_decode_mst` | `nx.maximum_spanning_arborescence` |
| `utils/metric.py` | `AttachmentMetric` | UAS/LAS accumulator |
| `utils/scheduler.py` | `AnnealingScheduler` | Step-based LR decay |
| `train.py` | `train`, `train_epoch`, `evaluate` | Full training loop with early stopping |
| `evaluate.py` | `predict`, `evaluate` | Auto-detects BERT vs BiLSTM from checkpoint |

---

## References

- Dozat, T., & Manning, C. D. (2017). Deep Biaffine Attention for Neural Dependency Parsing. *ICLR 2017*. arXiv:1611.01734
- Gal, Y., & Ghahramani, Z. (2016). A Theoretically Grounded Application of Dropout in Recurrent Neural Networks. *NeurIPS 2016*.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.
- Nivre, J. et al. (2020). Universal Dependencies v2. *LREC 2020*.
- Edmonds, J. (1967). Optimum Branchings. *Journal of Research of the National Bureau of Standards*.
