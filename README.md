# Fuse-T: Gated Residual Late Fusion for Unseen-Event Rumour Classification

This repository contains the official implementation of **Fuse‑T**, a *gated residual late‑fusion* architecture that combines:
- **Text semantics** from a RoBERTa encoder (stable backbone), and
- **Propagation topology** from a GraphSAGE encoder over conversational reply graphs,

for **binary rumour vs. non‑rumour classification** under **unseen‑event (LOEO) generalization** and **early detection**.

> In the repository root you will also find:
> - **Poster:** `Poster.png`
> - **Paper:** `IcETRAN_2026_paper_16.pdf`

---

## What is Fuse‑T?

Many models either:
- rely only on **text** (strong semantics, but can overfit event‑specific keywords), or
- rely only on **propagation structure** (can be weak early when reply graphs are small).

**Fuse‑T** keeps the text model as a *stable decision backbone* and injects topology as a **controlled residual**:

\[
h_{\text{fused}} = h_{\text{text}} + \alpha \odot z_{\text{graph}}
\]

- \(z_{\text{graph}} = \phi(h_{\text{graph}})\) is a projection of the graph embedding into the text latent space.
- \(\alpha \in (0,1)^d\) is a **learned element‑wise gate** computed from \([h_{\text{text}}; z_{\text{graph}}]\).
- The gate bias is initialized **negative** (e.g., `-4`) so the model starts close to **text‑only** and opens the topology path only when it helps.

---

## Main results

Evaluation is **leave‑one‑event‑out (LOEO)** on **7 PHEME events**:

`charliehebdo, ferguson, germanwings, gurlitt, ottawashooting, putinmissing, sydneysiege`

**Average LOEO Macro‑F1 (%):**
- Text‑only (RoBERTa): **62.13**
- TAG‑GNN (text‑attributed GNN early fusion): **62.42**
- **Fuse‑T:** **65.68**

**Early detection (10 minutes of replies):**
- Fuse‑T stays robust at **~63% Macro‑F1**, while structure‑only models degrade substantially.

See `IcETRAN_2026_paper_16.pdf` and `Poster.png` in the root for full tables and figures.

---

## Repository layout


```
.
├── README.md
├── Poster.png
├── IcETRAN_2026_paper_16.pdf
├── scripts/
│   ├── preprocess_pheme.py
│   └── make_splits_loeo.py
├── src/
│   ├── train_text.py
│   ├── train_gnn.py
│   ├── train_tag_gnn.py
│   └── train_fusion.py
└── results/
```

---

## Installation

### 1) Create an environment
```bash
conda create -n fuse_t python=3.10 -y
conda activate fuse_t
```

### 2) Install dependencies
Core packages:
```bash
pip install -U pip
pip install torch torchvision torchaudio
pip install transformers scikit-learn numpy tqdm python-dateutil
```

PyTorch Geometric (**PyG**) must match your **Torch + CUDA** version. Follow the official PyG install instructions for your setup.

---

## Data: PHEME

This repo expects the **PHEME** conversational threads dataset (event‑partitioned - https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619).  

Once you have PHEME locally, unzip it somewhere like:
```
data/raw/pheme/...
```

---

## Preprocessing

### 1) Convert raw PHEME threads to a single JSONL
The script scans for `rumours/` (or `rumors/`) and `non-rumours/` (or `non-rumors/`), loads tweet text + timestamps, parses thread structure, and emits one record per thread.

```bash
python scripts/preprocess_pheme.py \
  --data_root data/raw/pheme \
  --out_jsonl data/processed/threads.jsonl \
  --out_summary data/processed/summary.json \
  --label_mode binary
```

Outputs:
- `data/processed/threads.jsonl` – one JSON per thread: nodes, edges, labels, timestamps, etc.
- `data/processed/summary.json` – dataset counts and detected event directories.

### 2) Build LOEO splits
```bash
python scripts/make_splits_loeo.py \
  --threads_jsonl data/processed/threads.jsonl \
  --out data/processed/splits_loeo.json \
  --val_frac 0.10 \
  --seed 42
```

This creates a LOEO fold per eligible event:
- train/val = all other events
- test = the held‑out event

---

## Training & evaluation

All training scripts write:
- `best.pt` (checkpoint)
- `result.json` (metrics and config)

Metrics are reported per fold using **Macro‑F1** (primary) and **accuracy**.

### 1) Text baseline (RoBERTa)
```bash
python src/train_text.py \
  --fold charliehebdo-all-rnr-threads \
  --out_dir results/text/charliehebdo-all-rnr-threads \
  --model_name roberta-base \
  --max_length 256 \
  --k_replies 0
```

Notes:
- `--k_replies 0` means **root post only**.
- You can set `--k_replies K` to concatenate the earliest K replies (ordered by time) to the root.

### 2) Structure-only baseline (GraphSAGE)
```bash
python src/train_gnn.py \
  --fold charliehebdo-all-rnr-threads \
  --out_dir results/gnn/charliehebdo-all-rnr-threads \
  --readout rootmeanmax \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.2
```

This uses:
- node structural features: depth, log time since root, in/out degree, log subtree size, root/leaf flags
- graph-level features `gfeat` (size, depth, degree dispersion, time span, growth histogram, etc.)

### 3) TAG-GNN baseline (text-attributed GNN, early fusion)
This baseline precomputes RoBERTa node embeddings and concatenates them to structural features.

```bash
python src/train_tag_gnn.py \
  --fold charliehebdo-all-rnr-threads \
  --out_dir results/tag_gnn/charliehebdo-all-rnr-threads \
  --init_text_ckpt results/text/charliehebdo-all-rnr-threads/best.pt \
  --model_name roberta-base
```


### 4) Fuse‑T (gated residual late fusion)
Fuse‑T reuses the **text classifier head** and injects a projected graph embedding via a **gate**.

```bash
python src/train_fusion.py \
  --fold charliehebdo-all-rnr-threads \
  --out_dir results/fuse_t/charliehebdo-all-rnr-threads \
  --model_name roberta-base \
  --readout meanmax \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.2 \
  --fusion_type residual_gate \
  --gate_bias_init -4.0 \
  --aux_weight 0.3 \
  --gnn_warmup_epochs 3 \
  --freeze_text_epochs 1 \
  --init_text_ckpt results/text/charliehebdo-all-rnr-threads/best.pt
```

Optional:
- Initialize graph convs from a pre-trained GNN-only baseline:
```bash
  --init_gnn_ckpt results/gnn/charliehebdo-all-rnr-threads/best.pt
```

---

## Early detection experiments

Early detection truncates each thread to only include posts within the first **T minutes** from the root (plus the root).  
If timestamps are missing or too few nodes remain, scripts can fall back to keeping the earliest `K` replies via `--early_k`.

This script is available at `run_early_detection.sh`:

```bash
# Time thresholds in minutes
MINUTES=(10 30 60 120 240)
EVENTS=("charliehebdo-all-rnr-threads" "ebola-essien-all-rnr-threads" "ferguson-all-rnr-threads" "germanwings-crash-all-rnr-threads" "gurlitt-all-rnr-threads" "ottawashooting-all-rnr-threads" "prince-toronto-all-rnr-threads" "putinmissing-all-rnr-threads" "sydneysiege-all-rnr-threads")

for MINS in "${MINUTES[@]}"; do
    for EVENT in "${EVENTS[@]}"; do
        echo "=================================================="
        echo "Running Early Detection: ${MINS} mins | Fold: $EVENT"
        echo "=================================================="

        # 1. GNN-only at T minutes
        python src/train_gnn.py \
            --fold "$EVENT" \
            --early_minutes "$MINS" \
            --out_dir "results/early_${MINS}/gnn/$EVENT"

        # 2. Fuse-T at T minutes
        # Re-using the Text checkpoint, as text representation doesn't change with graph propagation time
        python src/train_fusion.py \
            --fold "$EVENT" \
            --early_minutes "$MINS" \
            --out_dir "results/early_${MINS}/fuse_t/$EVENT" \
            --init_text_ckpt "results/text/$EVENT/best.pt" \
            --init_gnn_ckpt "results/early_${MINS}/gnn/$EVENT/best.pt"
    done
done
```

---

## Reproducing the paper tables

After running LOEO folds, you can aggregate metrics from:
- `results/text/<fold>/result.json`
- `results/gnn/<fold>/result.json`
- `results/tag_gnn/<fold>/result.json`
- `results/fuse_t/<fold>/result.json`

The paper reports **Macro‑F1 per held‑out event** and the **mean across events**, plus ablations on:
- residual gate vs no gate
- gate bias initialization
- concat fusion

---

<!-- 
## Citation

If you use this code or build on Fuse‑T, please cite the paper (PDF in root):

```
@inproceedings{stankovic2026fuset,
  title     = {Fuse-T: Gated Residual Late Fusion of Text Semantics and Thread Topology for Unseen-Event Rumour Classification in Conversational Reply Graphs},
  author    = {Aleksandar Stankovi{\'c}},
  booktitle = {13th International Conference on Electrical, Electronics and Computer Engineering (IcETRAN)},
  year      = {2026}
}
```

--- -->

## Acknowledgment

Computational support is gratefully acknowledged from **Xinming Wang** (CASIA), as stated in the paper.
