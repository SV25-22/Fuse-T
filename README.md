# Fuse-T

Fuse-T is a gated residual late-fusion model for binary rumour classification under unseen-event shift. It combines RoBERTa text semantics with GraphSAGE representations of conversational reply topology, using text as the stable decision backbone and admitting graph evidence through a learned element-wise gate. This repository is the official implementation accompanying the IcETRAN 2026 paper.

## Publication information

**“Fuse-T Gated Residual Late Fusion of Text Semantics and Thread Topology for Unseen-Event Rumour Classification in Conversational Reply Graphs”** — Aleksandar Stanković, 13th International Conference on Electrical, Electronics and Computer Engineering (IcETRAN), Silver Lake, Serbia, 8–11 June 2026.

[Paper](IcETRAN_2026_paper_16.pdf) · [Poster](Poster.png) · [Official conference program (paper VII1.3)](https://www.conf.etran.rs/wp-content/uploads/2026/05/PROGRAM_ETRAN_2026_print_fajl_korekcija.pdf) · [Official IcETRAN 2026 proceedings catalogue](https://www.etran.rs/2026/en/proceeding/)

A paper-specific DOI or IEEE Xplore record is not listed because neither could be verified for this paper at release time.

## Key idea

Text-only models can overfit event-specific wording, while topology-only models have little evidence when reply graphs are young. Fuse-T projects the graph embedding into the text space and adds it as a gated residual:

```text
z_graph = φ(h_graph)
α       = sigmoid(W [h_text ; z_graph] + b)
h_fused = h_text + α ⊙ z_graph
```

The gate bias starts negative (`-4` in the reference configuration), so training begins near the text-only solution and opens the topology path when it is useful.

The graph branch uses seven per-node structural features (depth, log elapsed time, in/out-degree, log subtree size, and root/leaf indicators) plus thread-level size, depth, branching, degree-dispersion, time-span, virality, and temporal-growth features. TAG-GNN instead concatenates a RoBERTa embedding to each node before graph message passing.

## Main verified results

The paper reports leave-one-event-out (LOEO) Macro-F1 over seven PHEME events: `charliehebdo`, `ferguson`, `germanwings`, `gurlitt`, `ottawashooting`, `putinmissing`, and `sydneysiege`.

| Model | Mean Macro-F1 (%) |
|---|---:|
| RoBERTa text baseline | 62.13 |
| GraphSAGE topology baseline | 52.48 |
| TAG-GNN early fusion | 62.42 |
| **Fuse-T** | **65.68** |

At a 10-minute observation window, Fuse-T reports about 63% Macro-F1. These are reported paper values, not metrics recomputed during installation; event-level results and ablations are in the [paper](IcETRAN_2026_paper_16.pdf).

## Repository layout

```text
.
├── CITATION.cff                 Citation metadata for software and paper
├── LICENSE                      MIT license for source code
├── IcETRAN_2026_paper_16.pdf   Paper artifact
├── Poster.png                   Conference poster
├── pyproject.toml               Python metadata and dependencies
├── run_early_detection.sh       All-fold time-window experiments
├── scripts/
│   ├── preprocess_pheme.py      Raw PHEME to normalized JSONL
│   ├── make_splits_loeo.py      Deterministic LOEO split creation
│   ├── smoke_test.py            Dataset-free pipeline check
│   └── summarize_results.py     Markdown table generator
├── src/
│   ├── train_text.py            RoBERTa baseline
│   ├── train_gnn.py             GraphSAGE baseline
│   ├── train_tag_gnn.py         Text-attributed GNN baseline
│   └── train_fusion.py          Fuse-T and fusion ablations
└── tests/                       Release-tool regression tests
```

`data/`, `results/`, checkpoints, and caches are generated locally and excluded from Git.

## Requirements and installation

Python 3.10–3.12 is supported. A CUDA-capable GPU is recommended for the RoBERTa and full LOEO runs; preprocessing and the smoke test run on CPU.

1. Create and activate an environment:

   ```bash
   conda create -n fuse-t python=3.11 -y
   conda activate fuse-t
   ```

2. Install PyTorch using the command for your operating system and CUDA version from the [official PyTorch installer](https://pytorch.org/get-started/locally/).

3. Install PyTorch Geometric using its [official installation guide](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html), then install this project:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

The dependency ranges in `pyproject.toml` describe the supported release environment. Install the CUDA-specific PyTorch build before the editable install so `pip` does not select a CPU build unintentionally.

## Dataset acquisition and expected layout

Download the nine-event [PHEME dataset for Rumour Detection and Veracity Classification](https://doi.org/10.6084/m9.figshare.6392078.v1) from Figshare and extract `PHEME_veracity.tar.bz2` locally. The repository does not download or redistribute PHEME or its underlying social-media content.

The preprocessor searches recursively, but the expected structure is:

```text
data/raw/pheme/PHEME_veracity/
├── charliehebdo-all-rnr-threads/
│   ├── rumours/<thread-id>/
│   │   ├── source-tweets/*.json
│   │   ├── reactions/*.json
│   │   ├── structure.json
│   │   └── annotation.json
│   └── non-rumours/<thread-id>/...
└── ...
```

Use the dataset only under the terms stated by its distributors and the relevant social-media platform/content owners.

## Preprocessing and LOEO split creation

Create one JSONL record per conversation. By default, text is lowercased and URLs and user mentions are removed, matching the paper’s stated normalization. The corresponding `--no-lowercase`, `--no-strip-urls`, and `--no-strip-mentions` switches preserve those elements when needed.

```bash
python scripts/preprocess_pheme.py \
  --data_root data/raw/pheme/PHEME_veracity \
  --out_jsonl data/processed/threads.jsonl \
  --out_summary data/processed/summary.json \
  --label_mode binary
```

Create deterministic LOEO folds with a stratified 10% validation split. The explicit exclusions reproduce the seven-event scope reported in the paper; the script also prints any event rejected by its size, class-balance, or inferred-structure checks.

```bash
python scripts/make_splits_loeo.py \
  --threads_jsonl data/processed/threads.jsonl \
  --out data/processed/splits_loeo.json \
  --val_frac 0.10 \
  --seed 42 \
  --exclude_events ebola-essien-all-rnr-threads,prince-toronto-all-rnr-threads
```

Inspect `summary.json` and the split script’s output before training. Event directory names become the `--fold` values.

## Quick-start smoke test

Exercise preprocessing, paper-aligned text normalization, and LOEO split creation without downloading PHEME:

```bash
python scripts/smoke_test.py
```

Expected final line: `Smoke test passed: 8 synthetic threads, 2 LOEO folds.`

## Training the baselines and Fuse-T

The following commands run one fold. Replace `charliehebdo-all-rnr-threads` with each key in `data/processed/splits_loeo.json` for the complete evaluation. Every trainer writes `best.pt` and `result.json` beneath its `--out_dir`.

RoBERTa text baseline:

```bash
python src/train_text.py \
  --fold charliehebdo-all-rnr-threads \
  --out_dir results/text/charliehebdo-all-rnr-threads \
  --model_name roberta-base --max_length 256 --k_replies 0
```

GraphSAGE topology baseline:

```bash
python src/train_gnn.py \
  --fold charliehebdo-all-rnr-threads \
  --out_dir results/gnn/charliehebdo-all-rnr-threads \
  --readout rootmeanmax --hidden_dim 128 --num_layers 2 --dropout 0.2
```

TAG-GNN early-fusion baseline, initialized from the text checkpoint:

```bash
python src/train_tag_gnn.py \
  --fold charliehebdo-all-rnr-threads \
  --out_dir results/tag_gnn/charliehebdo-all-rnr-threads \
  --model_name roberta-base \
  --init_text_ckpt results/text/charliehebdo-all-rnr-threads/best.pt
```

Fuse-T, initialized from both unimodal baselines:

```bash
python src/train_fusion.py \
  --fold charliehebdo-all-rnr-threads \
  --out_dir results/fuse_t/charliehebdo-all-rnr-threads \
  --model_name roberta-base --max_length 256 \
  --readout meanmax --hidden_dim 128 --num_layers 2 --dropout 0.2 \
  --fusion_type residual_gate --gate_bias_init -4 \
  --aux_weight 0.3 --gnn_warmup_epochs 3 --freeze_text_epochs 1 \
  --init_text_ckpt results/text/charliehebdo-all-rnr-threads/best.pt \
  --init_gnn_ckpt results/gnn/charliehebdo-all-rnr-threads/best.pt
```

Use `--fusion_type residual_nogate`, `--gate_bias_init 0`, or `--fusion_type concat` for the principal fusion ablations.

## Early-detection experiments

Early detection retains the root plus replies timestamped within `--early_minutes`. If a window leaves fewer than two nodes, `--early_k K` can retain the earliest `K` replies as a fallback. After producing the text checkpoints for every fold, run all folds at the paper’s 10-minute window on Linux, macOS, or WSL:

```bash
bash run_early_detection.sh 10
```

With no arguments, the runner uses `10 30 60 120 240` minutes. It derives folds from the split file instead of hard-coding event names. Override paths with the `THREADS`, `SPLITS`, and `RESULTS_ROOT` environment variables.

## Reproducing paper tables

Once every model/fold has a `result.json`, generate a strict Markdown table of per-event Macro-F1 values and their unweighted mean:

```bash
python scripts/summarize_results.py \
  --results-root results \
  --strict \
  --output results/table_loeo.md
```

Summarize a time-window experiment with:

```bash
python scripts/summarize_results.py \
  --results-root results/early_10 \
  --models gnn fuse_t \
  --strict
```

The summarizer reports the runs on disk; it does not substitute the values printed in the paper.

## Tests

The tests require no PHEME download or model checkpoint:

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts src tests
```

Full neural training is intentionally not part of the lightweight test suite.

## Limitations and reproducibility notes

- PHEME data, trained checkpoints, and raw experiment outputs are not redistributed. The paper’s reported metrics therefore cannot be verified from this repository alone without acquiring the dataset and rerunning the experiments.
- The paper records hyperparameter ranges rather than a complete per-fold run manifest. The commands above are a reference configuration within those ranges, not a guarantee of bit-for-bit recovery of every published value.
- Seeds are set and cuDNN deterministic mode is requested, but exact results can still vary with PyTorch/PyG versions, GPU hardware, kernels, and pretrained-model revisions. Record the resolved environment and commit hash with new results.
- Missing thread structure can be inferred as a star by the preprocessor and missing timestamps affect early-window truncation. `summary.json` records preprocessing choices; review inferred-structure ratios before accepting folds.
- The task is binary rumour versus non-rumour classification. It does not predict true, false, and unverified veracity labels separately.

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable software and paper metadata.

```bibtex
@inproceedings{stankovic2026fuset,
  author    = {Stankovi\'c, Aleksandar},
  title     = {Fuse-T Gated Residual Late Fusion of Text Semantics and Thread Topology for Unseen-Event Rumour Classification in Conversational Reply Graphs},
  booktitle = {13th International Conference on Electrical, Electronics and Computer Engineering (IcETRAN)},
  address   = {Silver Lake, Serbia},
  year      = {2026}
}
```

## Acknowledgment

The paper acknowledges computational support from Xinming Wang at the Institute of Automation, Chinese Academy of Sciences (CASIA).

## License

Repository source code is licensed under the [MIT License](LICENSE). The paper and poster retain their respective publication and copyright terms. PHEME and the underlying social-media data are distributed under their own terms and are not redistributed by this repository. The MIT license does not apply to the paper, poster, dataset, or underlying social-media content.
