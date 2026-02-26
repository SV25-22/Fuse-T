import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from datetime import datetime
try:
    from dateutil import parser as du_parser
except Exception:
    du_parser = None

TWITTER_FMT = "%a %b %d %H:%M:%S %z %Y"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ts(created_at: Optional[str]) -> Optional[int]:
    if not created_at:
        return None
    s = str(created_at).strip()
    if not s:
        return None
    if du_parser is not None:
        try:
            return int(du_parser.parse(s).timestamp())
        except Exception:
            pass
    for fmt in [TWITTER_FMT, "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"]:
        try:
            return int(datetime.strptime(s, fmt).timestamp())
        except Exception:
            continue
    try:
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp())
    except Exception:
        return None


def load_threads_jsonl(path: Path) -> List[dict]:
    threads = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    return threads


def load_splits(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_text_example(thread: dict, tokenizer, k_replies: int) -> str:
    root_id = thread.get("root_id") or thread.get("thread_id")
    nodes = thread["nodes"]

    id2node = {n["id"]: n for n in nodes}
    root_text = (id2node.get(root_id, {}) or {}).get("text", "") or ""

    if k_replies <= 0:
        return root_text

    items = []
    for n in nodes:
        if n["id"] == root_id:
            continue
        t = parse_ts(n.get("created_at"))
        items.append((t if t is not None else 10**18, n.get("text", "") or ""))

    items.sort(key=lambda x: x[0])
    replies = [txt for _, txt in items[:k_replies] if txt.strip()]

    sep = tokenizer.sep_token or "</s>"
    parts = [root_text] + replies
    return f" {sep} ".join(parts)


class PhemeTextDataset(Dataset):
    def __init__(self, threads: List[dict], tokenizer, k_replies: int):
        self.threads = threads
        self.tokenizer = tokenizer
        self.k_replies = k_replies

    def __len__(self):
        return len(self.threads)

    def __getitem__(self, idx):
        t = self.threads[idx]
        text = build_text_example(t, self.tokenizer, self.k_replies)
        return {
            "text": text,
            "y": int(t["y"]),
            "thread_id": t["thread_id"],
        }


def collate_fn(batch, tokenizer, max_length: int):
    texts = [b["text"] for b in batch]
    ys = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc, ys


class RobertaClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2, dropout: float = 0.2, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, **enc):
        out = self.encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.fc(self.drop(cls))
        return logits


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for enc, y in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        y = y.to(device)
        logits = model(**enc)
        pred = torch.argmax(logits, dim=-1)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
    macro = f1_score(ys, ps, average="macro")
    acc = accuracy_score(ys, ps)
    return macro, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=str, default="data/processed/threads.jsonl")
    ap.add_argument("--splits", type=str, default="data/processed/splits_loeo.json")
    ap.add_argument("--fold", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_name", type=str, default="roberta-base")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--k_replies", type=int, default=0, help="0 = root-only; else root + first K replies")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=2)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--freeze_encoder", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    threads = load_threads_jsonl(Path(args.threads))
    splits = load_splits(Path(args.splits))
    if args.fold not in splits:
        raise ValueError(f"Fold '{args.fold}' not found in splits file.")

    fold = splits[args.fold]
    train_ids = set(fold["train_thread_ids"])
    val_ids = set(fold["val_thread_ids"])
    test_ids = set(fold["test_thread_ids"])

    train_threads = [t for t in threads if t["thread_id"] in train_ids]
    val_threads = [t for t in threads if t["thread_id"] in val_ids]
    test_threads = [t for t in threads if t["thread_id"] in test_ids]

    train_ds = PhemeTextDataset(train_threads, tokenizer, args.k_replies)
    val_ds = PhemeTextDataset(val_threads, tokenizer, args.k_replies)
    test_ds = PhemeTextDataset(test_threads, tokenizer, args.k_replies)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )

    y_train = np.array([t["y"] for t in train_threads], dtype=np.int64)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    class_w = torch.tensor(weights, dtype=torch.float32).to(device)

    model = RobertaClassifier(
        model_name=args.model_name,
        num_classes=2,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    best_val = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for enc, y in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)

            logits = model(**enc)
            loss = nn.functional.cross_entropy(logits, y, weight=class_w)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            total_loss += float(loss.item())

        val_f1, val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch:02d}] loss={total_loss/max(1,len(train_loader)):.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f}")

        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop (best val_f1={best_val:.4f})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_f1, test_acc = evaluate(model, test_loader, device)

    result = {
        "title": "Fuse-T RoBERTa-only baseline",
        "fold": args.fold,
        "seed": args.seed,
        "model_name": args.model_name,
        "k_replies": args.k_replies,
        "max_length": args.max_length,
        "freeze_encoder": bool(args.freeze_encoder),
        "best_val_macro_f1": float(best_val),
        "test_macro_f1": float(test_f1),
        "test_acc": float(test_acc),
        "train_class_counts": counts.tolist(),
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_state, "config": vars(args)}, out_dir / "best.pt")

    print("Saved:", out_dir / "best.pt")
    print("Test macro-F1:", test_f1, "Test acc:", test_acc)


if __name__ == "__main__":
    main()
