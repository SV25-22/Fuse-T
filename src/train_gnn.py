import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
except Exception as e:
    raise RuntimeError(
        "torch_geometric not available. Make sure PyG is installed for your torch/cuda version."
    ) from e

from datetime import datetime
try:
    from dateutil import parser as du_parser
except Exception:
    du_parser = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


TWITTER_FMT = "%a %b %d %H:%M:%S %z %Y"

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



def gini_coef(values: List[float]) -> float:
    if not values:
        return 0.0
    x = np.asarray(values, dtype=np.float64)
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.sum((2 * np.arange(1, n + 1) - n - 1) * x)
    denom = n * np.sum(x)
    if denom == 0:
        return 0.0
    return float(cum / denom)


def build_graph_from_thread(
    thread: dict,
    make_undirected: bool = True,
    early_minutes: int = -1,
    early_k: int = 0,
) -> Data:
    nodes = thread["nodes"]
    edges = thread["edges"]
    root_id = thread.get("root_id") or thread.get("thread_id")

    node_ids = [n["id"] for n in nodes]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    root_idx = id2idx.get(root_id, 0)

    ts = [parse_ts(n.get("created_at")) for n in nodes]
    root_ts = ts[root_idx] if ts[root_idx] is not None else None

    dt_sec = []
    for i, t in enumerate(ts):
        if i == root_idx:
            dt_sec.append(0.0)
        elif root_ts is None or t is None:
            dt_sec.append(None)
        else:
            dt_sec.append(max(0.0, float(t - root_ts)))
    keep = set(range(len(nodes)))

    if early_minutes and early_minutes > 0 and root_ts is not None:
        limit = float(early_minutes) * 60.0
        keep = {root_idx}
        for i, d in enumerate(dt_sec):
            if i == root_idx:
                continue
            if d is not None and d <= limit:
                keep.add(i)

        if len(keep) < 2 and early_k and early_k > 0:
            order = [(dt_sec[i] if dt_sec[i] is not None else 1e18, i) for i in range(len(nodes))]
            order.sort()
            keep = {root_idx}
            for _, i in order:
                if i == root_idx:
                    continue
                keep.add(i)
                if len(keep) >= 1 + early_k:
                    break

    elif early_k and early_k > 0:
        order = [((ts[i] if ts[i] is not None else 1e18), i) for i in range(len(nodes))]
        order.sort()
        keep = {root_idx}
        for _, i in order:
            if i == root_idx: 
                continue
            keep.add(i)
            if len(keep) >= 1 + early_k:
                break   

    keep_list = sorted(list(keep))
    old2new = {old: new for new, old in enumerate(keep_list)}
    new_root_idx = old2new.get(root_idx, 0)

    node_ids_f = [node_ids[i] for i in keep_list]
    id2idx_f = {nid: i for i, nid in enumerate(node_ids_f)}

    dt_sec_f = []
    for old in keep_list:
        d = dt_sec[old]
        if old == root_idx:
            dt_sec_f.append(0.0)
        else:
            dt_sec_f.append(float(d) if d is not None else 0.0)  

    bins = [60, 2*60, 5*60, 10*60, 30*60, 60*60]  
    hist = [0]*(len(bins)+1)

    for i, d in enumerate(dt_sec_f):
        if i == new_root_idx:
            continue
        b = 0
        while b < len(bins) and d > bins[b]:
            b += 1
        hist[b] += 1

    hist_log = [math.log1p(h) for h in hist]

    children = defaultdict(list)
    indeg = [0] * len(node_ids_f)
    outdeg = [0] * len(node_ids_f)
    edge_pairs: List[Tuple[int, int]] = []

    for e in edges:
        p, c = e[0], e[1]
        if p not in id2idx or c not in id2idx:
            continue
        pi_old, ci_old = id2idx[p], id2idx[c]
        if pi_old not in keep or ci_old not in keep:
            continue
        pi, ci = old2new[pi_old], old2new[ci_old]
        children[pi].append(ci)
        outdeg[pi] += 1
        indeg[ci] += 1
        edge_pairs.append((pi, ci))

    depth = [0] * len(node_ids_f)
    q = deque([new_root_idx])
    seen = {new_root_idx}
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            if v in seen:
                continue
            depth[v] = depth[u] + 1
            seen.add(v)
            q.append(v)

    subtree = [1] * len(node_ids_f)
    stack = [new_root_idx]
    order = []
    visited = set()
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        for v in children.get(u, []):
            if v not in visited:
                stack.append(v)

    for u in reversed(order):
        s = 1
        for v in children.get(u, []):
            s += subtree[v]
        subtree[u] = s

    N = len(node_ids_f)


    wiener = 0.0
    for (p, c) in edge_pairs:
        s = float(subtree[c])
        wiener += s * float(N - s)

    virality = (2.0 * wiener) / (float(N) * float(N - 1)) if N > 1 else 0.0
    is_root = [0.0] * len(node_ids_f)
    is_root[new_root_idx] = 1.0
    is_leaf = [1.0 if outdeg[i] == 0 else 0.0 for i in range(len(node_ids_f))]

    x = torch.tensor(
        [
            [
                float(depth[i]),
                math.log1p(float(dt_sec_f[i])),
                float(outdeg[i]),
                float(indeg[i]),
                math.log1p(float(subtree[i])),
                float(is_root[i]),
                float(is_leaf[i]),
            ]
            for i in range(len(node_ids_f))
        ],
        dtype=torch.float32,
    )

    if len(edge_pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        E = 0
    else:
        src = [p for (p, c) in edge_pairs]
        dst = [c for (p, c) in edge_pairs]
        E = len(src)
        if make_undirected:
            src2 = src + dst
            dst2 = dst + src
            edge_index = torch.tensor([src2, dst2], dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)

    max_depth = max(depth) if depth else 0
    mean_depth = float(np.mean(depth)) if depth else 0.0
    root_children = float(outdeg[new_root_idx]) if N > 0 else 0.0
    leaf_ratio = float(np.mean(is_leaf)) if is_leaf else 0.0
    mean_out = float(np.mean(outdeg)) if outdeg else 0.0
    gini_out = gini_coef(outdeg)
    time_span = float(max(dt_sec_f)) if dt_sec_f else 0.0
    resp = [d for i, d in enumerate(dt_sec_f) if i != new_root_idx]
    med_resp = float(np.median(resp)) if resp else 0.0

    gfeat = torch.tensor(
        [
            math.log1p(float(N)),
            math.log1p(float(E)),
            math.log1p(float(max_depth)),
            math.log1p(float(mean_depth)),
            math.log1p(float(root_children)),
            float(leaf_ratio),
            math.log1p(float(mean_out)),
            float(gini_out),
            math.log1p(float(time_span)),
            math.log1p(float(med_resp)),
            math.log1p(float(virality)),  
            *hist_log,
        ],
        dtype=torch.float32,
    )

    y = torch.tensor(int(thread["y"]), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.thread_id = thread["thread_id"]
    data.event = thread["event"]
    data.root_idx = torch.tensor(new_root_idx, dtype=torch.long)
    data.gfeat = gfeat.unsqueeze(0)
    return data

class ThreadGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        threads: List[dict],
        make_undirected: bool = True,
        norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,           
        gfeat_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,     
        early_minutes: int = -1,
        early_k: int = 0,
    ):
        self.threads = threads
        self.make_undirected = make_undirected
        self.norm = norm
        self.gfeat_norm = gfeat_norm
        self.early_minutes = early_minutes
        self.early_k = early_k

    def __len__(self):
        return len(self.threads)

    def __getitem__(self, idx):
        data = build_graph_from_thread(
            self.threads[idx],
            make_undirected=self.make_undirected,
            early_minutes=self.early_minutes,
            early_k=self.early_k,
        )
        if self.gfeat_norm is not None:
            gm, gs = self.gfeat_norm
            data.gfeat = (data.gfeat - gm) / gs
        if self.norm is not None:
            mean, std = self.norm
            data.x = (data.x - mean) / std
        return data

    
def fit_gfeat_normalizer(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    for i in range(len(dataset)):
        d = dataset[i]
        xs.append(d.gfeat.squeeze(0))
    X = torch.stack(xs, dim=0)   
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=True)
    std[std < 1e-6] = 1.0
    return mean, std

def fit_normalizer(dataset: ThreadGraphDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    count = 0
    mean = None
    M2 = None
    for i in range(len(dataset)):
        x = dataset[i].x
        if mean is None:
            mean = torch.zeros(x.size(1), dtype=torch.float64)
            M2 = torch.zeros(x.size(1), dtype=torch.float64)
        for row in x.to(torch.float64):
            count += 1
            delta = row - mean
            mean += delta / count
            delta2 = row - mean
            M2 += delta * delta2
    var = M2 / max(1, count - 1)
    std = torch.sqrt(var)
    std[std < 1e-6] = 1.0
    return mean.to(torch.float32), std.to(torch.float32)

class GraphSageClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        readout: str,
        gfeat_dim: int,
    ):
        super().__init__()
        assert num_layers >= 1
        self.readout = readout
        self.dropout = dropout
        self.gfeat_dim = gfeat_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        if readout == "mean":
            ro_dim = hidden_dim
        elif readout == "meanmax":
            ro_dim = hidden_dim * 2
        elif readout == "root":
            ro_dim = hidden_dim
        elif readout == "rootmeanmax":
            ro_dim = hidden_dim * 3
        else:
            raise ValueError("readout must be one of: mean, meanmax, root, rootmeanmax")

        ro_dim = ro_dim + gfeat_dim

        self.mlp = nn.Sequential(
            nn.Linear(ro_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, ln in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = ln(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        ptr = data.ptr
        root_local = data.root_idx
        root_global = ptr[:-1] + root_local
        root_emb = x[root_global]

        if self.readout == "mean":
            g = global_mean_pool(x, batch)
        elif self.readout == "meanmax":
            g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        elif self.readout == "root":
            g = root_emb
        else: 
            g = torch.cat([root_emb, global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)

        if hasattr(data, "gfeat"):
            gfeat = data.gfeat
            if gfeat.dim() == 1:
                gfeat = gfeat.view(g.size(0), -1)     
            elif gfeat.dim() == 2 and gfeat.size(0) != g.size(0):
                gfeat = gfeat.view(g.size(0), -1)
            g = torch.cat([g, gfeat.to(g.device)], dim=-1)


        return self.mlp(g)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        y = batch.y.cpu().numpy().tolist()
        ys.extend(y)
        ps.extend(pred)
    macro = f1_score(ys, ps, average="macro")
    acc = accuracy_score(ys, ps)
    return macro, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=str, default="data/processed/threads.jsonl")
    ap.add_argument("--splits", type=str, default="data/processed/splits_loeo.json")
    ap.add_argument("--fold", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--readout", type=str, default="rootmeanmax", choices=["mean", "meanmax", "root", "rootmeanmax"])

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)

    ap.add_argument("--undirected", action="store_true")

    ap.add_argument("--early_minutes", type=int, default=-1, help="keep only nodes within T minutes from root (plus root). -1 disables")
    ap.add_argument("--early_k", type=int, default=0, help="keep root + earliest K nodes by time (fallback / alternative)")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threads = load_threads_jsonl(Path(args.threads))
    splits = load_splits(Path(args.splits))
    if args.fold not in splits:
        raise ValueError(f"Fold '{args.fold}' not found. Available keys: {list(splits.keys())[:10]}...")

    fold = splits[args.fold]
    train_ids = set(fold["train_thread_ids"])
    val_ids = set(fold["val_thread_ids"])
    test_ids = set(fold["test_thread_ids"])

    train_threads = [t for t in threads if t["thread_id"] in train_ids]
    val_threads = [t for t in threads if t["thread_id"] in val_ids]
    test_threads = [t for t in threads if t["thread_id"] in test_ids]

    train_ds_raw = ThreadGraphDataset(
        train_threads,
        make_undirected=args.undirected,
        norm=None,
        gfeat_norm=None,
        early_minutes=args.early_minutes,
        early_k=args.early_k,
    )
    mean, std = fit_normalizer(train_ds_raw)
    gmean, gstd = fit_gfeat_normalizer(train_ds_raw)

    gmean = gmean.view(1, -1)
    gstd  = gstd.view(1, -1)

    train_ds = ThreadGraphDataset(
        train_threads,
        make_undirected=args.undirected,
        norm=(mean, std),
        gfeat_norm=(gmean, gstd),
        early_minutes=args.early_minutes,
        early_k=args.early_k,
    )
    val_ds = ThreadGraphDataset(
        val_threads,
        make_undirected=args.undirected,
        norm=(mean, std),
        gfeat_norm=(gmean, gstd),
        early_minutes=args.early_minutes,
        early_k=args.early_k,
    )
    test_ds = ThreadGraphDataset(
        test_threads,
        make_undirected=args.undirected,
        norm=(mean, std),
        gfeat_norm=(gmean, gstd),
        early_minutes=args.early_minutes,
        early_k=args.early_k,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    y_train = [int(t["y"]) for t in train_threads]
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    class_w = torch.tensor(weights, dtype=torch.float32).to(device)

    in_dim = 7
    gfeat_dim = train_ds[0].gfeat.size(-1)

    model = GraphSageClassifier(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
        readout=args.readout,
        gfeat_dim=gfeat_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-5)

    best_val = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y, weight=class_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())

        val_f1, val_acc = evaluate(model, val_loader, device)
        sched.step(val_f1)

        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 5 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"[epoch {epoch:03d}] loss={total_loss/ max(1,len(train_loader)):.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}")

        if bad >= args.patience:
            print(f"Early stop at epoch {epoch} (best val_f1={best_val:.4f})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_f1, test_acc = evaluate(model, test_loader, device)

    result = {
        "title": "Fuse-T GNN-only baseline (early + gfeat)",
        "fold": args.fold,
        "seed": args.seed,
        "readout": args.readout,
        "undirected": bool(args.undirected),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "early_minutes": args.early_minutes,
        "early_k": args.early_k,
        "best_val_macro_f1": float(best_val),
        "test_macro_f1": float(test_f1),
        "test_acc": float(test_acc),
        "train_class_counts": counts.tolist(),
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(
        {"state_dict": best_state, "mean": mean, "std": std, "config": vars(args)},
        out_dir / "best.pt"
    )
    print("Saved:", out_dir / "best.pt")
    print("Test macro-F1:", test_f1, "Test acc:", test_acc)


if __name__ == "__main__":
    main()
