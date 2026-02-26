import argparse
import json
import math
import random
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool
except Exception as e:
    raise RuntimeError("torch_geometric not available. Install PyG correctly.") from e

TWITTER_FMT = "%a %b %d %H:%M:%S %z %Y"

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

def gini_coef(values: List[float]) -> float:
    if not values: return 0.0
    x = np.asarray(values, dtype=np.float64)
    if np.all(x == 0): return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.sum((2 * np.arange(1, n + 1) - n - 1) * x)
    denom = n * np.sum(x)
    return 0.0 if denom == 0 else float(cum / denom)

def parse_ts(created_at: Optional[str]) -> Optional[int]:
    if not created_at: return None
    s = str(created_at).strip()
    if not s: return None
    if du_parser is not None:
        try: return int(du_parser.parse(s).timestamp())
        except Exception: pass
    for fmt in [TWITTER_FMT, "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"]:
        try: return int(datetime.strptime(s, fmt).timestamp())
        except Exception: continue
    try: return int(datetime.fromisoformat(s).timestamp())
    except Exception: return None

def load_threads_jsonl(path: Path) -> List[dict]:
    threads = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    return threads

def load_splits(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

class TextModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
def precompute_node_embeddings(threads: List[dict], tokenizer, txt_model, device, batch_size=128) -> Dict[str, torch.Tensor]:
    txt_model.eval()
    all_nodes = {}
    for t in threads:
        for n in t["nodes"]:
            if n["id"] not in all_nodes:
                all_nodes[n["id"]] = n.get("text", "")
    
    node_ids = list(all_nodes.keys())
    node_embeds = {}
    
    print(f"Pre-computing embeddings for {len(node_ids)} unique nodes...")
    with torch.no_grad():
        for i in tqdm(range(0, len(node_ids), batch_size), leave=False, desc="Encoding Nodes"):
            b_ids = node_ids[i:i+batch_size]
            b_texts = [all_nodes[nid] for nid in b_ids]
            
            enc = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            
            out = txt_model.encoder(**enc)
            cls = out.last_hidden_state[:, 0, :]
            
            for j, nid in enumerate(b_ids):
                node_embeds[nid] = cls[j].cpu().clone()
                
    return node_embeds

def build_graph_from_thread(thread: dict, make_undirected: bool = True, early_minutes: int = -1, early_k: int = 0, node_embeds: dict = None) -> Data:
    nodes = thread["nodes"]
    edges = thread["edges"]
    root_id = thread.get("root_id") or thread.get("thread_id")

    node_ids = [n["id"] for n in nodes]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    root_idx = id2idx[root_id] if root_id in id2idx else 0

    ts = [parse_ts(n.get("created_at")) for n in nodes]
    root_ts = ts[root_idx] if (0 <= root_idx < len(ts) and ts[root_idx] is not None) else None

    dt_sec: List[Optional[float]] = []
    for i, t in enumerate(ts):
        if i == root_idx: dt_sec.append(0.0)
        elif root_ts is None or t is None: dt_sec.append(None)
        else: dt_sec.append(max(0.0, float(t - root_ts)))

    keep = set(range(len(nodes)))
    if early_minutes and early_minutes > 0 and root_ts is not None:
        limit = float(early_minutes) * 60.0
        keep = {root_idx}
        for i, d in enumerate(dt_sec):
            if i == root_idx: continue
            if d is not None and d <= limit: keep.add(i)

        if len(keep) < 2 and early_k and early_k > 0:
            order = [((dt_sec[i] if dt_sec[i] is not None else 1e18), i) for i in range(len(nodes))]
            order.sort()
            keep = {root_idx}
            for _, i in order:
                if i == root_idx: continue
                keep.add(i)
                if len(keep) >= 1 + early_k: break
    elif early_k and early_k > 0:
        order = [((ts[i] if ts[i] is not None else 1e18), i) for i in range(len(nodes))]
        order.sort()
        keep = {root_idx}
        for _, i in order:
            if i == root_idx: continue
            keep.add(i)
            if len(keep) >= 1 + early_k: break

    keep_list = sorted(list(keep))
    old2new = {old: new for new, old in enumerate(keep_list)}
    new_root_idx = old2new.get(root_idx, 0)
    node_ids_f = [node_ids[i] for i in keep_list]
    N = len(node_ids_f)

    dt_sec_f: List[float] = []
    for old in keep_list:
        d = dt_sec[old] if 0 <= old < len(dt_sec) else None
        if old == root_idx: dt_sec_f.append(0.0)
        else: dt_sec_f.append(float(d) if d is not None else 0.0)

    children = defaultdict(list)
    indeg = [0] * N
    outdeg = [0] * N
    edge_pairs: List[Tuple[int, int]] = []
    for e in edges:
        p, c = e[0], e[1]
        if p not in id2idx or c not in id2idx: continue
        pi_old, ci_old = id2idx[p], id2idx[c]
        if pi_old not in keep or ci_old not in keep: continue
        pi, ci = old2new[pi_old], old2new[ci_old]
        children[pi].append(ci)
        outdeg[pi] += 1
        indeg[ci] += 1
        edge_pairs.append((pi, ci))

    depth = [0] * N
    q = deque([new_root_idx])
    seen = {new_root_idx}
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            if v in seen: continue
            depth[v] = depth[u] + 1
            seen.add(v)
            q.append(v)

    subtree = [1] * N
    stack = [new_root_idx]
    order = []
    visited = set()
    while stack:
        u = stack.pop()
        if u in visited: continue
        visited.add(u)
        order.append(u)
        for v in children.get(u, []):
            if v not in visited: stack.append(v)
    for u in reversed(order):
        s = 1
        for v in children.get(u, []):
            s += subtree[v]
        subtree[u] = s

    is_root = [0.0] * N
    if 0 <= new_root_idx < N: is_root[new_root_idx] = 1.0
    is_leaf = [1.0 if outdeg[i] == 0 else 0.0 for i in range(N)]

    x_struct = torch.tensor(
        [
            [
                float(depth[i]), math.log1p(float(dt_sec_f[i])),
                float(outdeg[i]), float(indeg[i]),
                math.log1p(float(subtree[i])),
                float(is_root[i]), float(is_leaf[i]),
            ] for i in range(N)
        ], dtype=torch.float32,
    )

    if node_embeds is not None:
        x_text = torch.stack([node_embeds[node_ids_f[i]] for i in range(N)])
        x = torch.cat([x_struct, x_text], dim=-1)
    else:
        x = x_struct

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

    wiener = 0.0
    for (p, c) in edge_pairs:
        s = float(subtree[c])
        wiener += s * float(N - s)
    virality = (2.0 * wiener) / (float(N) * float(N - 1)) if N > 1 else 0.0

    bins = [60, 2 * 60, 5 * 60, 10 * 60, 30 * 60, 60 * 60]
    hist = [0] * (len(bins) + 1)
    for i, d in enumerate(dt_sec_f):
        if i == new_root_idx: continue
        b = 0
        while b < len(bins) and d > bins[b]: b += 1
        hist[b] += 1
    hist_log = [math.log1p(h) for h in hist]

    gfeat = torch.tensor(
        [
            math.log1p(float(N)), math.log1p(float(E)), math.log1p(float(max_depth)),
            math.log1p(float(mean_depth)), math.log1p(float(root_children)), float(leaf_ratio),
            math.log1p(float(mean_out)), float(gini_out), math.log1p(float(time_span)),
            math.log1p(float(med_resp)), math.log1p(float(virality)), *hist_log,
        ], dtype=torch.float32,
    ).unsqueeze(0)

    data = Data(x=x, edge_index=edge_index, y=torch.tensor(int(thread["y"]), dtype=torch.long))
    data.thread_id = thread["thread_id"]
    data.event = thread["event"]
    data.root_idx = torch.tensor(new_root_idx, dtype=torch.long)
    data.gfeat = gfeat
    return data

class TAGDataset(torch.utils.data.Dataset):
    def __init__(self, threads, undirected, norm, gfeat_norm, early_minutes, early_k, node_embeds):
        self.threads = threads
        self.undirected = undirected
        self.norm = norm
        self.gfeat_norm = gfeat_norm
        self.early_minutes = early_minutes
        self.early_k = early_k
        self.node_embeds = node_embeds

    def __len__(self):
        return len(self.threads)

    def __getitem__(self, idx):
        t = self.threads[idx]
        data = build_graph_from_thread(
            t, make_undirected=self.undirected, early_minutes=self.early_minutes,
            early_k=self.early_k, node_embeds=self.node_embeds
        )
        if self.gfeat_norm is not None:
            gm, gs = self.gfeat_norm
            data.gfeat = (data.gfeat - gm) / gs
        if self.norm is not None:
            mean, std = self.norm
            data.x[:, :7] = (data.x[:, :7] - mean) / std
        return data

def fit_normalizer(threads, undirected, early_minutes, early_k) -> Tuple[torch.Tensor, torch.Tensor]:
    count = 0
    mean = torch.zeros(7, dtype=torch.float64)
    M2 = torch.zeros(7, dtype=torch.float64)
    for t in threads:
        d = build_graph_from_thread(t, make_undirected=undirected, early_minutes=early_minutes, early_k=early_k, node_embeds=None)
        x = d.x.to(torch.float64)
        for row in x:
            count += 1
            delta = row - mean
            mean += delta / count
            delta2 = row - mean
            M2 += delta * delta2
    var = M2 / max(1, count - 1)
    std = torch.sqrt(var)
    std[std < 1e-6] = 1.0
    return mean.to(torch.float32), std.to(torch.float32)

def fit_gfeat_normalizer(threads, undirected, early_minutes, early_k) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    for t in threads:
        d = build_graph_from_thread(t, make_undirected=undirected, early_minutes=early_minutes, early_k=early_k, node_embeds=None)
        xs.append(d.gfeat.squeeze(0))
    X = torch.stack(xs, dim=0)
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=True)
    std[std < 1e-6] = 1.0
    return mean.view(1, -1).to(torch.float32), std.view(1, -1).to(torch.float32)

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, readout, gfeat_dim):
        super().__init__()
        self.readout = readout
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.readout == "mean": g = global_mean_pool(x, batch)
        elif self.readout == "meanmax": g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        elif self.readout == "root":
            root_global = data.ptr[:-1] + data.root_idx
            g = x[root_global]

        if hasattr(data, "gfeat"):
            gf = data.gfeat.view(g.size(0), -1)
            g = torch.cat([g, gf.to(g.device)], dim=-1)
        return g

class GraphHead(nn.Module):
    def __init__(self, graph_dim, dropout, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(graph_dim),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, num_classes),
        )
    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        return self.net(h_graph)

@torch.no_grad()
def evaluate(gnn_enc, g_head, loader, device):
    gnn_enc.eval(); g_head.eval()
    ys, pg = [], []
    for batch in loader:
        batch = batch.to(device)
        y = batch.y
        h_graph = gnn_enc(batch)
        logits_graph = g_head(h_graph)
        ys.extend(y.cpu().tolist())
        pg.extend(logits_graph.argmax(-1).cpu().tolist())
    macro = f1_score(ys, pg, average="macro")
    acc = accuracy_score(ys, pg)
    return macro, acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=str, default="data/processed/threads.jsonl")
    ap.add_argument("--splits", type=str, default="data/processed/splits_loeo.json")
    ap.add_argument("--fold", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_name", type=str, default="roberta-base")
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--readout", type=str, default="meanmax", choices=["mean", "meanmax", "root"])
    ap.add_argument("--early_minutes", type=int, default=-1)
    ap.add_argument("--early_k", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--init_text_ckpt", type=str, required=True)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threads = load_threads_jsonl(Path(args.threads))
    splits = load_splits(Path(args.splits))
    fold = splits[args.fold]

    train_ids = set(fold["train_thread_ids"])
    val_ids = set(fold["val_thread_ids"])
    test_ids = set(fold["test_thread_ids"])

    train_threads = [t for t in threads if t["thread_id"] in train_ids]
    val_threads = [t for t in threads if t["thread_id"] in val_ids]
    test_threads = [t for t in threads if t["thread_id"] in test_ids]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    txt = TextModel(args.model_name).to(device)
    ck = torch.load(args.init_text_ckpt, map_location="cpu")
    txt.load_state_dict(ck.get("state_dict", ck), strict=False)
    
    all_threads = train_threads + val_threads + test_threads
    node_embeds = precompute_node_embeddings(all_threads, tokenizer, txt, device)
    
    del txt, tokenizer
    torch.cuda.empty_cache()

    mean, std = fit_normalizer(train_threads, args.undirected, args.early_minutes, args.early_k)
    gmean, gstd = fit_gfeat_normalizer(train_threads, args.undirected, args.early_minutes, args.early_k)

    train_ds = TAGDataset(train_threads, args.undirected, (mean, std), (gmean, gstd), args.early_minutes, args.early_k, node_embeds)
    val_ds = TAGDataset(val_threads, args.undirected, (mean, std), (gmean, gstd), args.early_minutes, args.early_k, node_embeds)
    test_ds = TAGDataset(test_threads, args.undirected, (mean, std), (gmean, gstd), args.early_minutes, args.early_k, node_embeds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    y_train = np.array([t["y"] for t in train_threads], dtype=np.int64)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    class_w = torch.tensor(weights / weights.mean(), dtype=torch.float32).to(device)

    text_dim = 768
    in_dim = 7 + text_dim
    ro = args.hidden_dim if args.readout != "meanmax" else args.hidden_dim * 2
    graph_out_dim = ro + train_ds[0].gfeat.size(-1)

    gnn_enc = GraphEncoder(in_dim, args.hidden_dim, args.num_layers, args.dropout, args.readout, gfeat_dim=train_ds[0].gfeat.size(-1)).to(device)
    g_head = GraphHead(graph_dim=graph_out_dim, dropout=args.dropout, num_classes=2).to(device)

    opt = AdamW(list(gnn_enc.parameters()) + list(g_head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)

    best_val = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        gnn_enc.train(); g_head.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            batch = batch.to(device)
            y = batch.y.to(device)

            h_graph = gnn_enc(batch)
            logits = g_head(h_graph)
            loss = F.cross_entropy(logits, y, weight=class_w)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(gnn_enc.parameters()) + list(g_head.parameters()), 1.0)
            opt.step()
            sched.step()
            total_loss += float(loss.item())

        val_f1, val_acc = evaluate(gnn_enc, g_head, val_loader, device)
        print(f"[epoch {epoch:02d}] loss={total_loss/max(1,len(train_loader)):.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f}")

        if val_f1 > best_val:
            best_val = val_f1
            best_state = {
                "gnn_enc": {k: v.detach().cpu() for k, v in gnn_enc.state_dict().items()},
                "g_head": {k: v.detach().cpu() for k, v in g_head.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop (best val_f1={best_val:.4f})")
                break

    gnn_enc.load_state_dict(best_state["gnn_enc"])
    g_head.load_state_dict(best_state["g_head"])
    test_f1, test_acc = evaluate(gnn_enc, g_head, test_loader, device)

    result = {
        "title": "Text-Attributed GNN Baseline",
        "fold": args.fold,
        "best_val_macro_f1": float(best_val),
        "test_macro_f1": float(test_f1),
        "test_acc": float(test_acc),
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(best_state, out_dir / "best.pt")
    print(f"Test macro-F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()