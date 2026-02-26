import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


LABEL_MAP_4 = {
    "non-rumour": 0,
    "true": 1,
    "false": 2,
    "unverified": 3,
}


RUMOUR_DIR_CANDIDATES = ["rumours", "rumors"]
NONRUMOUR_DIR_CANDIDATES = ["non-rumours", "non-rumors"]


def read_json(path: Path):
    try:
        b = path.read_bytes()
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            s = b.decode("utf-8", errors="replace")
        return json.loads(s)
    except Exception:
        return None



def find_first_existing(parent: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = parent / n
        if p.exists() and p.is_dir():
            return p
    return None


def parse_created_at(tweet_obj: dict) -> Optional[str]:
    if not tweet_obj:
        return None
    return tweet_obj.get("created_at") or tweet_obj.get("createdAt") or tweet_obj.get("date")


def parse_text(tweet_obj: dict) -> str:
    if not tweet_obj:
        return ""
    return (
        tweet_obj.get("text")
        or tweet_obj.get("full_text")
        or tweet_obj.get("content")
        or ""
    )


def load_tweets(thread_dir: Path) -> Dict[str, dict]:
    tweets: Dict[str, dict] = {}

    source_dir = thread_dir / "source-tweets"
    if source_dir.exists():
        for p in source_dir.glob("*.json"):
            obj = read_json(p)
            if obj is None:
                continue
            tid = p.stem
            tweets[tid] = {
                "text": parse_text(obj),
                "created_at": parse_created_at(obj),
            }

    reactions_dir = thread_dir / "reactions"
    if reactions_dir.exists():
        for p in reactions_dir.glob("*.json"):
            obj = read_json(p)
            if obj is None:
                continue
            tid = p.stem
            tweets[tid] = {
                "text": parse_text(obj),
                "created_at": parse_created_at(obj),
            }

    return tweets


def edges_from_structure(struct_obj: dict) -> Tuple[str, List[Tuple[str, str]]]:
    if not struct_obj:
        return ("", [])

    if all(isinstance(v, list) for v in struct_obj.values()):
        parents = set(struct_obj.keys())
        children = set()
        edges = []
        for p, childs in struct_obj.items():
            for c in childs:
                edges.append((str(p), str(c)))
                children.add(str(c))
        roots = list(parents - children)
        root_id = roots[0] if roots else list(parents)[0]
        return (str(root_id), edges)

    if isinstance(struct_obj, dict):
        top_keys = list(struct_obj.keys())
        if len(top_keys) == 1 and isinstance(struct_obj[top_keys[0]], dict):
            root_id = str(top_keys[0])
            edges: List[Tuple[str, str]] = []

            def rec(parent: str, subtree: dict):
                if not isinstance(subtree, dict):
                    return
                for child, sub in subtree.items():
                    c = str(child)
                    edges.append((parent, c))
                    rec(c, sub)

            rec(root_id, struct_obj[top_keys[0]])
            return (root_id, edges)

    return ("", [])



def _to_int(v):
    if v is None: return None
    if isinstance(v, bool): return int(v)
    if isinstance(v, (int, float)): return int(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ["true","t","yes"]: return 1
        if s in ["false","f","no"]: return 0
        try: return int(float(s))
        except: return None
    return None


def parse_label(thread_dir: Path, is_rumour: bool) -> str:
    if not is_rumour:
        return "non-rumour"

    vpath = thread_dir / "veracity.json"
    if vpath.exists():
        v = read_json(vpath) or {}
        cand = (v.get("veracity") or v.get("label") or v.get("value") or "").lower()
        if cand in LABEL_MAP_4:
            return cand

    apath = thread_dir / "annotation.json"
    a = read_json(apath) if apath.exists() else None
    if isinstance(a, dict):
        t = _to_int(a.get("true"))
        m = _to_int(a.get("misinformation"))

        if t == 1:
            return "true"
        if m == 1:
            return "false"
        return "unverified"

    return "unverified"


def find_event_dirs(data_root: Path) -> List[Path]:
    candidates = []
    for p in data_root.rglob("*"):
        if p.is_dir():
            rd = find_first_existing(p, RUMOUR_DIR_CANDIDATES)
            nd = find_first_existing(p, NONRUMOUR_DIR_CANDIDATES)
            if rd or nd:
                candidates.append(p)
    candidates = sorted(set(candidates), key=lambda x: len(str(x)))
    filtered = []
    for c in candidates:
        if not any(str(c).startswith(str(prev) + os.sep) for prev in filtered):
            filtered.append(c)
    return filtered


def iter_threads(event_dir: Path):
    rumours_dir = find_first_existing(event_dir, RUMOUR_DIR_CANDIDATES)
    nonrumours_dir = find_first_existing(event_dir, NONRUMOUR_DIR_CANDIDATES)

    if rumours_dir:
        for thread_dir in sorted(rumours_dir.iterdir()):
            if thread_dir.is_dir():
                yield True, thread_dir

    if nonrumours_dir:
        for thread_dir in sorted(nonrumours_dir.iterdir()):
            if thread_dir.is_dir():
                yield False, thread_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, default="data/processed/threads.jsonl")
    ap.add_argument("--out_summary", type=str, default="data/processed/summary.json")
    ap.add_argument("--max_threads", type=int, default=0, help="0 = no limit")
    ap.add_argument("--label_mode", type=str, default="binary", choices=["binary", "4class"])
    ap.add_argument("--infer_missing_structure", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--infer_strategy", type=str, default="star", choices=["star"])

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    event_dirs = find_event_dirs(data_root)
    if not event_dirs:
        raise RuntimeError(f"No event directories found under {data_root}. Check your unzip path.")

    total = 0
    counts_by_label = {k: 0 for k in LABEL_MAP_4.keys()}
    counts_by_event = {}

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for event_dir in event_dirs:
            event_name = event_dir.name
            counts_by_event.setdefault(event_name, 0)

            threads = list(iter_threads(event_dir))
            if not threads:
                continue

            for is_rumour, thread_dir in tqdm(threads, desc=f"event={event_name}", leave=False):
                if args.max_threads and total >= args.max_threads:
                    break

                struct_path = thread_dir / "structure.json"
                struct_obj = read_json(struct_path) if struct_path.exists() else None
                root_id, edges = edges_from_structure(struct_obj or {})

                tweets = load_tweets(thread_dir)

                if not root_id:
                    source_dir = thread_dir / "source-tweets"
                    if source_dir.exists():
                        src_files = list(source_dir.glob("*.json"))
                        if src_files:
                            root_id = src_files[0].stem

                label_str = parse_label(thread_dir, is_rumour=is_rumour)
                y4 = LABEL_MAP_4[label_str]
                if args.label_mode == "binary":
                    y = 0 if label_str == "non-rumour" else 1
                else:
                    y = y4
                node_ids = set(tweets.keys())
                for s, t in edges:
                    node_ids.add(s)
                    node_ids.add(t)
                if root_id:
                    node_ids.add(root_id)

                edges_inferred = False
                if args.infer_missing_structure and len(edges) == 0 and root_id and len(node_ids) > 1:
                    if args.infer_strategy == "star":
                        edges = [(str(root_id), str(nid)) for nid in node_ids if str(nid) != str(root_id)]
                        edges_inferred = True

                nodes = []
                for nid in sorted(node_ids):
                    info = tweets.get(nid, {"text": "", "created_at": None})
                    nodes.append({
                        "id": nid,
                        "text": info.get("text", ""),
                        "created_at": info.get("created_at", None),
                    })

                record = {
                    "thread_id": thread_dir.name,  
                    "event": event_name,
                    "is_rumour_dir": bool(is_rumour),
                    "label_str": label_str,
                    "y":int(y),
                    "y4": int(y4),
                    "root_id": root_id,
                    "edges": edges,      
                    "edges_inferred": bool(edges_inferred),
                    "nodes": nodes,     
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                total += 1
                counts_by_label[label_str] += 1
                counts_by_event[event_name] += 1

            if args.max_threads and total >= args.max_threads:
                break

    summary = {
        "total_threads": total,
        "label_map": LABEL_MAP_4,
        "counts_by_label": counts_by_label,
        "counts_by_event": counts_by_event,
        "event_dirs_detected": [str(p) for p in event_dirs],
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote:", out_jsonl)
    print("Summary:", out_summary)
    print("Total threads:", total)
    print("Counts by label:", counts_by_label)
    print("Counts by event:", counts_by_event)


if __name__ == "__main__":
    main()
