import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.model_selection import train_test_split


def load_threads(jsonl_path: Path) -> List[dict]:
    threads = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    return threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads_jsonl", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/processed/splits_loeo.json")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_test_n", type=int, default=50)
    ap.add_argument("--min_test_pos", type=int, default=10)
    ap.add_argument("--min_test_neg", type=int, default=10)
    ap.add_argument("--max_inferred_ratio", type=float, default=0.6)
    ap.add_argument("--exclude_events", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)

    threads = load_threads(Path(args.threads_jsonl))
    if not threads:
        raise RuntimeError("threads.jsonl is empty.")

    # group by event
    by_event: Dict[str, List[dict]] = defaultdict(list)
    for t in threads:
        by_event[t["event"]].append(t)


    excluded = set([e.strip() for e in args.exclude_events.split(",") if e.strip()])

    def is_good_event(ev: str, threads_ev: List[dict]) -> Tuple[bool, str]:
        n = len(threads_ev)
        ys = [int(t["y"]) for t in threads_ev]
        pos = sum(1 for y in ys if y == 1)
        neg = sum(1 for y in ys if y == 0)

        multi = [t for t in threads_ev if len(t.get("nodes", [])) > 1]
        if multi:
            inferred = sum(1 for t in multi if t.get("edges_inferred", False))
            inferred_ratio = inferred / len(multi)
        else:
            inferred_ratio = 1.0

        if ev in excluded:
            return False, "manually excluded"
        if n < args.min_test_n:
            return False, f"test_n={n} < {args.min_test_n}"
        if pos < args.min_test_pos:
            return False, f"pos={pos} < {args.min_test_pos}"
        if neg < args.min_test_neg:
            return False, f"neg={neg} < {args.min_test_neg}"
        if inferred_ratio > args.max_inferred_ratio:
            return False, f"inferred_ratio={inferred_ratio:.2f} > {args.max_inferred_ratio}"
        return True, "ok"

    good_events = []
    bad_reasons = {}
    for ev in sorted(by_event.keys()):
        ok, reason = is_good_event(ev, by_event[ev])
        if ok:
            good_events.append(ev)
        else:
            bad_reasons[ev] = reason

    print("Good events:", good_events)
    print("Dropped events:")
    for ev, r in bad_reasons.items():
        print(f"  - {ev}: {r}")

    events = good_events
    splits = {}

    for test_event in events:
        test_threads = by_event[test_event]
        train_pool = [t for e in events if e != test_event for t in by_event[e]]

        train_ids = [t["thread_id"] for t in train_pool]
        train_y = [t["y"] for t in train_pool]

        try:
            tr_ids, va_ids = train_test_split(
                train_ids,
                test_size=args.val_frac,
                random_state=args.seed,
                shuffle=True,
                stratify=train_y,
            )
        except Exception:
            tr_ids, va_ids = train_test_split(
                train_ids,
                test_size=args.val_frac,
                random_state=args.seed,
                shuffle=True,
                stratify=None,
            )

        fold = {
            "test_event": test_event,
            "train_events": [e for e in events if e != test_event],
            "test_events": [test_event],
            "train_thread_ids": tr_ids,
            "val_thread_ids": va_ids,
            "test_thread_ids": [t["thread_id"] for t in test_threads],
        }

        id_to_y = {t["thread_id"]: t["y"] for t in threads}

        def dist(ids):
            c = Counter(id_to_y[i] for i in ids if i in id_to_y)
            return dict(sorted(c.items(), key=lambda x: x[0]))

        fold["train_label_dist"] = dist(fold["train_thread_ids"])
        fold["val_label_dist"] = dist(fold["val_thread_ids"])
        fold["test_label_dist"] = dist(fold["test_thread_ids"])

        splits[test_event] = fold

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print("Wrote:", out_path)
    print("Folds:", len(splits))
    print("Events:", events)


if __name__ == "__main__":
    main()
