"""Summarize per-fold result.json files as a Markdown LOEO table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


MODEL_LABELS = {
    "text": "Text",
    "gnn": "GNN",
    "tag_gnn": "TAG-GNN",
    "fuse_t": "Fuse-T",
}


def load_results(results_root: Path, models: list[str], metric: str) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for model in models:
        model_results: dict[str, float] = {}
        for path in sorted((results_root / model).glob("*/result.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if metric not in record:
                raise KeyError(f"{path} does not contain '{metric}'")
            fold = str(record.get("fold") or path.parent.name)
            model_results[fold] = float(record[metric])
        results[model] = model_results
    return results


def render_markdown(
    results: dict[str, dict[str, float]], models: list[str], *, strict: bool = False
) -> str:
    events = sorted({event for model_results in results.values() for event in model_results})
    if not events:
        raise ValueError("No result.json files were found.")

    if strict:
        missing = [
            f"{model}/{event}"
            for model in models
            for event in events
            if event not in results[model]
        ]
        if missing:
            raise ValueError("Missing results: " + ", ".join(missing))

    labels = [MODEL_LABELS.get(model, model) for model in models]
    lines = [
        "| Held-out event | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(models),
    ]
    for event in events:
        cells = []
        for model in models:
            value = results[model].get(event)
            cells.append("—" if value is None else f"{100.0 * value:.2f}")
        lines.append(f"| {event} | " + " | ".join(cells) + " |")

    averages = []
    for model in models:
        values = list(results[model].values())
        averages.append("—" if not values else f"{100.0 * fmean(values):.2f}")
    lines.append("| **Average** | " + " | ".join(averages) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["text", "gnn", "tag_gnn", "fuse_t"],
    )
    parser.add_argument("--metric", default="test_macro_f1")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = load_results(args.results_root, args.models, args.metric)
    table = render_markdown(results, args.models, strict=args.strict) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
