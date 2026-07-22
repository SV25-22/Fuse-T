"""Run the PHEME preprocessing and LOEO split pipeline on synthetic threads."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_thread(event: Path, label_dir: str, index: int) -> None:
    root_id = f"{event.name}-{label_dir}-{index}"
    reply_id = f"{root_id}-reply"
    thread = event / label_dir / root_id

    _write_json(
        thread / "source-tweets" / f"{root_id}.json",
        {
            "text": f"@Reporter Breaking NEWS {index} https://example.test/{index}",
            "created_at": "Mon Jan 01 12:00:00 +0000 2024",
        },
    )
    _write_json(
        thread / "reactions" / f"{reply_id}.json",
        {
            "text": "A follow-up reply",
            "created_at": "Mon Jan 01 12:01:00 +0000 2024",
        },
    )
    _write_json(thread / "structure.json", {root_id: {reply_id: []}})
    if label_dir == "rumours":
        _write_json(thread / "annotation.json", {"misinformation": 1, "true": 0})


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fuse-t-smoke-") as tmp:
        work = Path(tmp)
        raw = work / "raw" / "PHEME_veracity"
        processed = work / "processed"
        threads_path = processed / "threads.jsonl"
        summary_path = processed / "summary.json"
        splits_path = processed / "splits_loeo.json"

        for event_name in ("event-a-all-rnr-threads", "event-b-all-rnr-threads"):
            event = raw / event_name
            for label_dir in ("rumours", "non-rumours"):
                for index in range(2):
                    _make_thread(event, label_dir, index)

        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "preprocess_pheme.py"),
                "--data_root",
                str(raw),
                "--out_jsonl",
                str(threads_path),
                "--out_summary",
                str(summary_path),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "make_splits_loeo.py"),
                "--threads_jsonl",
                str(threads_path),
                "--out",
                str(splits_path),
                "--val_frac",
                "0.5",
                "--min_test_n",
                "2",
                "--min_test_pos",
                "1",
                "--min_test_neg",
                "1",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        records = [json.loads(line) for line in threads_path.read_text(encoding="utf-8").splitlines()]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        splits = json.loads(splits_path.read_text(encoding="utf-8"))

        assert len(records) == 8
        assert summary["total_threads"] == 8
        assert summary["text_normalization"] == {
            "lowercase": True,
            "strip_urls": True,
            "strip_mentions": True,
        }
        assert len(splits) == 2
        assert all(len(fold["test_thread_ids"]) == 4 for fold in splits.values())
        assert records[0]["nodes"][0]["text"] == "breaking news 0"

    print("Smoke test passed: 8 synthetic threads, 2 LOEO folds.")


if __name__ == "__main__":
    main()
