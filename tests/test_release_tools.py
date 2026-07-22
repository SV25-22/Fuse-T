from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_results import load_results, render_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseToolsTest(unittest.TestCase):
    def test_synthetic_preprocessing_smoke_test(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "smoke_test.py")],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Smoke test passed", completed.stdout)

    def test_result_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fuse-t-results-") as tmp:
            root = Path(tmp)
            for model, values in {
                "text": {"event-a": 0.5, "event-b": 0.7},
                "fuse_t": {"event-a": 0.6, "event-b": 0.8},
            }.items():
                for fold, value in values.items():
                    path = root / model / fold / "result.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps({"fold": fold, "test_macro_f1": value}),
                        encoding="utf-8",
                    )

            results = load_results(root, ["text", "fuse_t"], "test_macro_f1")
            table = render_markdown(results, ["text", "fuse_t"], strict=True)
            self.assertIn("| event-a | 50.00 | 60.00 |", table)
            self.assertIn("| **Average** | 60.00 | 70.00 |", table)


if __name__ == "__main__":
    unittest.main()
