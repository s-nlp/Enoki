from __future__ import annotations

import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from evaluation.metrics import (
    calculate_span_f1,
    calculate_span_f1_exact_match,
    print_span_metrics_summary,
)
from evaluation.span_coverage_report import main as build_span_report
from evaluation.span_metrics import span_iou_macro, span_iou_one

try:
    from evaluation.predictions_io import compute_span_threshold_curves
except ModuleNotFoundError:  # NumPy is part of the benchmark extra, not core metrics.
    compute_span_threshold_curves = None


class SpanIoUTest(unittest.TestCase):
    def test_matches_participant_kit_half_open_semantics(self):
        self.assertAlmostEqual(span_iou_one([[0, 4]], [[2, 6]]), 2 / 6)

    def test_merges_overlapping_spans(self):
        self.assertEqual(span_iou_one([[0, 3], [2, 5]], [[1, 4]]), 3 / 5)

    def test_empty_gold_and_prediction_is_perfect(self):
        self.assertEqual(span_iou_one([], []), 1.0)

    def test_one_empty_side_has_zero_iou(self):
        self.assertEqual(span_iou_one([[0, 3]], []), 0.0)
        self.assertEqual(span_iou_one([], [[0, 3]]), 0.0)

    def test_macro_is_mean_per_example(self):
        self.assertEqual(span_iou_macro([[], [[0, 2]]], [[], [[4, 6]]]), 0.5)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            span_iou_macro([[]], [])

    def test_span_metrics_report_iou_after_coverage_f1(self):
        metrics = calculate_span_f1([[[0, 4]]], [[[2, 6]]])
        self.assertAlmostEqual(metrics["iou"], 2 / 6)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_span_metrics_summary(metrics)
        lines = output.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("Span Coverage F1:"))
        self.assertTrue(lines[1].startswith("Mean IoU:"))

    def test_legacy_exact_match_summary_still_works(self):
        metrics = calculate_span_f1_exact_match([[[0, 4]]], [[[0, 4]]])
        with contextlib.redirect_stdout(io.StringIO()):
            print_span_metrics_summary(metrics)

    @unittest.skipIf(compute_span_threshold_curves is None, "benchmark dependencies absent")
    def test_threshold_curve_carries_iou_without_using_it_as_primary(self):
        rows = compute_span_threshold_curves(
            [
                {
                    "gold_spans": [[0, 4]],
                    "fact_spans": [
                        {"start": 2, "end": 6, "neutral": 0.8, "contradiction": 0.0}
                    ],
                }
            ],
            n_thresholds=2,
        )
        self.assertTrue(rows)
        self.assertTrue(all("iou" in row for row in rows))

    def test_report_places_iou_after_span_coverage_f1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pred_dir = root / "predictions"
            pred_dir.mkdir()
            with (pred_dir / "demo_mushroom.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["gold", "pred"])
                writer.writeheader()
                writer.writerow({"gold": "[[0, 4]]", "pred": "[[2, 6]]"})

            out_dir = root / "reports"
            argv = [
                "span_coverage_report.py",
                "--pred-dir",
                str(pred_dir),
                "--out-dir",
                str(out_dir),
                "--main-table",
                "--latex",
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    build_span_report()

            dataset_table = (out_dir / "span_coverage_mushroom.md").read_text()
            main_table = (out_dir / "span_coverage_main_table.md").read_text()
            latex_table = (out_dir / "span_coverage_main_table.tex").read_text()
            self.assertIn("F1_micro@d5 | IoU", dataset_table)
            self.assertIn("mushroom_micro_F1@d0 | mushroom_micro_IoU", main_table)
            self.assertIn("P & R & F1 & IoU", latex_table)


if __name__ == "__main__":
    unittest.main()
