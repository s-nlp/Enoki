from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Tuple

from evaluation.span_metrics import span_coverage_macro, span_coverage_micro, span_iou_macro


DATASETS = ("mushroom", "ragtruth", "psiloqa")
DELTAS = tuple(range(0, 6))


@dataclass(frozen=True)
class RunKey:
    method: str
    dataset: str


def _parse_run_key(path: Path) -> RunKey:
    """
    Parse filename to extract method and dataset.

    Format: method_dataset.csv
    Where method can include modifiers like: modernbert_coref, alignscore_default, etc.
    """
    name = path.stem

    # Try dataset at end: method_dataset
    for ds in DATASETS:
        suffix = f"_{ds}"
        if name.endswith(suffix):
            method = name[: -len(suffix)]
            if not method:
                raise ValueError(f"Empty method name in file: {path}")
            return RunKey(method=method, dataset=ds)
    # Try dataset in middle: prefix_dataset_extractor → method = prefix_extractor
    for ds in DATASETS:
        marker = f"_{ds}_"
        idx = name.find(marker)
        if idx != -1:
            prefix = name[:idx]
            suffix = name[idx + len(marker):]
            if not prefix:
                raise ValueError(f"Empty method prefix in file: {path}")
            method = f"{prefix}_{suffix}" if suffix else prefix
            return RunKey(method=method, dataset=ds)
    raise ValueError(f"Can't infer dataset from filename (expected method_{{{','.join(DATASETS)}}}.csv): {path}")


def _read_gold_pred_csv(path: Path) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    golds: List[List[List[int]]] = []
    preds: List[List[List[int]]] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "gold" not in reader.fieldnames or "pred" not in reader.fieldnames:
            raise ValueError(f"CSV must have 'gold' and 'pred' columns: {path}")
        for row in reader:
            gold = ast.literal_eval(row["gold"])
            pred = ast.literal_eval(row["pred"])
            if not isinstance(gold, list) or not isinstance(pred, list):
                raise ValueError(f"Expected list values in gold/pred columns: {path}")
            golds.append(gold)
            preds.append(pred)

    return golds, preds


def _fmt_f1(x: float) -> str:
    return f"{x:.4f}"

def _fmt_metric(x: float) -> str:
    return f"{x:.4f}"

def _render_latex_table(rows: List[Dict[str, str]], columns: List[str], *, caption: str = "", label: str = "") -> str:
    """
    Minimal LaTeX table generator (tabular).
    - Escapes '_' and '%'.
    - Right-aligns numeric columns, left-aligns the first column.
    """
    def esc(s: str) -> str:
        return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")

    align = "l" + ("r" * max(0, len(columns) - 1))
    header = " & ".join(esc(c) for c in columns) + " \\\\"
    lines = [f"\\begin{{table}}[t]", "\\centering", f"\\begin{{tabular}}{{{align}}}", "\\hline", header, "\\hline"]
    for r in rows:
        lines.append(" & ".join(esc(str(r.get(c, ""))) for c in columns) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    if caption:
        lines.append(f"\\caption{{{esc(caption)}}}")
    if label:
        lines.append(f"\\label{{{esc(label)}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def _render_latex_main_table_booktabs(
    rows: List[Dict[str, str]],
    *,
    caption: str = "",
    label: str = "",
) -> str:
    """
    Render a "main table" in the expected paper format:
    - booktabs rules
    - grouped dataset headers via \\multicolumn{4}{c}{...}
    - columns: Method + (P,R,F1,IoU) per dataset
    Assumes row keys:
      - "method"
      - micro P/R/F1@d0 and IoU for every dataset
      - (or macro variants, but this renderer is intended for the micro main-table)
    """
    def esc(s: str) -> str:
        return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")

    header1 = (
        " & "
        + " & ".join(
            [
                "\\multicolumn{4}{c}{MuSHROOM}",
                "\\multicolumn{4}{c}{RAGTruth}",
                "\\multicolumn{4}{c}{PsiloQA}",
            ]
        )
        + " \\\\ \\midrule"
    )
    header2 = "Method & " + " & ".join(["P", "R", "F1", "IoU"] * 3) + " \\\\ \\midrule"

    lines = [
        "\\begin{table}[]",
        "\\begin{tabular}{@{}lcccccccccccc@{}}",
        "\\toprule",
        header1,
        header2,
    ]

    for r in rows:
        method = esc(str(r.get("method", "")))
        cells: List[str] = [method]
        for ds in DATASETS:
            for metric in ("P@d0", "R@d0", "F1@d0", "IoU"):
                key = f"{ds}_micro_{metric}"
                cells.append(esc(str(r.get(key, ""))))
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    if caption:
        lines.append(f"\\caption{{{esc(caption)}}}")
    if label:
        lines.append(f"\\label{{{esc(label)}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _render_markdown_table(rows: List[Dict[str, str]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(r.get(c, "") for c in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def _render_text_table(rows: List[Dict[str, str]], columns: List[str]) -> str:
    values = [[str(r.get(c, "")) for c in columns] for r in rows]
    widths = [len(c) for c in columns]
    for row in values:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def hr(ch: str) -> str:
        return "+" + "+".join((ch * (w + 2)) for w in widths) + "+"

    def fmt_row(row: List[str]) -> str:
        cells: List[str] = []
        for i, cell in enumerate(row):
            if i == 0:
                cells.append(" " + cell.ljust(widths[i]) + " ")
            else:
                cells.append(" " + cell.rjust(widths[i]) + " ")
        return "|" + "|".join(cells) + "|"

    out: List[str] = [hr("-"), fmt_row(list(columns)), hr("=")]
    for row in values:
        out.append(fmt_row(row))
    out.append(hr("-"))
    return "\n".join(out)


def _iter_prediction_files(pred_dir: Path) -> Iterable[Path]:
    for p in sorted(pred_dir.glob("*.csv")):
        if p.name.startswith(".") or "parse_table" in p.name or "facts" in p.name:
            continue
        yield p


def _max_gold_end(gold: List[List[int]]) -> int | None:
    if not gold:
        return None
    m = None
    for s, e in gold:
        e_i = int(e)
        m = e_i if m is None else max(m, e_i)
    return m


def _baseline_always_zero(golds: List[List[List[int]]]) -> List[List[List[int]]]:
    return [[] for _ in golds]


def _baseline_always_yes(golds: List[List[List[int]]]) -> List[List[List[int]]]:
    preds: List[List[List[int]]] = []
    for g in golds:
        m = _max_gold_end(g)
        preds.append([[0, m]] if m is not None else [])
    return preds


def _baseline_random(
    golds: List[List[List[int]]],
    *,
    rng: random.Random,
    max_spans: int = 5,
    min_len: int = 1,
) -> List[List[List[int]]]:
    preds: List[List[List[int]]] = []
    for g in golds:
        m = _max_gold_end(g)
        if m is None or m < 0:
            preds.append([])
            continue

        k = rng.randint(0, max_spans)
        spans: List[List[int]] = []
        for _ in range(k):
            if m <= 0:
                continue
            s = rng.randint(0, m - 1)
            max_len = max(min_len, m - s)
            ln = rng.randint(min_len, max_len)
            e = min(m, s + ln)
            spans.append([s, e])
        preds.append(spans)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute SpanCoverage F1 and IoU tables from predictions/*.csv"
    )
    ap.add_argument("--pred-dir", type=Path, action="append", dest="pred_dirs",
                    metavar="DIR", help="Prediction directory (repeat to merge multiple)")
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--min-pred-len", type=int, default=1)
    ap.add_argument("--print", action="store_true", help="Print tables to stdout")
    ap.add_argument("--include-baselines", action="store_true", help="Add simple baselines to each dataset table")
    ap.add_argument(
        "--include-prf-d0",
        action="store_true",
        help="Include Precision/Recall (and F1) for delta=0 in addition to F1@d* columns",
    )
    ap.add_argument(
        "--main-table",
        action="store_true",
        help="Write/print one combined table with P/R/F1@d0/IoU for all datasets",
    )
    ap.add_argument(
        "--latex",
        action="store_true",
        help="Additionally write LaTeX .tex tables (per-dataset and main-table if requested)",
    )
    ap.add_argument(
        "--latex-percent",
        action="store_true",
        default=True,
        help="Format LaTeX metrics as percentages (x100) with 2 decimals (e.g., 12.34) (default)",
    )
    ap.add_argument(
        "--latex-prob",
        action="store_true",
        help="Format LaTeX metrics as probabilities (0-1) with 4 decimals instead of percent",
    )
    ap.add_argument("--random-seed", type=int, default=0, help="Seed for random baseline")
    ap.add_argument("--agg", choices=["micro", "macro", "both"], default="micro")
    args = ap.parse_args()

    pred_dirs: List[Path] = args.pred_dirs or [Path("predictions")]
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # results[dataset][method][agg] =
    # {"f1": {delta: float}, "p_d0": float, "r_d0": float, "iou": float}
    results: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {ds: {} for ds in DATASETS}
    golds_by_ds: Dict[str, List[List[List[int]]]] = {}

    all_csv_paths = (p for pred_dir in pred_dirs for p in _iter_prediction_files(pred_dir))
    for csv_path in all_csv_paths:
        try:
            key = _parse_run_key(csv_path)
        except ValueError as exc:
            print(f"Skipping {csv_path.name}: {exc}", file=sys.stderr)
            continue
        golds, preds = _read_gold_pred_csv(csv_path)
        golds_by_ds.setdefault(key.dataset, golds)
        iou = span_iou_macro(golds, preds)
        by_agg: Dict[str, Dict[str, object]] = {}
        if args.agg in ("micro", "both"):
            f1_by_delta: Dict[int, float] = {}
            pr_d0 = rc_d0 = 0.0
            for d in DELTAS:
                prf = span_coverage_micro(golds, preds, delta=d, min_pred_len=args.min_pred_len)
                f1_by_delta[d] = prf.fbeta
                if d == 0:
                    pr_d0, rc_d0 = prf.precision, prf.recall
            by_agg["micro"] = {
                "f1": f1_by_delta,
                "p_d0": pr_d0,
                "r_d0": rc_d0,
                "iou": iou,
            }
        if args.agg in ("macro", "both"):
            f1_by_delta = {}
            pr_d0 = rc_d0 = 0.0
            for d in DELTAS:
                prf = span_coverage_macro(golds, preds, delta=d, min_pred_len=args.min_pred_len)
                f1_by_delta[d] = prf.fbeta
                if d == 0:
                    pr_d0, rc_d0 = prf.precision, prf.recall
            by_agg["macro"] = {
                "f1": f1_by_delta,
                "p_d0": pr_d0,
                "r_d0": rc_d0,
                "iou": iou,
            }
        results[key.dataset][key.method] = by_agg

    if args.include_baselines:
        rng = random.Random(args.random_seed)
        for ds, golds in golds_by_ds.items():
            baselines: Dict[str, List[List[List[int]]]] = {
                "baseline_always_zero": _baseline_always_zero(golds),
                "baseline_always_yes": _baseline_always_yes(golds),
                "baseline_random": _baseline_random(golds, rng=rng),
            }
            for method, preds in baselines.items():
                by_agg: Dict[str, Dict[str, object]] = {}
                iou = span_iou_macro(golds, preds)
                if args.agg in ("micro", "both"):
                    f1_by_delta: Dict[int, float] = {}
                    pr_d0 = rc_d0 = 0.0
                    for d in DELTAS:
                        prf = span_coverage_micro(golds, preds, delta=d, min_pred_len=args.min_pred_len)
                        f1_by_delta[d] = prf.fbeta
                        if d == 0:
                            pr_d0, rc_d0 = prf.precision, prf.recall
                    by_agg["micro"] = {
                        "f1": f1_by_delta,
                        "p_d0": pr_d0,
                        "r_d0": rc_d0,
                        "iou": iou,
                    }
                if args.agg in ("macro", "both"):
                    f1_by_delta = {}
                    pr_d0 = rc_d0 = 0.0
                    for d in DELTAS:
                        prf = span_coverage_macro(golds, preds, delta=d, min_pred_len=args.min_pred_len)
                        f1_by_delta[d] = prf.fbeta
                        if d == 0:
                            pr_d0, rc_d0 = prf.precision, prf.recall
                    by_agg["macro"] = {
                        "f1": f1_by_delta,
                        "p_d0": pr_d0,
                        "r_d0": rc_d0,
                        "iou": iou,
                    }
                results[ds][method] = by_agg

    def _get_f1(ds: str, method: str, agg: str, d: int) -> float:
        return float((results[ds][method][agg]["f1"])[d])  # type: ignore[index]

    def _get_p(ds: str, method: str, agg: str) -> float:
        return float(results[ds][method][agg]["p_d0"])  # type: ignore[arg-type]

    def _get_r(ds: str, method: str, agg: str) -> float:
        return float(results[ds][method][agg]["r_d0"])  # type: ignore[arg-type]

    def _get_iou(ds: str, method: str, agg: str) -> float:
        return float(results[ds][method][agg]["iou"])  # type: ignore[arg-type]

    def _fmt_latex_value(x: float) -> str:
        if args.latex_prob:
            return f"{x:.4f}"
        return f"{x * 100.0:.2f}"

    def _pretty_method(method: str) -> str:
        if method == "baseline_always_yes":
            return "always yes"
        if method == "baseline_always_zero":
            return "always no"
        return method

    for ds in DATASETS:
        methods = sorted(results[ds].keys())
        def f1_cols(prefix: str) -> List[str]:
            if not args.include_prf_d0:
                return [f"F1_{prefix}@d{d}" for d in DELTAS]
            return [
                f"P_{prefix}@d0",
                f"R_{prefix}@d0",
                "F1_" + prefix + "@d0",
                *[f"F1_{prefix}@d{d}" for d in DELTAS if d != 0],
            ]

        if args.agg == "micro":
            columns = ["method", *f1_cols("micro"), "IoU"]
        elif args.agg == "macro":
            columns = ["method", *f1_cols("macro"), "IoU"]
        else:
            columns = ["method", *f1_cols("micro"), *f1_cols("macro"), "IoU"]
        rows: List[Dict[str, str]] = []
        for method in methods:
            row: Dict[str, str] = {"method": _pretty_method(method)}
            if args.agg in ("micro", "both"):
                for d in DELTAS:
                    row[f"F1_micro@d{d}"] = _fmt_f1(_get_f1(ds, method, "micro", d))
                if args.include_prf_d0:
                    row["P_micro@d0"] = _fmt_metric(_get_p(ds, method, "micro"))
                    row["R_micro@d0"] = _fmt_metric(_get_r(ds, method, "micro"))
            if args.agg in ("macro", "both"):
                for d in DELTAS:
                    row[f"F1_macro@d{d}"] = _fmt_f1(_get_f1(ds, method, "macro", d))
                if args.include_prf_d0:
                    row["P_macro@d0"] = _fmt_metric(_get_p(ds, method, "macro"))
                    row["R_macro@d0"] = _fmt_metric(_get_r(ds, method, "macro"))
            iou_agg = "micro" if args.agg in ("micro", "both") else "macro"
            row["IoU"] = _fmt_metric(_get_iou(ds, method, iou_agg))
            rows.append(row)

        md = f"# {ds}\n\n" + _render_markdown_table(rows, columns) + "\n"
        out_path = out_dir / f"span_coverage_{ds}.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {out_path}")
        if args.latex:
            # For per-dataset LaTeX, keep a simple tabular, but optionally allow percent formatting.
            latex_rows: List[Dict[str, str]] = []
            for r in rows:
                lr: Dict[str, str] = {"method": r["method"]}
                for c in columns:
                    if c == "method":
                        continue
                    v = r.get(c, "")
                    if args.latex_percent and v != "":
                        try:
                            lr[c] = _fmt_latex_value(float(v))
                        except Exception:
                            lr[c] = v
                    else:
                        lr[c] = v
                latex_rows.append(lr)

            tex = _render_latex_table(
                latex_rows,
                columns,
                caption=f"SpanCoverage F1 and IoU results on {ds}.",
                label=f"tab:span_coverage_{ds}",
            )
            tex_path = out_dir / f"span_coverage_{ds}.tex"
            tex_path.write_text(tex + "\n", encoding="utf-8")
            print(f"Wrote {tex_path}")
        if args.print:
            print()
            print(ds)
            print(_render_text_table(rows, columns))

    if args.main_table:
        # Union of methods across datasets, sorted for stable output.
        all_methods = sorted({m for ds in DATASETS for m in results[ds].keys()})

        def _ds_metric_cols(ds: str, prefix: str) -> List[str]:
            # P/R/F1 at d0 plus mean character IoU.
            return [
                f"{ds}_{prefix}_P@d0",
                f"{ds}_{prefix}_R@d0",
                f"{ds}_{prefix}_F1@d0",
                f"{ds}_{prefix}_IoU",
            ]

        columns = ["method"]
        if args.agg in ("micro", "both"):
            for ds in DATASETS:
                columns.extend(_ds_metric_cols(ds, "micro"))
        if args.agg in ("macro", "both"):
            for ds in DATASETS:
                columns.extend(_ds_metric_cols(ds, "macro"))

        rows: List[Dict[str, str]] = []
        for method in all_methods:
            row: Dict[str, str] = {"method": _pretty_method(method)}
            for ds in DATASETS:
                if method not in results[ds]:
                    continue
                if args.agg in ("micro", "both"):
                    p = _get_p(ds, method, "micro")
                    r = _get_r(ds, method, "micro")
                    f1 = _get_f1(ds, method, "micro", 0)
                    row[f"{ds}_micro_P@d0"] = _fmt_metric(p)
                    row[f"{ds}_micro_R@d0"] = _fmt_metric(r)
                    row[f"{ds}_micro_F1@d0"] = _fmt_f1(f1)
                    row[f"{ds}_micro_IoU"] = _fmt_metric(_get_iou(ds, method, "micro"))
                if args.agg in ("macro", "both"):
                    row[f"{ds}_macro_P@d0"] = _fmt_metric(_get_p(ds, method, "macro"))
                    row[f"{ds}_macro_R@d0"] = _fmt_metric(_get_r(ds, method, "macro"))
                    row[f"{ds}_macro_F1@d0"] = _fmt_f1(_get_f1(ds, method, "macro", 0))
                    row[f"{ds}_macro_IoU"] = _fmt_metric(_get_iou(ds, method, "macro"))
            rows.append(row)

        md = "# Main table (P/R/F1@d0/IoU)\n\n" + _render_markdown_table(rows, columns) + "\n"
        out_path = out_dir / "span_coverage_main_table.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {out_path}")
        if args.latex:
            if args.agg != "micro":
                # For now, only the "paper-style" main table is defined for micro.
                # Fall back to the simple renderer for other agg settings.
                latex_rows = []
                for r in rows:
                    lr = {"method": r["method"]}
                    for c in columns:
                        if c == "method":
                            continue
                        v = r.get(c, "")
                        if args.latex_percent and v != "":
                            try:
                                lr[c] = _fmt_latex_value(float(v))
                            except Exception:
                                lr[c] = v
                        else:
                            lr[c] = v
                    latex_rows.append(lr)
                tex = _render_latex_table(
                    latex_rows,
                    columns,
                    caption="SpanCoverage main table (P/R/F1 at d0 and IoU) across datasets.",
                    label="tab:span_coverage_main",
                )
            else:
                # Convert micro columns to percent-format strings if requested.
                latex_rows = []
                for r in rows:
                    lr = {"method": r["method"]}
                    for ds in DATASETS:
                        for metric in ("P@d0", "R@d0", "F1@d0", "IoU"):
                            k = f"{ds}_micro_{metric}"
                            v = r.get(k, "")
                            if args.latex_percent and v != "":
                                try:
                                    lr[k] = _fmt_latex_value(float(v))
                                except Exception:
                                    lr[k] = v
                            else:
                                lr[k] = v
                    latex_rows.append(lr)
                tex = _render_latex_main_table_booktabs(
                    latex_rows,
                    caption="SpanCoverage main table (P/R/F1 at d0 and IoU) across datasets.",
                    label="tab:span_coverage_main",
                )
            tex_path = out_dir / "span_coverage_main_table.tex"
            tex_path.write_text(tex + "\n", encoding="utf-8")
            print(f"Wrote {tex_path}")
        if args.print:
            print()
            print("main_table")
            print(_render_text_table(rows, columns))


if __name__ == "__main__":
    main()
