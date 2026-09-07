"""CARB-family benchmark scoring for the rules pipeline.

Runs the pipeline over the vendored CaRB test sentences, writes predictions
in allennlp format, and scores them with the vendored carb scorers:
carb(s,s), carb(s,m), oie16, wire57. Test split only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from ..config import ExtractionConfig
from ..models import Triplet
from ..pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED = REPO_ROOT / "third_party" / "carb"
CARB_SENTENCES = VENDORED / "data" / "carb_sentences.txt"
GOLD_TSV = VENDORED / "data" / "gold" / "test.tsv"
GOLD_ALLENNLP = VENDORED / "data" / "test_gold_allennlp_format.txt"

# Python import names the vendored scorers need at module import time.
REQUIRED_DEPS = ("docopt", "sklearn", "nltk", "tqdm", "regex", "ipdb")
SCORERS = ("carb_ss", "carb_sm", "oie16", "wire57")


def _install_hint() -> str:
    return (
        "carb scorer dependencies missing. Install into the enoki env "
        "(never base):\n"
        "  conda run -n enoki pip install docopt scikit-learn nltk "
        "tqdm regex ipdb"
    )


def triplet_to_allennlp(triplet: "Triplet", sentence: str) -> str:
    """Render one Triplet as one allennlp-format predictions line.

    Format: ``sentence \\t <arg1> s </arg1> <rel> r </rel>
    <arg2> a </arg2> \\t confidence``. The sentence is emitted verbatim
    (scorers key predictions by exact sentence text). Negation is folded
    into the relation; modality is dropped; intransitives get an empty
    arg2.
    """
    subj = triplet.subject.text.strip()
    rel = triplet.predicate_surface.strip()
    if triplet.negated:
        rel = f"not {rel}"
    arg = "" if triplet.argument is None else triplet.argument.span.text.strip()
    tagged = (
        f"<arg1> {subj} </arg1> <rel> {rel} </rel> <arg2> {arg} </arg2>"
    )
    return f"{sentence}\t{tagged}\t{triplet.confidence:.6f}"


@dataclass(frozen=True)
class ScorerResult:
    name: str
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc: Optional[float] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "error": self.error,
        }


# NumPy 2.x prints scalars as np.float64(0.5) / np.int64(1) instead of
# bare numbers.  The _NUM fragment matches either representation.
_NUM = r"(?:np\.\w+\()?([0-9]*\.?[0-9]+)\)?"
_AUC_RE = re.compile(r"AUC:\s*" + _NUM)
_OPT_RE = re.compile(
    r"Optimal \(precision, recall, F1[^)]*\):\s*\(\s*"
    + _NUM + r",\s*" + _NUM + r",\s*" + _NUM
)
_WIRE_RE = re.compile(
    r"prec/rec/f1:\s*([0-9]*\.?[0-9]+)%\s+([0-9]*\.?[0-9]+)%\s+"
    r"([0-9]*\.?[0-9]+)"
)


def parse_scorer_output(name: str, combined: str) -> ScorerResult:
    """Parse combined stdout+stderr of a vendored scorer.

    carb.py prints to stdout; oie16.py logs the same shape to stderr
    (with a 4-tuple incl. threshold); wire57 prints percentages.
    """
    if name == "wire57":
        m = _WIRE_RE.search(combined)
        if not m:
            return ScorerResult(name, error="could not parse wire57 output")
        return ScorerResult(
            name,
            precision=float(m.group(1)) / 100.0,
            recall=float(m.group(2)) / 100.0,
            f1=float(m.group(3)),
        )
    opt = _OPT_RE.search(combined)
    if not opt:
        return ScorerResult(name, error=f"could not parse {name} output")
    auc_m = _AUC_RE.search(combined)
    return ScorerResult(
        name,
        precision=float(opt.group(1)),
        recall=float(opt.group(2)),
        f1=float(opt.group(3)),
        auc=float(auc_m.group(1)) if auc_m else None,
    )


def merge_incremental_triplets(triplets: List["Triplet"]) -> List["Triplet"]:
    """Drop incremental sub-facts emitted by the multi-granularity rules.

    For each (subject, predicate) group, remove any Triplet whose
    argument text is a case-insensitive STRICT substring of a longer
    argument in the same group (mirrors ``evaluate_openie.merge_incremental``).
    Triplets with ``argument is None`` are kept as-is — they have no
    arg span to compare. After our IncrementalFactGroup adapter (or
    the multi-granularity rules incremental_minimal_arg /
    incremental_maximal_arg / incremental_maximal_subject), the same
    proposition is often emitted at multiple widths
    ('limits' / 'strict limits' / 'strict limits on industrial emissions').
    With merging on, we keep only the widest variant per (subj, pred)
    group — what CaRB-style benchmarks prefer.
    """
    groups: dict = {}
    none_args: List["Triplet"] = []
    for t in triplets:
        if t.argument is None or t.argument.span is None:
            none_args.append(t)
            continue
        key = (t.subject.text.lower(), t.predicate_surface.lower())
        groups.setdefault(key, []).append(t)

    kept: List["Triplet"] = list(none_args)
    for group in groups.values():
        args_lower = [t.argument.span.text.lower() for t in group]
        for i, t in enumerate(group):
            ai = args_lower[i]
            is_substring_of_longer = any(
                i != j and len(args_lower[j]) > len(ai) and ai in args_lower[j]
                for j in range(len(group))
            )
            if not is_substring_of_longer:
                kept.append(t)
    return kept


def write_predictions(
    extract: Callable[[str], List["Triplet"]],
    sentences: Iterable[str],
    out_path: Path,
    merge_incremental: bool = False,
) -> int:
    """Run ``extract`` over each sentence; write allennlp lines; return count.

    Blank lines are skipped. The sentence is right-stripped of newlines
    only (its internal whitespace is preserved for exact-text keying).
    When ``merge_incremental`` is True, the per-sentence triplet list
    is filtered through :func:`merge_incremental_triplets` before
    serialization — useful for CaRB-style benchmarks that penalise
    sub-facts.
    """
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for raw in sentences:
            sent = raw.rstrip("\n")
            if not sent.strip():
                continue
            triplets = list(extract(sent))
            if merge_incremental:
                triplets = merge_incremental_triplets(triplets)
            for t in triplets:
                f.write(triplet_to_allennlp(t, sent) + "\n")
                n += 1
    return n


@dataclass
class Report:
    num_sentences: int
    num_predictions: int
    results: List[ScorerResult] = field(default_factory=list)
    merge_incremental: bool = False

    def as_dict(self) -> dict:
        return {
            "num_sentences": self.num_sentences,
            "num_predictions": self.num_predictions,
            "merge_incremental": self.merge_incremental,
            "results": [r.as_dict() for r in self.results],
        }


def write_report(report: Report, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report.as_dict(), indent=2)
    )
    lines = [
        "# CARB benchmark report",
        "",
        f"- sentences: {report.num_sentences}",
        f"- predictions: {report.num_predictions}",
        f"- merge_incremental: {report.merge_incremental}",
        "",
        "| metric | precision | recall | f1 | auc |",
        "|--------|-----------|--------|----|-----|",
    ]
    for r in report.results:
        if r.error:
            lines.append(f"| {r.name} | — | — | — | — |  _{r.error}_")
        else:
            auc = "—" if r.auc is None else f"{r.auc:.4f}"
            lines.append(
                f"| {r.name} | {r.precision:.4f} | {r.recall:.4f} "
                f"| {r.f1:.4f} | {auc} |"
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def ensure_scorer_deps(python: str) -> None:
    probe = "import " + ", ".join(REQUIRED_DEPS)
    res = subprocess.run(
        [python, "-c", probe], capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(
            _install_hint() + f"\n(probe error: {res.stderr.strip()})"
        )


def _scorer_argv(name: str, python: str, predictions: Path) -> List[str]:
    if name == "carb_ss":
        return [python, str(VENDORED / "carb.py"), "--allennlp",
                str(predictions), "--gold", str(GOLD_TSV), "--out",
                os.devnull, "--single_match"]
    if name == "carb_sm":
        return [python, str(VENDORED / "carb.py"), "--allennlp",
                str(predictions), "--gold", str(GOLD_TSV), "--out",
                os.devnull]
    if name == "oie16":
        return [python, str(VENDORED / "oie16.py"), "--allennlp",
                str(predictions), "--gold", str(GOLD_TSV), "--out",
                os.devnull]
    if name == "wire57":
        return [python, str(VENDORED / "wire57_evaluation.py"),
                "--system", str(predictions), "--gold",
                str(GOLD_ALLENNLP)]
    raise ValueError(f"unknown scorer {name!r}")


def run_scorer(name: str, argv: Sequence[str]) -> ScorerResult:
    try:
        res = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=3600
        )
    except Exception as exc:  # noqa: BLE001
        return ScorerResult(name, error=f"subprocess failed: {exc}")
    combined = (res.stdout or "") + "\n" + (res.stderr or "")
    parsed = parse_scorer_output(name, combined)
    if parsed.error and res.returncode != 0:
        tail = (res.stderr or "").strip()[-500:]
        return ScorerResult(name, error=f"exit {res.returncode}: {tail}")
    return parsed


def benchmark(
    config: Optional[ExtractionConfig] = None,
    *,
    python: Optional[str] = None,
    max_sentences: Optional[int] = None,
    predictions_path: Optional[Path] = None,
    merge_incremental: bool = False,
) -> Report:
    config = config or ExtractionConfig()
    python = python or sys.executable
    for p in (CARB_SENTENCES, GOLD_TSV, GOLD_ALLENNLP):
        if not p.exists():
            raise FileNotFoundError(
                f"Vendored carb data missing: {p}. Re-vendor "
                "third_party/carb (see third_party/carb/README.md)."
            )
    ensure_scorer_deps(python)

    sents = CARB_SENTENCES.read_text(encoding="utf-8").splitlines()
    if max_sentences is not None:
        sents = sents[:max_sentences]

    pipeline = Pipeline(config)
    keep = predictions_path is not None
    pred_path = predictions_path or (
        Path(os.getenv("TMPDIR", "/tmp")) / "carb_preds.tsv"
    )
    n = write_predictions(
        pipeline.extract, sents, pred_path,
        merge_incremental=merge_incremental,
    )

    report = Report(
        num_sentences=len(sents),
        num_predictions=n,
        merge_incremental=merge_incremental,
    )
    for name in SCORERS:
        report.results.append(
            run_scorer(name, _scorer_argv(name, python, pred_path))
        )
    if not keep:
        try:
            os.unlink(pred_path)
        except OSError:
            pass
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score the rules pipeline with CARB-family benchmarks "
        "(carb(s,s), carb(s,m), oie16, wire57) on the CaRB test split."
    )
    ap.add_argument(
        "--out", type=Path, default=Path("evaluation_reports") / "carb",
        help="Output dir for report.json / report.md.",
    )
    ap.add_argument(
        "--predictions", type=Path, default=None,
        help="Keep the generated allennlp predictions file at this path.",
    )
    ap.add_argument(
        "--max-sentences", type=int, default=None,
        help="Cap sentences (smoke testing).",
    )
    ap.add_argument(
        "--python", default=None,
        help="Interpreter for scorer subprocesses (default: this one).",
    )
    ap.add_argument(
        "--merge-incremental", action="store_true",
        help="Drop incremental sub-facts before scoring: for each "
        "(subject, predicate) group, remove any triplet whose argument "
        "text is a case-insensitive substring of a longer argument in "
        "the same group. Mirrors evaluate_openie.merge_incremental.",
    )
    args = ap.parse_args(argv)

    config = ExtractionConfig()
    report = benchmark(
        config,
        python=args.python,
        max_sentences=args.max_sentences,
        predictions_path=args.predictions,
        merge_incremental=args.merge_incremental,
    )
    write_report(report, args.out)
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
