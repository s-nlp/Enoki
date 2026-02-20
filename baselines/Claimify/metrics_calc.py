"""
post_run_metrics.py

Post-process Claimify runs from segments.jsonl
and compute:
- Avg claims / sentence
- Avg extract/verify/total time per sentence
- FLOPs aggregates (extract/verify/total), and aggregate TFLOPs/s = sum(flops)/sum(time)
- Optional: same metrics for "extracted sentences only" (claims>0 and not stopped)

Usage:
  python post_run_metrics.py \
    --segments out_claimify/segments.jsonl \
    --out out_claimify/post_metrics.json

Optional:
  python post_run_metrics.py --segments ... --metrics_json out_claimify/metrics.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_eval_row(r: Dict[str, Any]) -> bool:
    # Evaluable for efficiency: has gold and no model failure.
    return (r.get("gold_supported") is not None) and (r.get("fail_reason") is None)


def is_extracted_row(r: Dict[str, Any]) -> bool:
    # Strict "successful extraction": evaluable + produced claims + not stopped.
    if not is_eval_row(r):
        return False
    if len(r.get("claims") or []) == 0:
        return False
    if r.get("stop_reason") is not None:
        return False
    return True


def is_verified_row(r: Dict[str, Any]) -> bool:
    # Sentences with at least one claim verified (verification list non-empty)
    if not is_eval_row(r):
        return False
    ver = r.get("verification") or []
    return len(ver) > 0


def sum_time(rows: List[Dict[str, Any]], key: str) -> float:
    # key: "extract_s" | "verify_s" | "total_s" | "selection_s" etc.
    s = 0.0
    for r in rows:
        t = (r.get("timing") or {}).get(key, 0.0)
        if isinstance(t, (int, float)):
            s += float(t)
    return s


def sum_tokens_total(rows: List[Dict[str, Any]]) -> int:
    s = 0
    for r in rows:
        tok = r.get("tokens") or {}
        s += int(tok.get("total_prompt", 0) or 0)
        s += int(tok.get("total_gen", 0) or 0)
    return s


def sum_flops(rows: List[Dict[str, Any]], key: str) -> float:
    # key: "extract_flops" | "verify_flops" | "total_flops"
    s = 0.0
    for r in rows:
        f = (r.get("flops") or {}).get(key, None)
        if isinstance(f, (int, float)):
            s += float(f)
    return s


def count_claims(rows: List[Dict[str, Any]]) -> int:
    return sum(len(r.get("claims") or []) for r in rows)


def count_verified_claims(rows: List[Dict[str, Any]]) -> int:
    # Number of verification entries, not number of unique claims.
    return sum(len(r.get("verification") or []) for r in rows)


def tflops_per_s_agg(total_flops: float, total_s: float) -> Optional[float]:
    if total_s <= 0:
        return None
    return (total_flops / total_s) / 1e12


def compute_block(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Returns None if FLOPs are not present (e.g., model_params_b=0).
    Otherwise returns sums + aggregate TFLOPs/s.
    """
    sum_extract_flops = sum_flops(rows, "extract_flops")
    sum_verify_flops = sum_flops(rows, "verify_flops")
    sum_total_flops = sum_flops(rows, "total_flops")

    # If no flops info in file, return None
    if (sum_extract_flops + sum_verify_flops + sum_total_flops) <= 0.0:
        return None

    sum_extract_s = sum_time(rows, "extract_s")
    sum_verify_s = sum_time(rows, "verify_s")
    sum_total_s = sum_time(rows, "total_s")

    # Try to pull params_b and k from the first row that has them
    params_b = None
    k = None
    for r in rows:
        f = r.get("flops") or {}
        if "params_b" in f and "k" in f:
            params_b = f.get("params_b")
            k = f.get("k")
            break

    return {
        "params_b": params_b,
        "flops_per_param_k": k,
        "sum_extract_flops": sum_extract_flops,
        "sum_verify_flops": sum_verify_flops,
        "sum_total_flops": sum_total_flops,
        "sum_extract_time_s": sum_extract_s,
        "sum_verify_time_s": sum_verify_s,
        "sum_total_time_s": sum_total_s,
        "extract_tflops_per_s_agg": tflops_per_s_agg(sum_extract_flops, sum_extract_s),
        "verify_tflops_per_s_agg": tflops_per_s_agg(sum_verify_flops, sum_verify_s),
        "total_tflops_per_s_agg": tflops_per_s_agg(sum_total_flops, sum_total_s),
    }


def efficiency_block(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    n = len(rows)
    sum_claims = count_claims(rows)
    sum_tokens = sum_tokens_total(rows)

    sum_extract_s = sum_time(rows, "extract_s")
    sum_verify_s = sum_time(rows, "verify_s")
    sum_total_s = sum_time(rows, "total_s")

    # Optional claim-level verify latency
    total_verified_claims = count_verified_claims(rows)
    verify_time_per_claim = safe_div(sum_verify_s, total_verified_claims)

    out = {
        "name": name,
        "n_sentences": n,
        "sum_claims": sum_claims,
        "avg_claims_per_sentence": safe_div(sum_claims, n),
        "sum_total_tokens": sum_tokens,
        "avg_total_tokens_per_sentence": safe_div(sum_tokens, n),
        "sum_extract_time_s": sum_extract_s,
        "sum_verify_time_s": sum_verify_s,
        "sum_total_time_s": sum_total_s,
        "avg_extract_time_s_per_sentence": safe_div(sum_extract_s, n),
        "avg_verify_time_s_per_sentence": safe_div(sum_verify_s, n),
        "avg_total_time_s_per_sentence": safe_div(sum_total_s, n),
        "sum_verified_claims": total_verified_claims,
        "verify_time_s_per_claim": verify_time_per_claim,
    }

    comp = compute_block(rows)
    if comp is not None:
        # Also provide per-sentence average flops (use sums / n)
        out["compute"] = {
            **comp,
            "avg_extract_flops_per_sentence": safe_div(comp["sum_extract_flops"], n),
            "avg_verify_flops_per_sentence": safe_div(comp["sum_verify_flops"], n),
            "avg_total_flops_per_sentence": safe_div(comp["sum_total_flops"], n),
        }
    else:
        out["compute"] = None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segments", type=str, required=True, help="Path to segments.jsonl"
    )
    ap.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="Optional path to original metrics.json (merged into output)",
    )
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path")
    args = ap.parse_args()

    segments_path = Path(args.segments)
    rows = load_jsonl(segments_path)

    eval_rows = [r for r in rows if is_eval_row(r)]
    extracted_rows = [r for r in rows if is_extracted_row(r)]
    verified_rows = [r for r in rows if is_verified_row(r)]

    result: Dict[str, Any] = {
        "source": str(segments_path),
        "counts": {
            "n_rows_total": len(rows),
            "n_eval_rows": len(eval_rows),
            "n_extracted_rows": len(extracted_rows),
            "n_verified_rows": len(verified_rows),
        },
        # Coverage-aware efficiency (includes stops, excludes hard failures)
        "efficiency_eval": efficiency_block(eval_rows, "eval_rows"),
        # Success-only efficiency (only sentences that actually produced claims and weren't stopped)
        "efficiency_extracted": efficiency_block(extracted_rows, "extracted_rows"),
        # Rows that had verification entries (usually do_verify runs)
        "efficiency_verified": efficiency_block(verified_rows, "verified_rows"),
    }

    # Merge in original metrics.json if provided (so you keep F1/CM etc.)
    if args.metrics_json:
        mj = load_json(Path(args.metrics_json))
        result["original_metrics_json"] = mj

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
