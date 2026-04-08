#!/usr/bin/env python3
import argparse
import os
import json
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **_: Any):
        return x

try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None


_SENT_RE = re.compile(r"sentence(\d+)$")


def clean(s: Any) -> str:
    return str(s or "").strip()


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def flatten_evidence(x: Any) -> List[str]:
    out: List[str] = []

    def rec(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif isinstance(v, (list, tuple)):
            for item in v:
                rec(item)
        elif isinstance(v, dict):
            out.append(json.dumps(v, ensure_ascii=False))
        else:
            out.append(str(v))

    rec(x)
    return out


def norm_gold_label_factbench(x: Any) -> Optional[bool]:
    if x is True or x is False:
        return bool(x)
    if x is None:
        return None
    if isinstance(x, str) and x.strip().upper() == "NA":
        return None
    return None


def to_gold_supported_felm(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"n/a", "na", "none", "null", ""}:
        return None
    if s in {"supported", "support", "entailed", "entails", "true", "yes", "1"}:
        return True
    if s in {"not_supported", "not supported", "unsupported", "contradicted", "false", "no", "0"}:
        return False
    return None


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def update_cm_not_supported_positive(cm: Dict[str, int], gold_supported: bool, pred_supported: bool) -> None:
    gold_pos = not gold_supported
    pred_pos = not pred_supported
    if gold_pos and pred_pos:
        cm["TP"] += 1
    elif (not gold_pos) and pred_pos:
        cm["FP"] += 1
    elif gold_pos and (not pred_pos):
        cm["FN"] += 1
    else:
        cm["TN"] += 1


def cm_to_macro_f1(cm: Dict[str, int]) -> Dict[str, Any]:
    p_ns, r_ns, f1_ns = prf1(cm["TP"], cm["FP"], cm["FN"])
    p_s, r_s, f1_s = prf1(cm["TN"], cm["FN"], cm["FP"])
    return {
        "precision_not_supported": p_ns,
        "recall_not_supported": r_ns,
        "f1_not_supported": f1_ns,
        "precision_supported": p_s,
        "recall_supported": r_s,
        "f1_supported": f1_s,
        "f1_macro": (f1_ns + f1_s) / 2.0,
        "confusion_matrix": cm,
    }


def extract_claim_texts(result: Any) -> List[Any]:
    claims = getattr(result, "claims", None)
    if claims is None and isinstance(result, dict):
        claims = result.get("claims")
    out: List[Any] = []
    for claim in claims or []:
        content = getattr(claim, "content", claim)
        if isinstance(content, (list, tuple)):
            out.append(list(content))
        else:
            s = clean(content)
            if s:
                out.append(s)
    return out


def normalize_label(label: Any) -> str:
    if hasattr(label, "value"):
        label = label.value
    return str(label or "").strip()


def is_entailment(label: Any) -> bool:
    return normalize_label(label).lower() in {"entailment", "entailed", "support", "supported", "true"}


def strict_sentence_supported(labels: List[Any]) -> Optional[bool]:
    if not labels:
        return None
    return all(is_entailment(label) for label in labels)


def sentence_risk_not_supported(labels: List[Any]) -> Optional[float]:
    if not labels:
        return None
    return sum(0 if is_entailment(label) else 1 for label in labels) / len(labels)


def import_refchecker():
    try:
        from refchecker import AlignScoreChecker, LLMChecker, LLMExtractor, NLIChecker
    except Exception as e:
        raise RuntimeError(
            "Could not import RefChecker. Install it with `pip install refchecker` "
            "and run `python -m spacy download en_core_web_sm` if the extractor requires spaCy."
        ) from e
    return LLMExtractor, LLMChecker, AlignScoreChecker, NLIChecker


def api_host_for_connect(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def local_vllm_api_base(args: argparse.Namespace) -> str:
    return f"http://{api_host_for_connect(args.local_vllm_host)}:{args.local_vllm_port}/v1"


def local_openai_model_name(served_model_name: str) -> str:
    return served_model_name if served_model_name.startswith("openai/") else f"openai/{served_model_name}"


def wait_for_local_vllm(api_base: str, proc: subprocess.Popen, timeout_s: float) -> None:
    url = f"{api_base.rstrip('/')}/models"
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None

    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Local vLLM exited early with code {proc.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(e)
        time.sleep(2.0)

    raise TimeoutError(f"Timed out waiting for local vLLM at {url}. Last error: {last_err}")


@contextmanager
def maybe_local_vllm(args: argparse.Namespace) -> Iterator[None]:
    if not args.run_local_vllm:
        yield
        return

    served_name = args.local_vllm_served_model_name or args.local_vllm_model
    args.extractor_name = local_openai_model_name(served_name)
    if args.checker_type == "llm":
        args.checker_name = local_openai_model_name(served_name)
    args.extractor_api_base = local_vllm_api_base(args)
    args.checker_api_base = args.extractor_api_base
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.local_vllm_model,
        "--served-model-name",
        served_name,
        "--host",
        args.local_vllm_host,
        "--port",
        str(args.local_vllm_port),
        "--gpu-memory-utilization",
        str(args.local_vllm_gpu_memory_utilization),
    ]
    if args.local_vllm_trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.local_vllm_tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(args.local_vllm_tensor_parallel_size)])
    if args.local_vllm_max_model_len > 0:
        cmd.extend(["--max-model-len", str(args.local_vllm_max_model_len)])
    if args.local_vllm_dtype:
        cmd.extend(["--dtype", args.local_vllm_dtype])
    if args.local_vllm_extra_args:
        cmd.extend(shlex.split(args.local_vllm_extra_args))

    print("Starting local vLLM:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.Popen(cmd)
    try:
        wait_for_local_vllm(args.extractor_api_base, proc, args.local_vllm_startup_timeout_s)
        print(f"Local vLLM is ready at {args.extractor_api_base}", flush=True)
        yield
    finally:
        if proc.poll() is None:
            print("Stopping local vLLM...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)


def build_refchecker(args: argparse.Namespace) -> Tuple[Any, Any]:
    LLMExtractor, LLMChecker, AlignScoreChecker, NLIChecker = import_refchecker()

    extractor_kwargs = {"model": args.extractor_name, "batch_size": args.batch_size_extractor}
    checker_kwargs = {"batch_size": args.batch_size_checker}
    if args.extractor_api_base:
        extractor_kwargs["api_base"] = args.extractor_api_base
    if args.checker_api_base:
        checker_kwargs["api_base"] = args.checker_api_base

    extractor = LLMExtractor(**extractor_kwargs)

    if args.checker_type == "llm":
        checker_kwargs["model"] = args.checker_name
        checker = LLMChecker(**checker_kwargs)
    elif args.checker_type == "nli":
        checker = NLIChecker(device=args.device, batch_size=args.batch_size_checker)
    elif args.checker_type == "alignscore":
        checker = AlignScoreChecker(device=args.device, batch_size=args.batch_size_checker)
    else:
        raise ValueError(f"Unknown checker_type: {args.checker_type}")

    return extractor, checker


def load_factbench_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    samples = 0
    with open(args.data, "r", encoding="utf-8") as f:
        for sid, line in enumerate(f):
            if args.max_samples and sid >= args.max_samples:
                break
            ex = json.loads(line)
            samples += 1
            question = clean(ex.get("prompt"))
            sent_items = []
            for sent_key, sent_data in (ex.get("sentences") or {}).items():
                m = _SENT_RE.search(sent_key)
                idx = int(m.group(1)) if m else 10**9
                sent_items.append((idx, sent_key, sent_data))
            sent_items.sort(key=lambda x: x[0])

            for _, sent_key, sent_data in sent_items:
                sentence = clean(sent_data.get("text"))
                gold_supported = norm_gold_label_factbench(sent_data.get("sentence_factuality_label"))
                ev_chunks = []
                ev_chunks += flatten_evidence(sent_data.get("auto_evidence"))
                ev_chunks += flatten_evidence(sent_data.get("human_evidence"))
                rows.append(
                    {
                        "sample_id": sid,
                        "sentence_key": sent_key,
                        "question": question,
                        "sentence": sentence,
                        "reference": "\n\n".join(ev_chunks),
                        "gold_supported": gold_supported,
                    }
                )
    return rows, samples


def load_felm_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], int]:
    if load_from_disk is None:
        raise RuntimeError("FELM runner requires `datasets` (pip install datasets).")

    root = Path(args.felm_dir)
    p_split = root / args.subset / args.split
    if p_split.exists():
        ds = load_from_disk(str(p_split))
    else:
        obj = load_from_disk(str(root / args.subset))
        ds = obj[args.split]
    if args.max_examples and args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    rows: List[Dict[str, Any]] = []
    for ex in ds:
        question = clean(ex.get("prompt"))
        sentences = [clean(s) for s in (ex.get("segmented_response") or [])]
        labels = [to_gold_supported_felm(x) for x in (ex.get("labels") or [])]
        reference = clean(ex.get("ref_text"))
        ex_index = ex.get("index", None)
        for i, sentence in enumerate(sentences):
            rows.append(
                {
                    "example_index": ex_index,
                    "seg_id": i,
                    "question": question,
                    "sentence": sentence,
                    "reference": reference,
                    "gold_supported": labels[i] if i < len(labels) else None,
                }
            )
    return rows, len(ds)


def run_refchecker_on_rows(
    args: argparse.Namespace, rows: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    extractor, checker = build_refchecker(args)
    fail = {"empty_sentence": 0, "no_reference": 0, "extract_exception": 0, "check_exception": 0}
    stop = {"no_claims": 0}

    for row in rows:
        row.update(
            {
                "claims": [],
                "verification": [],
                "pred_supported_strict": None,
                "risk_not_supported_strict": None,
                "fail_reason": None,
                "stop_reason": None,
                "timing": {"extract_s": 0.0, "verify_s": 0.0, "total_s": 0.0},
            }
        )

    extract_candidates = [i for i, r in enumerate(rows) if r.get("sentence")]
    for i, r in enumerate(rows):
        if not r.get("sentence"):
            r["fail_reason"] = "empty_sentence"
            fail["empty_sentence"] += 1
        elif not r.get("reference"):
            r["fail_reason"] = "no_reference"
            fail["no_reference"] += 1

    for start in tqdm(range(0, len(extract_candidates), args.batch_size_extractor), desc="RefChecker extract"):
        idxs = extract_candidates[start : start + args.batch_size_extractor]
        batch_responses = [rows[i]["sentence"] for i in idxs]
        batch_questions = [rows[i].get("question") or "" for i in idxs]
        t0 = time.perf_counter()
        try:
            results = extractor.extract(
                batch_responses=batch_responses,
                batch_questions=batch_questions,
                max_new_tokens=args.extractor_max_new_tokens,
            )
            dt = time.perf_counter() - t0
            per_item = dt / len(idxs) if idxs else 0.0
            for rid, result in zip(idxs, results):
                rows[rid]["claims"] = extract_claim_texts(result)
                rows[rid]["timing"]["extract_s"] += per_item
        except Exception as e:
            dt = time.perf_counter() - t0
            per_item = dt / len(idxs) if idxs else 0.0
            err = {"type": type(e).__name__, "message": str(e)}
            for rid in idxs:
                rows[rid]["fail_reason"] = "extract_exception"
                rows[rid]["extract_error"] = err
                rows[rid]["timing"]["extract_s"] += per_item
                fail["extract_exception"] += 1

    check_candidates = [
        i for i, r in enumerate(rows) if r.get("claims") and r.get("reference") and r.get("fail_reason") is None
    ]
    for row in rows:
        if row.get("sentence") and row.get("reference") and row.get("fail_reason") is None and not row.get("claims"):
            row["stop_reason"] = "no_claims"
            stop["no_claims"] += 1

    for start in tqdm(range(0, len(check_candidates), args.batch_size_checker), desc="RefChecker check"):
        idxs = check_candidates[start : start + args.batch_size_checker]
        batch_claims = [rows[i]["claims"] for i in idxs]
        batch_references = [rows[i]["reference"] for i in idxs]
        t0 = time.perf_counter()
        try:
            batch_labels = checker.check(
                batch_claims=batch_claims,
                batch_references=batch_references,
                max_reference_segment_length=args.max_reference_segment_length,
            )
            dt = time.perf_counter() - t0
            per_item = dt / len(idxs) if idxs else 0.0
            for rid, labels in zip(idxs, batch_labels):
                norm_labels = [normalize_label(label) for label in (labels or [])]
                rows[rid]["verification"] = [
                    {"claim": claim, "label": label}
                    for claim, label in zip(rows[rid].get("claims") or [], norm_labels)
                ]
                rows[rid]["pred_supported_strict"] = strict_sentence_supported(norm_labels)
                rows[rid]["risk_not_supported_strict"] = sentence_risk_not_supported(norm_labels)
                rows[rid]["timing"]["verify_s"] += per_item
        except Exception as e:
            dt = time.perf_counter() - t0
            per_item = dt / len(idxs) if idxs else 0.0
            err = {"type": type(e).__name__, "message": str(e)}
            for rid in idxs:
                rows[rid]["fail_reason"] = "check_exception"
                rows[rid]["check_error"] = err
                rows[rid]["timing"]["verify_s"] += per_item
                fail["check_exception"] += 1

    for row in rows:
        row["timing"]["total_s"] = row["timing"]["extract_s"] + row["timing"]["verify_s"]

    return rows, fail, stop


def compute_metrics(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    fail: Dict[str, int],
    stop: Dict[str, int],
    n_examples: int,
) -> Dict[str, Any]:
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    n_eval = 0
    eval_rows = []

    for row in rows:
        gold = row.get("gold_supported")
        if gold is None:
            continue
        pred = row.get("pred_supported_strict")
        if pred is None:
            if args.no_claim_policy_all == "skip":
                continue
            pred_supported = False
        else:
            pred_supported = bool(pred)
        n_eval += 1
        eval_rows.append(row)
        update_cm_not_supported_positive(cm, bool(gold), pred_supported)

    sum_extract_s = sum(float(r["timing"]["extract_s"]) for r in eval_rows)
    sum_verify_s = sum(float(r["timing"]["verify_s"]) for r in eval_rows)
    sum_total_s = sum(float(r["timing"]["total_s"]) for r in eval_rows)
    sum_claims = sum(len(r.get("claims") or []) for r in eval_rows)

    metrics = {
        "dataset": args.dataset,
        "subset": args.subset if args.dataset == "felm" else None,
        "split": args.split if args.dataset == "felm" else None,
        "extractor_name": args.extractor_name,
        "checker_type": args.checker_type,
        "checker_name": args.checker_name if args.checker_type == "llm" else args.checker_type,
        "run_local_vllm": bool(args.run_local_vllm),
        "local_vllm": (
            {
                "model": args.local_vllm_model,
                "served_model_name": args.local_vllm_served_model_name or args.local_vllm_model,
                "api_base": args.extractor_api_base,
                "gpu_memory_utilization": args.local_vllm_gpu_memory_utilization,
                "tensor_parallel_size": args.local_vllm_tensor_parallel_size,
                "max_model_len": args.local_vllm_max_model_len or None,
                "dtype": args.local_vllm_dtype or None,
            }
            if args.run_local_vllm
            else None
        ),
        "claim_format": "RefChecker default triplets",
        "aggregation": "strict: supported iff every extracted claim label is Entailment",
        "no_claim_policy_all": args.no_claim_policy_all,
        "counts": {
            "n_examples": n_examples,
            "n_segments": len(rows),
            "n_eval": n_eval,
        },
        "fail_breakdown": fail,
        "stop_breakdown": stop,
        "all_sentences": {"n": n_eval, **cm_to_macro_f1(cm)},
        "efficiency": {
            "n_eval_rows": len(eval_rows),
            "avg_claims_per_sentence": safe_div(sum_claims, len(eval_rows)),
            "avg_extract_time_s_per_sentence": safe_div(sum_extract_s, len(eval_rows)),
            "avg_verify_time_s_per_sentence": safe_div(sum_verify_s, len(eval_rows)),
            "avg_total_time_s_per_sentence": safe_div(sum_total_s, len(eval_rows)),
            "sum_extract_time_s": sum_extract_s,
            "sum_verify_time_s": sum_verify_s,
            "sum_total_time_s": sum_total_s,
        },
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RefChecker on FactBench or FELM with strict sentence aggregation.")
    ap.add_argument("--dataset", choices=["factbench", "felm"], required=True)
    ap.add_argument("--out_root", type=str, default="out_refchecker")

    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--max_samples", type=int, default=0)

    ap.add_argument("--felm_dir", type=str, default="")
    ap.add_argument("--subset", type=str, default="writing_rec")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=0)

    ap.add_argument("--extractor_name", type=str, default="gpt-4o")
    ap.add_argument("--checker_type", choices=["llm", "nli", "alignscore"], default="llm")
    ap.add_argument("--checker_name", type=str, default="gpt-4o")
    ap.add_argument("--extractor_api_base", type=str, default="")
    ap.add_argument("--checker_api_base", type=str, default="")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size_extractor", type=int, default=8)
    ap.add_argument("--batch_size_checker", type=int, default=8)
    ap.add_argument("--extractor_max_new_tokens", type=int, default=500)
    ap.add_argument("--max_reference_segment_length", type=int, default=0)
    ap.add_argument("--no_claim_policy_all", choices=["skip", "penalize"], default="penalize")

    ap.add_argument(
        "--run_local_vllm",
        action="store_true",
        help="Start a local vLLM OpenAI-compatible server for RefChecker LLM extractor/checker.",
    )
    ap.add_argument("--local_vllm_model", type=str, default="VityaVitalich/Llama3.1-8b-instruct")
    ap.add_argument("--local_vllm_served_model_name", type=str, default="")
    ap.add_argument("--local_vllm_host", type=str, default="0.0.0.0")
    ap.add_argument("--local_vllm_port", type=int, default=5000)
    ap.add_argument("--local_vllm_gpu_memory_utilization", type=float, default=0.55)
    ap.add_argument("--local_vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--local_vllm_max_model_len", type=int, default=0)
    ap.add_argument("--local_vllm_dtype", type=str, default="")
    ap.add_argument("--local_vllm_trust_remote_code", action="store_true")
    ap.add_argument("--local_vllm_extra_args", type=str, default="")
    ap.add_argument("--local_vllm_startup_timeout_s", type=float, default=900.0)

    args = ap.parse_args()
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "factbench":
        if not args.data:
            raise RuntimeError("--data is required for dataset=factbench")
        rows, n_examples = load_factbench_rows(args)
    else:
        if not args.felm_dir:
            raise RuntimeError("--felm_dir is required for dataset=felm")
        rows, n_examples = load_felm_rows(args)

    with maybe_local_vllm(args):
        rows, fail, stop = run_refchecker_on_rows(args, rows)
        metrics = compute_metrics(args, rows, fail, stop, n_examples)

        save_json(metrics, out_dir / "metrics.json")
        save_jsonl(rows, out_dir / "segments.jsonl")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
