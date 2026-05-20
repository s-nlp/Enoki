import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from claimify_utils import (
    VERIFY_SYSTEM,
    BM25Lite,
    GenParams,
    LLMBackend,
    OpenRouterBackend,
    StageCfg,
    VLLMBackend,
    _is_eval_row,
    _safe_tflops_per_s,
    _sum_flops,
    _sum_time,
    add_flops,
    build_excerpt,
    build_messages,
    build_verify_user,
    clean,
    finalize_row_times_and_tokens,
    flatten_evidence,
    is_meta_rewrite,
    majority_normalized,
    parse_decomposition_output,
    parse_disambiguation_output,
    parse_selection_output,
    parse_verdict,
    ref_text_to_passages,
    run_stage_claimify,
    safe_div,
    sentence_risk_strict,
    strict_sentence_supported,
    tokenize,
)
from tqdm import tqdm

try:
    from settings import (
        CLAIMIFY_DECOMP_SYSTEM,
        CLAIMIFY_DECOMP_USER,
        CLAIMIFY_DISAMBIG_SYSTEM,
        CLAIMIFY_DISAMBIG_USER,
        CLAIMIFY_SELECTION_SYSTEM,
        CLAIMIFY_SELECTION_USER,
    )
except Exception as e:
    raise RuntimeError(
        "Could not import Claimify prompts from settings.py. "
        "Place settings.py next to this script, or adjust import."
    ) from e


_SENT_RE = re.compile(r"sentence(\d+)$")


def iter_sentences_in_order(
    sent_dict: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    items = []
    for k, v in (sent_dict or {}).items():
        m = _SENT_RE.search(k)
        idx = int(m.group(1)) if m else 10**9
        items.append((idx, k, v))
    items.sort(key=lambda t: t[0])
    return [(k, v) for _, k, v in items]


def norm_gold_label_factbench(x) -> Optional[bool]:
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
        try:
            return bool(int(x))
        except Exception:
            return None
    s = str(x).strip().lower()
    if s in {"n/a", "na", "none", "null", ""}:
        return None
    if s in {"supported", "support", "entailed", "entails", "true", "yes", "1"}:
        return True
    if s in {
        "not_supported",
        "not supported",
        "unsupported",
        "contradicted",
        "false",
        "no",
        "0",
    }:
        return False
    return None


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def save_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def prf1(tp, fp, fn):
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def update_cm_not_supported_positive(cm, gold_supported: bool, pred_supported: bool):
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


def cm_to_macro_f1(cm):
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


def build_backend(args) -> LLMBackend:
    if args.backend == "vllm":
        return VLLMBackend(
            model=args.model,
            tokenizer_name=args.tokenizer_name or args.model,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    if args.backend == "openrouter":
        key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError(
                "OpenRouter backend selected but no API key provided (OPENROUTER_API_KEY)."
            )
        return OpenRouterBackend(
            model=args.model,
            api_key=key,
            base_url=args.openrouter_base_url,
            http_referer=args.openrouter_http_referer,
            x_title=args.openrouter_x_title,
            timeout_s=args.openrouter_timeout_s,
            max_workers=args.openrouter_max_workers,
        )
    if args.backend == "openai":
        key = args.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "OpenAI backend selected but no API key provided (OPENAI_API_KEY)."
            )
        from claimify_utils import OpenAIBackend

        return OpenAIBackend(
            model=args.model,
            api_key=key,
            base_url=args.openai_base_url,
            timeout_s=args.openai_timeout_s,
            max_workers=args.openai_max_workers,
            use_developer_role=args.openai_use_developer_role,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def close_backend(b: LLMBackend):
    for name in ["close"]:
        if hasattr(b, name):
            try:
                getattr(b, name)()
            except Exception:
                pass


def make_empty_row(**base_fields: Any) -> Dict[str, Any]:
    row = {
        **base_fields,
        "claimify": {},
        "stop_reason": None,
        "fail_reason": None,
        "claims": [],
        "verification": [],
    }
    row["timing"] = {
        "selection_s": 0.0,
        "disambiguation_s": 0.0,
        "decomposition_s": 0.0,
        "extract_s": 0.0,
        "verify_s": 0.0,
        "total_s": 0.0,
    }
    row["tokens"] = {
        "selection_prompt": 0,
        "selection_gen": 0,
        "disambiguation_prompt": 0,
        "disambiguation_gen": 0,
        "decomposition_prompt": 0,
        "decomposition_gen": 0,
        "verify_prompt": 0,
        "verify_gen": 0,
        "total_prompt": 0,
        "total_gen": 0,
    }
    return row


def run_claimify_sentence_pipeline(
    *,
    backend: LLMBackend,
    row: Dict[str, Any],
    question: str,
    sentence: str,
    segs: List[str],
    sent_idx: int,
    sel_cfg: StageCfg,
    dis_cfg: StageCfg,
    dec_cfg: StageCfg,
    stop_common: List[str],
    max_claims: int,
    selection_mode: str,
    fail: Dict[str, int],
    stop: Dict[str, int],
) -> bool:
    if not sentence:
        row["fail_reason"] = "empty_sentence"
        fail["empty_sentence"] += 1
        return False

    excerpt = build_excerpt(segs, sent_idx, p=sel_cfg.p, f=sel_cfg.f)
    sel_texts, sel_parsed, sel_err, sel_t, sel_u = run_stage_claimify(
        backend=backend,
        cfg=sel_cfg,
        question=question,
        excerpt=excerpt,
        sentence=sentence,
        parse_fn=lambda t: parse_selection_output(t, sentence),
        is_parseable=lambda p: (
            p is not None and isinstance(p, tuple) and len(p) == 3 and p[0] is not None
        ),
        stop=stop_common,
    )
    row["timing"]["selection_s"] += sel_t
    row["tokens"]["selection_prompt"] += sel_u.prompt_tokens
    row["tokens"]["selection_gen"] += sel_u.gen_tokens

    if sel_err is not None or not sel_texts or not sel_parsed:
        row["fail_reason"] = "selection_failed"
        row["claimify"]["selection_error"] = sel_err
        fail["selection_failed"] += 1
        return False
    row["claimify"]["selection_raw"] = sel_texts
    row["claimify"]["selection_parsed"] = sel_parsed

    sel_successes: List[str] = []
    for p in sel_parsed:
        contains, selected, _decision = p
        if contains is True:
            if selection_mode == "detector":
                sel_successes.append(sentence)
            elif selected and not is_meta_rewrite(selected):
                sel_successes.append(selected)
            else:
                sel_successes.append(sentence)

    if len(sel_successes) < sel_cfg.min_successes:
        row["stop_reason"] = "selection_no_verifiable"
        stop["selection_no_verifiable"] += 1
        return False
    selected_sentence = majority_normalized(sel_successes)
    row["claimify"]["selected_sentence"] = selected_sentence

    excerpt = build_excerpt(segs, sent_idx, p=dis_cfg.p, f=dis_cfg.f)
    dis_texts, dis_parsed, dis_err, dis_t, dis_u = run_stage_claimify(
        backend=backend,
        cfg=dis_cfg,
        question=question,
        excerpt=excerpt,
        sentence=selected_sentence,
        parse_fn=parse_disambiguation_output,
        is_parseable=lambda p: (
            p is not None and isinstance(p, tuple) and len(p) == 2 and p[0] is not None
        ),
        stop=stop_common,
    )
    row["timing"]["disambiguation_s"] += dis_t
    row["tokens"]["disambiguation_prompt"] += dis_u.prompt_tokens
    row["tokens"]["disambiguation_gen"] += dis_u.gen_tokens

    if dis_err is not None or not dis_texts or not dis_parsed:
        row["fail_reason"] = "disambiguation_failed"
        row["claimify"]["disambiguation_error"] = dis_err
        fail["disambiguation_failed"] += 1
        return False
    row["claimify"]["disambiguation_raw"] = dis_texts
    row["claimify"]["disambiguation_parsed"] = dis_parsed

    dis_successes: List[str] = []
    saw_cannot = False
    for p in dis_parsed:
        status, sent2 = p
        if status == "cannot":
            saw_cannot = True
            continue
        if status == "ok" and sent2:
            dis_successes.append(sent2)

    if len(dis_successes) < dis_cfg.min_successes:
        row["stop_reason"] = (
            "disambiguation_cannot" if saw_cannot else "disambiguation_gate_failed"
        )
        stop[row["stop_reason"]] += 1
        return False
    dectx_sentence = majority_normalized(dis_successes)
    row["claimify"]["decontextualized_sentence"] = dectx_sentence

    excerpt = build_excerpt(segs, sent_idx, p=dec_cfg.p, f=dec_cfg.f)
    dec_texts, dec_parsed, dec_err, dec_t, dec_u = run_stage_claimify(
        backend=backend,
        cfg=dec_cfg,
        question=question,
        excerpt=excerpt,
        sentence=dectx_sentence,
        parse_fn=lambda t: parse_decomposition_output(t, max_claims=max_claims),
        is_parseable=lambda p: (
            p is not None and isinstance(p, tuple) and len(p) == 3 and p[0] is True
        ),
        stop=stop_common,
    )
    row["timing"]["decomposition_s"] += dec_t
    row["tokens"]["decomposition_prompt"] += dec_u.prompt_tokens
    row["tokens"]["decomposition_gen"] += dec_u.gen_tokens

    if dec_err is not None or not dec_texts or not dec_parsed:
        row["fail_reason"] = "decomposition_failed"
        row["claimify"]["decomposition_error"] = dec_err
        fail["decomposition_failed"] += 1
        return False

    _, claims, used_blk = dec_parsed[0]
    row["claims"] = claims
    row["claimify"]["decomposition_raw"] = dec_texts[0]
    row["claimify"]["decomposition_block"] = used_blk

    if not claims:
        row["stop_reason"] = "no_claims"
        stop["no_claims"] += 1
        return False

    return True


def queue_verification_for_claims(
    *,
    row: Dict[str, Any],
    rid: int,
    claims: List[str],
    passages: List[Dict[str, str]],
    bm25: Optional[BM25Lite],
    question: str,
    args: argparse.Namespace,
    verify_prompts: List[List[Dict[str, str]]],
    verify_meta: List[Tuple[int, str]],
    fail: Dict[str, int],
    flush_verify_batch,
) -> None:
    if not claims:
        return

    if not passages:
        err_key = "no_ref_text" if "no_ref_text" in fail else None
        if err_key is not None:
            fail[err_key] += 1
            err_label = "no_ref_text"
            default_label = None
        else:
            err_label = "no_passages"
            default_label = "not_supported"
        for c in claims:
            row["verification"].append(
                {"claim": c, "label": default_label, "error": err_label}
            )
        return

    for c in claims:
        if args.evidence_mode == "all":
            chosen = passages
        else:
            k = min(args.topk_passages, len(passages))
            query = c if args.bm25_query == "claim" else f"{question}\n{c}"
            idxs = bm25.topk(query, k=k) if bm25 else list(range(k))
            chosen = [passages[j] for j in idxs]

        user_v = build_verify_user(c, chosen)
        verify_prompts.append(build_messages(VERIFY_SYSTEM, user_v))
        verify_meta.append((rid, c))
        if len(verify_prompts) >= args.batch_size_verify:
            flush_verify_batch()


def main():
    ap = argparse.ArgumentParser()

    # dataset switch
    ap.add_argument("--dataset", choices=["factbench", "felm", "anah"], required=True)

    # factbench args
    ap.add_argument("--data", type=str, default="")  # jsonl
    ap.add_argument("--out_root", type=str, default="out_claimify")

    ap.add_argument("--max_samples", type=int, default=0)

    # felm args
    ap.add_argument("--felm_dir", type=str, default="")
    ap.add_argument("--subset", type=str, default="writing_rec")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=0)

    # anah args
    ap.add_argument(
        "--anah_split",
        type=str,
        default="train",
        help="HuggingFace split for ANAH. Only 'train' exists.",
    )
    ap.add_argument(
        "--anah_max_examples",
        type=int,
        default=0,
        help="Max ANAH rows to process. 0 = all.",
    )
    ap.add_argument(
        "--anah_sample_file",
        type=str,
        default="",
        help="Path to a pre-sampled ANAH jsonl (e.g. anah_250_sample.jsonl). "
             "If set, skips HuggingFace download and uses this file directly.",
    )

    # backend
    ap.add_argument(
        "--backend", choices=["vllm", "openrouter", "openai"], default="vllm"
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--tokenizer_name", type=str, default="")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.55)
    ap.add_argument(
        "--model_params_b",
        type=float,
        default=0.0,
        help="Model size in billions of parameters (for FLOPs estimate). Example: 8 for 8B.",
    )
    ap.add_argument(
        "--flops_per_param",
        type=float,
        default=2.0,
        help="FLOPs per parameter per token multiplier (rough). Common: 2 or 6.",
    )

    # openrouter
    ap.add_argument("--openrouter_api_key", type=str, default="")
    ap.add_argument(
        "--openrouter_base_url", type=str, default="https://openrouter.ai/api/v1"
    )
    ap.add_argument("--openrouter_http_referer", type=str, default="")
    ap.add_argument("--openrouter_x_title", type=str, default="")
    ap.add_argument("--openrouter_timeout_s", type=float, default=120.0)
    ap.add_argument("--openrouter_max_workers", type=int, default=8)

    # openai
    ap.add_argument("--openai_api_key", type=str, default="")
    ap.add_argument("--openai_base_url", type=str, default="https://api.openai.com/v1")
    ap.add_argument("--openai_timeout_s", type=float, default=120.0)
    ap.add_argument("--openai_max_workers", type=int, default=8)
    ap.add_argument(
        "--openai_use_developer_role",
        action="store_true",
        help="Map system->developer for OpenAI Chat Completions (recommended for some reasoning models).",
    )

    # claimify
    ap.add_argument("--max_claims", type=int, default=8)
    ap.add_argument("--stop_common", type=str, default="<|eot_id|>,</s>,<|im_end|>")

    ap.add_argument(
        "--selection_mode", choices=["rewrite", "detector"], default="rewrite"
    )

    # verification (optional)
    ap.add_argument("--do_verify", action="store_true")
    ap.add_argument("--evidence_mode", choices=["all", "bm25"], default="bm25")
    ap.add_argument(
        "--bm25_query", choices=["claim", "question_claim"], default="claim"
    )
    ap.add_argument("--max_passages", type=int, default=80)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--topk_passages", type=int, default=12)
    ap.add_argument("--batch_size_verify", type=int, default=16)
    ap.add_argument(
        "--verify_max_tokens",
        type=int,
        default=512,
        help="Max tokens for verification generation. 64 is too small for models that produce reasoning.",
    )

    ap.add_argument(
        "--evidence_scope", choices=["sentence", "sample"], default="sentence"
    )  # factbench only
    ap.add_argument(
        "--no_claim_policy_all", choices=["skip", "penalize"], default="skip"
    )

    args = ap.parse_args()
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    stop_common = [s.strip() for s in args.stop_common.split(",") if s.strip()]

    # Claimify stage cfgs (Appendix D)
    sel_cfg = StageCfg(
        name="selection",
        system=CLAIMIFY_SELECTION_SYSTEM,
        user_tmpl=CLAIMIFY_SELECTION_USER,
        p=5,
        f=5,
        n=3,
        min_successes=2,
        max_tokens=1024,
        max_retries=2,
    )
    dis_cfg = StageCfg(
        name="disambiguation",
        system=CLAIMIFY_DISAMBIG_SYSTEM,
        user_tmpl=CLAIMIFY_DISAMBIG_USER,
        p=5,
        f=0,
        n=3,
        min_successes=2,
        max_tokens=1536,
        max_retries=2,
    )
    dec_cfg = StageCfg(
        name="decomposition",
        system=CLAIMIFY_DECOMP_SYSTEM,
        user_tmpl=CLAIMIFY_DECOMP_USER,
        p=5,
        f=0,
        n=1,
        min_successes=1,
        max_tokens=768,
        max_retries=2,
    )

    backend = build_backend(args)

    # Verification batching works for both backends:
    verify_prompts: List[List[Dict[str, str]]] = []
    verify_meta: List[Tuple[int, str]] = []

    # use explicit GP (temperature 0)
    if "gpt-5" in args.model.lower():
        max_verify_tokens = max(args.verify_max_tokens, 1024)
    else:
        max_verify_tokens = args.verify_max_tokens
    gp_verify = GenParams(temperature=0.01, max_tokens=max_verify_tokens, stop=stop_common, n=1)

    segments_out: List[Dict[str, Any]] = []  # per-run output

    def flush_verify_batch():
        nonlocal verify_prompts, verify_meta
        if not verify_prompts:
            return
        n = len(verify_prompts)
        t0 = time.perf_counter()
        try:
            br = backend.generate_with_usage(verify_prompts, gp_verify)
            dt = time.perf_counter() - t0
            per_item_time = dt / n if n else 0.0

            usages = br.usage or [None] * n

            # VALIDATION: Check for empty responses
            for i, (out_list, (rid, claim), u) in enumerate(
                zip(br.texts, verify_meta, usages)
            ):
                if not out_list or not out_list[0].strip():
                    print(
                        f"WARNING: Empty verification response for claim {i+1}/{n}: {claim[:50]}..."
                    )
                    txt = ""
                else:
                    txt = out_list[0]

                lab = parse_verdict(txt)
                segments_out[rid]["verification"].append(
                    {"claim": claim, "label": lab, "raw_stripped": txt}
                )
                segments_out[rid]["timing"]["verify_s"] += per_item_time

                if u is not None:
                    segments_out[rid]["tokens"]["verify_prompt"] += u.prompt_tokens
                    segments_out[rid]["tokens"]["verify_gen"] += u.gen_tokens

        except Exception as e:
            dt = time.perf_counter() - t0
            per_item_time = dt / n if n else 0.0
            err = {"type": type(e).__name__, "message": str(e)}
            for rid, claim in verify_meta:
                segments_out[rid]["verification"].append(
                    {"claim": claim, "label": None, "raw_stripped": None, "error": err}
                )
                segments_out[rid]["timing"]["verify_s"] += per_item_time

        verify_prompts, verify_meta = [], []

    # DATASET: FACTBENCH
    if args.dataset == "factbench":
        segments_out = []
        if not args.data:
            raise RuntimeError("--data is required for dataset=factbench")

        samples: List[Dict[str, Any]] = []
        with open(args.data, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.max_samples and i >= args.max_samples:
                    break
                samples.append(json.loads(line))

        # Precompute passages
        topic2pass_sentence: Dict[Tuple[int, str], List[Dict[str, str]]] = {}
        topic2pass_sample: Dict[int, List[Dict[str, str]]] = {}
        keys_by_sample: List[List[str]] = []
        texts_by_sample: List[List[str]] = []
        golds_by_sample: List[List[Optional[bool]]] = []
        prompts_by_sample: List[str] = []

        for sid, ex in enumerate(samples):
            prompts_by_sample.append(clean(ex.get("prompt") or ""))

            sent_items = iter_sentences_in_order(ex.get("sentences") or {})
            keys: List[str] = []
            texts: List[str] = []
            golds: List[Optional[bool]] = []

            ev_chunks_sample: List[str] = []

            for sent_key, s in sent_items:
                keys.append(sent_key)
                sent_text = clean((s.get("text") or ""))
                texts.append(sent_text)
                golds.append(
                    norm_gold_label_factbench(s.get("sentence_factuality_label"))
                )

                ev_chunks_sentence: List[str] = []
                ev_chunks_sentence += flatten_evidence(s.get("auto_evidence"))
                ev_chunks_sentence += flatten_evidence(s.get("human_evidence"))

                if args.evidence_scope == "sentence":
                    ctx_sentence = "\n\n".join(ev_chunks_sentence)
                    topic2pass_sentence[(sid, sent_key)] = ref_text_to_passages(
                        topic=f"factbench::{sid}",
                        ref_text=ctx_sentence,
                        max_passages=args.max_passages,
                        max_chars=args.max_chars,
                    )

                ev_chunks_sample.extend(ev_chunks_sentence)

            if args.evidence_scope == "sample":
                ctx_sample = "\n\n".join(ev_chunks_sample)
                topic2pass_sample[sid] = ref_text_to_passages(
                    topic=f"factbench::{sid}",
                    ref_text=ctx_sample,
                    max_passages=args.max_passages,
                    max_chars=args.max_chars,
                )

            keys_by_sample.append(keys)
            texts_by_sample.append(texts)
            golds_by_sample.append(golds)

        # run
        fail = {
            "empty_sentence": 0,
            "selection_failed": 0,
            "disambiguation_failed": 0,
            "decomposition_failed": 0,
        }
        stop = {
            "selection_no_verifiable": 0,
            "disambiguation_cannot": 0,
            "disambiguation_gate_failed": 0,
            "no_claims": 0,
        }

        for sid, ex in enumerate(tqdm(samples, desc="Claimify FactBench")):
            question = prompts_by_sample[sid]
            keys = keys_by_sample[sid]
            segs = texts_by_sample[sid]
            golds = golds_by_sample[sid]

            # sample bm25
            psgs_sample = (
                topic2pass_sample.get(sid, [])
                if args.evidence_scope == "sample"
                else []
            )
            bm25_sample = None
            if (
                args.do_verify
                and args.evidence_mode == "bm25"
                and args.evidence_scope == "sample"
                and psgs_sample
            ):
                bm25_sample = BM25Lite(
                    [tokenize(p["title"] + " " + p["snippet"]) for p in psgs_sample]
                )

            for i, (sent_key, sentence) in enumerate(zip(keys, segs)):
                gold_supported = golds[i] if i < len(golds) else None

                rid = len(segments_out)
                row = make_empty_row(
                    sample_id=sid,
                    sentence_key=sent_key,
                    question=question,
                    sentence=sentence,
                    gold_supported=gold_supported,
                )
                segments_out.append(row)

                if gold_supported is None:
                    row["stop_reason"] = "gold_NA"
                    continue

                if not sentence:
                    row["fail_reason"] = "empty_sentence"
                    fail["empty_sentence"] += 1
                    continue

                # passages (sentence vs sample)
                if args.evidence_scope == "sample":
                    passages = psgs_sample
                    bm25 = bm25_sample
                else:
                    passages = topic2pass_sentence.get((sid, sent_key), [])
                    bm25 = None
                    if args.do_verify and args.evidence_mode == "bm25" and passages:
                        bm25 = BM25Lite(
                            [
                                tokenize(p["title"] + " " + p["snippet"])
                                for p in passages
                            ]
                        )

                ok = run_claimify_sentence_pipeline(
                    backend=backend,
                    row=row,
                    question=question,
                    sentence=sentence,
                    segs=segs,
                    sent_idx=i,
                    sel_cfg=sel_cfg,
                    dis_cfg=dis_cfg,
                    dec_cfg=dec_cfg,
                    stop_common=stop_common,
                    max_claims=args.max_claims,
                    selection_mode=args.selection_mode,
                    fail=fail,
                    stop=stop,
                )
                if not ok:
                    continue

                if args.do_verify:
                    queue_verification_for_claims(
                        row=row,
                        rid=rid,
                        claims=row["claims"],
                        passages=passages,
                        bm25=bm25,
                        question=question,
                        args=args,
                        verify_prompts=verify_prompts,
                        verify_meta=verify_meta,
                        fail=fail,
                        flush_verify_batch=flush_verify_batch,
                    )

        flush_verify_batch()
        for r in segments_out:
            # замеряем общее время, токены, FLOPs
            finalize_row_times_and_tokens(r)
            add_flops(r, args.model_params_b, args.flops_per_param)

        eval_rows = [r for r in segments_out if _is_eval_row(r)]
        n_eval_rows = len(eval_rows)
        sum_claims = sum(len(r.get("claims") or []) for r in eval_rows)

        sum_extract_s = _sum_time(eval_rows, "extract_s")
        sum_verify_s = _sum_time(eval_rows, "verify_s")
        sum_total_s = _sum_time(eval_rows, "total_s")

        sum_extract_flops = _sum_flops(eval_rows, "extract_flops")
        sum_verify_flops = _sum_flops(eval_rows, "verify_flops")
        sum_total_flops = _sum_flops(eval_rows, "total_flops")

        # Metrics
        cm_all = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        n_all = 0
        for r in segments_out:
            gold = r.get("gold_supported")
            if gold is None:
                continue
            strict_pred = None
            if args.do_verify:
                labels = [d.get("label") for d in (r.get("verification") or [])]
                strict_pred = strict_sentence_supported(labels)
                r["pred_supported_strict"] = strict_pred
                r["risk_not_supported_strict"] = sentence_risk_strict(labels)
            else:
                r["pred_supported_strict"] = None
                r["risk_not_supported_strict"] = None

            if strict_pred is None:
                if args.no_claim_policy_all == "skip":
                    continue
                pred_supported = False
            else:
                pred_supported = bool(strict_pred)

            n_all += 1
            update_cm_not_supported_positive(cm_all, bool(gold), pred_supported)

        metrics = {
            "dataset": "factbench",
            "backend": args.backend,
            "model": args.model,
            "do_verify": bool(args.do_verify),
            "evidence_mode": args.evidence_mode if args.do_verify else None,
            "evidence_scope": args.evidence_scope if args.do_verify else None,
            "bm25_query": args.bm25_query if args.do_verify else None,
            "selection_mode": args.selection_mode,
            "no_claim_policy_all": args.no_claim_policy_all,
            "counts": {
                "n_samples": len(samples),
                "n_segments": len(segments_out),
                "n_eval": n_all,
            },
            "fail_breakdown": fail,
            "stop_breakdown": stop,
            "all_sentences": {"n": n_all, **cm_to_macro_f1(cm_all)},
            "efficiency": {
                "n_eval_rows": n_eval_rows,
                "avg_claims_per_sentence": safe_div(sum_claims, n_eval_rows),
                "avg_extract_time_s_per_sentence": safe_div(sum_extract_s, n_eval_rows),
                "avg_verify_time_s_per_sentence": safe_div(sum_verify_s, n_eval_rows),
                "avg_total_time_s_per_sentence": safe_div(sum_total_s, n_eval_rows),
            },
            "compute": (
                None
                if args.model_params_b <= 0
                else {
                    "params_b": args.model_params_b,
                    "flops_per_param": args.flops_per_param,
                    "sum_extract_flops": sum_extract_flops,
                    "sum_verify_flops": sum_verify_flops,
                    "sum_total_flops": sum_total_flops,
                    "extract_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_extract_flops, sum_extract_s
                    ),
                    "verify_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_verify_flops, sum_verify_s
                    ),
                    "total_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_total_flops, sum_total_s
                    ),
                    "sum_extract_time_s": sum_extract_s,
                    "sum_verify_time_s": sum_verify_s,
                    "sum_total_time_s": sum_total_s,
                }
            ),
        }

        save_json(metrics, out_dir / "metrics.json")
        save_jsonl(segments_out, out_dir / "segments.jsonl")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        close_backend(backend)
        return

    # DATASET: FELM
    if args.dataset == "felm":
        segments_out = []
        if not args.felm_dir:
            raise RuntimeError("--felm_dir is required for dataset=felm")

        try:
            from datasets import load_from_disk
        except Exception as e:
            raise RuntimeError(
                "FELM runner requires `datasets` (pip install datasets)."
            ) from e

        root = Path(args.felm_dir)
        p_split = root / args.subset / args.split
        if p_split.exists():
            ds = load_from_disk(str(p_split))
        else:
            obj = load_from_disk(str(root / args.subset))
            ds = obj[args.split]

        if args.max_examples and args.max_examples > 0:
            ds = ds.select(range(min(args.max_examples, len(ds))))

        fail = {
            "empty_sentence": 0,
            "selection_failed": 0,
            "disambiguation_failed": 0,
            "decomposition_failed": 0,
            "no_ref_text": 0,
        }
        stop = {
            "selection_no_verifiable": 0,
            "disambiguation_cannot": 0,
            "disambiguation_gate_failed": 0,
            "no_claims": 0,
        }

        for ex in tqdm(ds, desc=f"Claimify FELM {args.subset}/{args.split}"):
            question = clean(ex.get("prompt") or "")
            segs = [clean(s) for s in (ex.get("segmented_response") or [])]
            golds = [to_gold_supported_felm(x) for x in (ex.get("labels") or [])]
            ex_index = ex.get("index", None)

            passages = []
            bm25 = None
            if args.do_verify:
                ref_text = ex.get("ref_text") or ""
                passages = ref_text_to_passages(
                    topic=f"felm::{args.subset}",
                    ref_text=ref_text,
                    max_passages=args.max_passages,
                    max_chars=args.max_chars,
                )
                if args.evidence_mode == "bm25" and passages:
                    bm25 = BM25Lite(
                        [tokenize(p["title"] + " " + p["snippet"]) for p in passages]
                    )

            for i, sentence in enumerate(segs):
                gold_supported = golds[i] if i < len(golds) else None
                rid = len(segments_out)
                row = make_empty_row(
                    example_index=ex_index,
                    seg_id=i,
                    question=question,
                    sentence=sentence,
                    gold_supported=gold_supported,
                )
                segments_out.append(row)

                ok = run_claimify_sentence_pipeline(
                    backend=backend,
                    row=row,
                    question=question,
                    sentence=sentence,
                    segs=segs,
                    sent_idx=i,
                    sel_cfg=sel_cfg,
                    dis_cfg=dis_cfg,
                    dec_cfg=dec_cfg,
                    stop_common=stop_common,
                    max_claims=args.max_claims,
                    selection_mode=args.selection_mode,
                    fail=fail,
                    stop=stop,
                )
                if not ok:
                    continue

                if args.do_verify:
                    queue_verification_for_claims(
                        row=row,
                        rid=rid,
                        claims=row["claims"],
                        passages=passages,
                        bm25=bm25,
                        question=question,
                        args=args,
                        verify_prompts=verify_prompts,
                        verify_meta=verify_meta,
                        fail=fail,
                        flush_verify_batch=flush_verify_batch,
                    )

        flush_verify_batch()
        for r in segments_out:
            # замеряем общее время, токены, FLOPs
            finalize_row_times_and_tokens(r)
            add_flops(r, args.model_params_b, args.flops_per_param)

        eval_rows = [r for r in segments_out if _is_eval_row(r)]
        n_eval_rows = len(eval_rows)
        sum_claims = sum(len(r.get("claims") or []) for r in eval_rows)

        sum_extract_s = _sum_time(eval_rows, "extract_s")
        sum_verify_s = _sum_time(eval_rows, "verify_s")
        sum_total_s = _sum_time(eval_rows, "total_s")

        sum_extract_flops = _sum_flops(eval_rows, "extract_flops")
        sum_verify_flops = _sum_flops(eval_rows, "verify_flops")
        sum_total_flops = _sum_flops(eval_rows, "total_flops")

        # Metrics (strict sentence)
        cm_all = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        n_all = 0
        for r in segments_out:
            gold = r.get("gold_supported")
            if gold is None:
                continue

            strict_pred = None
            if args.do_verify:
                labels = [d.get("label") for d in (r.get("verification") or [])]
                strict_pred = strict_sentence_supported(labels)
                r["pred_supported_strict"] = strict_pred
                r["risk_not_supported_strict"] = sentence_risk_strict(labels)
            else:
                r["pred_supported_strict"] = None
                r["risk_not_supported_strict"] = None

            if strict_pred is None:
                if args.no_claim_policy_all == "skip":
                    continue
                pred_supported = False
            else:
                pred_supported = bool(strict_pred)

            n_all += 1
            update_cm_not_supported_positive(cm_all, bool(gold), pred_supported)

        metrics = {
            "dataset": "felm",
            "subset": args.subset,
            "split": args.split,
            "backend": args.backend,
            "model": args.model,
            "do_verify": bool(args.do_verify),
            "evidence_mode": args.evidence_mode if args.do_verify else None,
            "bm25_query": args.bm25_query if args.do_verify else None,
            "selection_mode": args.selection_mode,
            "no_claim_policy_all": args.no_claim_policy_all,
            "counts": {
                "n_examples": len(ds),
                "n_segments": len(segments_out),
                "n_eval": n_all,
            },
            "fail_breakdown": fail,
            "stop_breakdown": stop,
            "all_sentences": {"n": n_all, **cm_to_macro_f1(cm_all)},
            "efficiency": {
                "n_eval_rows": n_eval_rows,
                "avg_claims_per_sentence": safe_div(sum_claims, n_eval_rows),
                "avg_extract_time_s_per_sentence": safe_div(sum_extract_s, n_eval_rows),
                "avg_verify_time_s_per_sentence": safe_div(sum_verify_s, n_eval_rows),
                "avg_total_time_s_per_sentence": safe_div(sum_total_s, n_eval_rows),
            },
            "compute": (
                None
                if args.model_params_b <= 0
                else {
                    "params_b": args.model_params_b,
                    "flops_per_param": args.flops_per_param,
                    "sum_extract_flops": sum_extract_flops,
                    "sum_verify_flops": sum_verify_flops,
                    "sum_total_flops": sum_total_flops,
                    "extract_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_extract_flops, sum_extract_s
                    ),
                    "verify_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_verify_flops, sum_verify_s
                    ),
                    "total_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_total_flops, sum_total_s
                    ),
                    "sum_extract_time_s": sum_extract_s,
                    "sum_verify_time_s": sum_verify_s,
                    "sum_total_time_s": sum_total_s,
                }
            ),
        }

        save_json(metrics, out_dir / "metrics.json")
        save_jsonl(segments_out, out_dir / "segments.jsonl")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        close_backend(backend)
        return

    if args.dataset == "anah":
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
        from anah_utils import iter_anah_sentences  # noqa: PLC0415

        if args.anah_sample_file:
            # Load from pre-sampled jsonl — skip HuggingFace entirely
            with open(args.anah_sample_file, encoding="utf-8") as _f:
                _sample_rows = [json.loads(l) for l in _f if l.strip()]
            anah_iter = iter(_sample_rows)
            n_examples_total = len(_sample_rows)
            split_label = _os.path.basename(args.anah_sample_file)
        else:
            try:
                from datasets import load_dataset as _load_anah  # noqa: PLC0415
            except Exception as exc:
                raise RuntimeError(
                    "datasets is not installed, but --dataset anah requires it: pip install datasets"
                ) from exc
            ds = _load_anah("opencompass/anah", split=args.anah_split)
            anah_iter = iter_anah_sentences(ds, max_examples=args.anah_max_examples or 0)
            n_examples_total = len(ds)
            split_label = args.anah_split

        out_dir = Path(args.out_root) / "anah" / (
            _os.path.splitext(_os.path.basename(args.anah_sample_file))[0]
            if args.anah_sample_file else args.anah_split
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        segments_out = []
        verify_prompts: List = []
        verify_meta: List[Tuple[int, str]] = []

        fail = {
            "empty_sentence": 0,
            "selection_failed": 0,
            "disambiguation_failed": 0,
            "decomposition_failed": 0,
            "no_ref_text": 0,
        }
        stop = {
            "selection_no_verifiable": 0,
            "disambiguation_cannot": 0,
            "disambiguation_gate_failed": 0,
            "no_claims": 0,
        }

        skipped_no_fact = 0

        for sent_row in tqdm(
            anah_iter,
            total=n_examples_total,
            desc=f"Claimify ANAH {split_label}",
        ):
            gold_supported: Optional[bool] = sent_row["gold_supported"]
            if gold_supported is None:
                skipped_no_fact += 1
                continue  # No Fact – skip

            question = clean(sent_row["question"])
            sentence = clean(sent_row["sentence"])
            hallucination_type = sent_row["hallucination_type"]
            # Use ann_reference (the specific cited fragment) as the per-sentence evidence
            reference = sent_row["ann_reference"]
            ex_i = sent_row["example_index"]

            # Build passage pool from the reference fragment
            passages = []
            bm25 = None
            if args.do_verify:
                passages = ref_text_to_passages(
                    topic=f"anah::{ex_i}_{sent_row['answer_index']}_{sent_row['sentence_index']}",
                    ref_text=reference,
                    max_passages=args.max_passages,
                    max_chars=args.max_chars,
                )
                if args.evidence_mode == "bm25" and passages:
                    bm25 = BM25Lite(
                        [tokenize(p["title"] + " " + p["snippet"]) for p in passages]
                    )

            segs = [
                clean(s)
                for s in (sent_row.get("answer_sentences") or [])
                if clean(s)
            ]
            sent_idx = int(sent_row.get("sentence_index", 0) or 0)
            if not segs:
                segs = [sentence]
                sent_idx = 0
            elif sent_idx >= len(segs):
                segs = [sentence]
                sent_idx = 0
            rid = len(segments_out)
            row = make_empty_row(
                example_index=ex_i,
                answer_index=sent_row["answer_index"],
                sentence_index=sent_row["sentence_index"],
                hallucination_type=hallucination_type,
                question=question,
                sentence=sentence,
                gold_supported=gold_supported,
            )
            segments_out.append(row)

            ok = run_claimify_sentence_pipeline(
                backend=backend,
                row=row,
                question=question,
                sentence=sentence,
                segs=segs,
                sent_idx=sent_idx,
                sel_cfg=sel_cfg,
                dis_cfg=dis_cfg,
                dec_cfg=dec_cfg,
                stop_common=stop_common,
                max_claims=args.max_claims,
                selection_mode=args.selection_mode,
                fail=fail,
                stop=stop,
            )
            if not ok:
                continue

            if args.do_verify:
                queue_verification_for_claims(
                    row=row,
                    rid=rid,
                    claims=row["claims"],
                    passages=passages,
                    bm25=bm25,
                    question=question,
                    args=args,
                    verify_prompts=verify_prompts,
                    verify_meta=verify_meta,
                    fail=fail,
                    flush_verify_batch=flush_verify_batch,
                )

        flush_verify_batch()
        for r in segments_out:
            finalize_row_times_and_tokens(r)
            add_flops(r, args.model_params_b, args.flops_per_param)

        eval_rows = [r for r in segments_out if _is_eval_row(r)]
        n_eval_rows = len(eval_rows)
        sum_claims = sum(len(r.get("claims") or []) for r in eval_rows)
        sum_extract_s = _sum_time(eval_rows, "extract_s")
        sum_verify_s = _sum_time(eval_rows, "verify_s")
        sum_total_s = _sum_time(eval_rows, "total_s")
        sum_extract_flops = _sum_flops(eval_rows, "extract_flops")
        sum_verify_flops = _sum_flops(eval_rows, "verify_flops")
        sum_total_flops = _sum_flops(eval_rows, "total_flops")

        cm_all = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        n_all = 0
        for r in segments_out:
            gold = r.get("gold_supported")
            if gold is None:
                continue
            strict_pred = None
            if args.do_verify:
                labels = [d.get("label") for d in (r.get("verification") or [])]
                strict_pred = strict_sentence_supported(labels)
                r["pred_supported_strict"] = strict_pred
                r["risk_not_supported_strict"] = sentence_risk_strict(labels)
            else:
                r["pred_supported_strict"] = None
                r["risk_not_supported_strict"] = None

            if strict_pred is None:
                if args.no_claim_policy_all == "skip":
                    continue
                pred_supported = False
            else:
                pred_supported = bool(strict_pred)

            n_all += 1
            update_cm_not_supported_positive(cm_all, bool(gold), pred_supported)

        metrics = {
            "dataset": "anah",
            "split": split_label,
            "backend": args.backend,
            "model": args.model,
            "do_verify": bool(args.do_verify),
            "evidence_mode": args.evidence_mode if args.do_verify else None,
            "bm25_query": args.bm25_query if args.do_verify else None,
            "selection_mode": args.selection_mode,
            "no_claim_policy_all": args.no_claim_policy_all,
            "skipped_no_fact": skipped_no_fact,
            "counts": {
                "n_examples": n_examples_total,
                "n_segments": len(segments_out),
                "n_eval": n_all,
            },
            "fail_breakdown": fail,
            "stop_breakdown": stop,
            "all_sentences": {"n": n_all, **cm_to_macro_f1(cm_all)},
            "efficiency": {
                "n_eval_rows": n_eval_rows,
                "avg_claims_per_sentence": safe_div(sum_claims, n_eval_rows),
                "avg_extract_time_s_per_sentence": safe_div(sum_extract_s, n_eval_rows),
                "avg_verify_time_s_per_sentence": safe_div(sum_verify_s, n_eval_rows),
                "avg_total_time_s_per_sentence": safe_div(sum_total_s, n_eval_rows),
            },
            "compute": (
                None
                if args.model_params_b <= 0
                else {
                    "params_b": args.model_params_b,
                    "flops_per_param": args.flops_per_param,
                    "sum_extract_flops": sum_extract_flops,
                    "sum_verify_flops": sum_verify_flops,
                    "sum_total_flops": sum_total_flops,
                    "extract_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_extract_flops, sum_extract_s
                    ),
                    "verify_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_verify_flops, sum_verify_s
                    ),
                    "total_tflops_per_s_agg": _safe_tflops_per_s(
                        sum_total_flops, sum_total_s
                    ),
                    "sum_extract_time_s": sum_extract_s,
                    "sum_verify_time_s": sum_verify_s,
                    "sum_total_time_s": sum_total_s,
                }
            ),
        }

        save_json(metrics, out_dir / "metrics.json")
        save_jsonl(segments_out, out_dir / "segments.jsonl")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\nSaved to: {out_dir.resolve()}")

        close_backend(backend)
        return


if __name__ == "__main__":
    main()
