import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from veriscore_utils import (
    AutoTokenizer,
    VLLMSession,
    _add_vllm_tokens,
    _agg_compute,
    _agg_efficiency,
    _finalize_eff_row,
    _init_eff_fields,
    build_bm25_index,
    build_candidate_passages_factbench,
    build_chat_prompt_with_tokenizer,
    build_extraction_snippet_with_window_factbench,
    build_extraction_snippet_with_window_felm,
    build_search_results_str,
    clean_seg,
    cm_to_macro_f1,
    dedup_claims,
    f1_at_k,
    fill_verification_fewshot_template,
    iter_sentences_in_order,
    load_fewshot_jsonl,
    median_int,
    norm_gold_label,
    parse_claims_from_extraction,
    parse_verdict_strict,
    parse_verdict_with_fallback,
    read_jsonl,
    read_text,
    ref_text_to_passages,
    roc_auc_manual,
    score_not_supported_from_label,
    select_topk_passages_bm25_indexed,
    sentence_risk_not_supported,
    strict_sentence_supported,
    topic_from_ref_or_prompt,
    update_confusion_not_supported_positive,
    vllm_generate_with_retries,
    wrap_two_slot_template_checked,
    write_json,
    write_jsonl,
)
from vllm import SamplingParams

# FELM only
try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None


def resolve_veriscore_assets(args) -> None:
    """
    If --veriscore_assets_dir is provided, fill missing asset paths in Claimify/VeriScore layout:
      prompt/non_qa_template.txt
      prompt/verification_instruction_binary.txt
      data/demos/few_shot_examples.jsonl
    """
    if not getattr(args, "veriscore_assets_dir", ""):
        return
    root = Path(args.veriscore_assets_dir)

    if not getattr(args, "extraction_template", ""):
        args.extraction_template = str(root / "prompt" / "non_qa_template.txt")
    if not getattr(args, "verification_instruction_binary", ""):
        args.verification_instruction_binary = str(
            root / "prompt" / "verification_instruction_binary.txt"
        )
    if not getattr(args, "fewshot_jsonl", ""):
        args.fewshot_jsonl = str(root / "data" / "demos" / "few_shot_examples.jsonl")


def ensure_assets_exist(*paths: str) -> None:
    for p in paths:
        if not p or not Path(p).exists():
            raise FileNotFoundError(f"Missing asset file: {p}")


def make_sampling_params(
    temperature: float,
    max_tokens_extract: int,
    max_tokens_verify: int,
    stop_extract: List[str],
    stop_verify: List[str],
) -> Tuple[SamplingParams, SamplingParams]:
    sp_extract = SamplingParams(
        temperature=temperature, max_tokens=max_tokens_extract, stop=stop_extract
    )
    sp_verify = SamplingParams(
        temperature=temperature, max_tokens=max_tokens_verify, stop=stop_verify
    )
    return sp_extract, sp_verify


def build_prompt_wrappers(
    *,
    use_alpaca_template: bool,
    alpaca_template_txt: str,
    tokenizer,
    system_extract: str,
    system_verify: str,
):
    """
    Returns two callables:
      wrap_extract(user_msg) -> prompt
      wrap_verify(user_msg) -> prompt
    """
    if use_alpaca_template:

        def wrap_extract(user_msg: str) -> str:
            return wrap_two_slot_template_checked(
                alpaca_template_txt,
                system_extract,
                user_msg,
                must_contain_system=system_extract,
            )

        def wrap_verify(user_msg: str) -> str:
            return wrap_two_slot_template_checked(
                alpaca_template_txt,
                system_verify,
                user_msg,
                must_contain_system=system_verify,
            )

        return wrap_extract, wrap_verify

    # tokenizer-based
    def wrap_extract(user_msg: str) -> str:
        return build_chat_prompt_with_tokenizer(tokenizer, system_extract, user_msg)

    def wrap_verify(user_msg: str) -> str:
        return build_chat_prompt_with_tokenizer(tokenizer, system_verify, user_msg)

    return wrap_extract, wrap_verify


# FACTBENCH runner
def run_factbench(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolve_veriscore_assets(args)
    ensure_assets_exist(
        args.extraction_template,
        args.verification_instruction_binary,
        args.fewshot_jsonl,
    )

    extraction_template = read_text(Path(args.extraction_template))
    verif_template = read_text(Path(args.verification_instruction_binary))
    fewshot_rows = load_fewshot_jsonl(Path(args.fewshot_jsonl))
    prompt_initial_temp = fill_verification_fewshot_template(
        verif_template, fewshot_rows
    )

    # Prompt wrapper selection
    alpaca_template_txt = ""
    tokenizer = None
    use_alpaca = bool(args.alpaca_template)
    if use_alpaca:
        alpaca_template_txt = read_text(Path(args.alpaca_template))
    else:
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required for tokenizer-based chat prompts. Install transformers or pass --alpaca_template."
            )
        tok_name = args.tokenizer_name or args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tok_name, trust_remote_code=args.trust_remote_code
        )

    wrap_extract, wrap_verify = build_prompt_wrappers(
        use_alpaca_template=use_alpaca,
        alpaca_template_txt=alpaca_template_txt,
        tokenizer=tokenizer,
        system_extract=args.system_extract,
        system_verify=args.system_verify,
    )

    # Stops
    stop_common = [s.strip() for s in args.stop_common.split(",") if s.strip()]
    stop_extract_extra = [
        s.encode("utf-8").decode("unicode_escape")
        for s in args.stop_extract_extra.split(",")
        if s.strip()
    ]
    stop_extract = stop_common + stop_extract_extra
    stop_verify = stop_common

    sp_extract, sp_verify = make_sampling_params(
        temperature=args.temperature,
        max_tokens_extract=args.max_tokens_extract,
        max_tokens_verify=args.max_tokens_verify,
        stop_extract=stop_extract,
        stop_verify=stop_verify,
    )

    rows_out: List[Dict[str, Any]] = []

    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    y_true: List[int] = []
    y_score: List[float] = []
    sentence_claim_counts: List[int] = []

    # Verification batching buffers
    batch_prompts: List[str] = []
    batch_meta: List[Tuple[int, str, List[str]]] = []  # (row_idx, claim, top3_snips)

    def flush_verify_batch(llm):
        nonlocal batch_prompts, batch_meta
        if not batch_prompts:
            return

        t0 = time.perf_counter()
        outs, err = vllm_generate_with_retries(
            llm, batch_prompts, sp_verify, max_tries=args.max_tries
        )
        dt = time.perf_counter() - t0
        n = len(batch_prompts)
        per_item = dt / n if n else 0.0

        # If outs is None or error, mark all as failed
        if err is not None or outs is None:
            for rid, claim, top3 in batch_meta:
                rows_out[rid]["timing"]["verify_s"] += per_item
                rows_out[rid]["verification_details"].append(
                    {
                        "claim": claim,
                        "label": None,
                        "score_not_supported": 1.0,
                        "llm_raw": None,
                        "llm_error": err,
                        "top_evidence_claim_query": top3,
                    }
                )
            batch_prompts, batch_meta = [], []
            return
        # Robust alignment (handle unexpected length mismatch)
        if len(outs) != len(batch_meta):
            mismatch_err = {
                "type": "length_mismatch",
                "message": f"vLLM returned {len(outs)} outs for {len(batch_meta)} prompts",
            }
        else:
            mismatch_err = None
        # Count tokens ONCE for the entire batch
        if outs and len(outs) > 0:
            # The first output has the shared prompt tokens (same for all in batch)
            shared_prompt_tokens = len(getattr(outs[0], "prompt_token_ids", []) or [])

            # Distribute prompt tokens evenly across claims
            n_items = len(batch_meta)
            per_claim_prompt = shared_prompt_tokens // n_items

            for j in range(len(batch_meta)):
                rid, claim, top3 = batch_meta[j]
                rows_out[rid]["timing"]["verify_s"] += per_item
                # Add prompt tokens (distributed from shared batch)
                rows_out[rid]["tokens"]["verify_prompt"] += per_claim_prompt

                out = outs[j] if j < len(outs) else None
                if out is not None:
                    # Count generation tokens for THIS specific claim
                    gen_tok = 0
                    for o in getattr(out, "outputs", None) or []:
                        gen_tok += len(getattr(o, "token_ids", []) or [])
                    rows_out[rid]["tokens"]["verify_gen"] += gen_tok

                    txt = out.outputs[0].text if out.outputs else ""
                    lab = (
                        parse_verdict_strict(txt)
                        if args.strict_verdict
                        else parse_verdict_with_fallback(txt)
                    )
                    rows_out[rid]["verification_details"].append(
                        {
                            "claim": claim,
                            "label": lab,
                            "score_not_supported": score_not_supported_from_label(lab),
                            "llm_raw": txt,
                            "top_evidence_claim_query": top3,
                            **({"llm_error": mismatch_err} if mismatch_err else {}),
                        }
                    )
                else:
                    rows_out[rid]["verification_details"].append(
                        {
                            "claim": claim,
                            "label": None,
                            "score_not_supported": 1.0,
                            "llm_raw": None,
                            "llm_error": mismatch_err or {"type": "missing_output"},
                            "top_evidence_claim_query": top3,
                        }
                    )

        batch_prompts, batch_meta = [], []

    data_path = Path(args.data_jsonl)
    _total = sum(1 for _ in data_path.open("r", encoding="utf-8"))

    with VLLMSession(
        args.model, gpu_memory_utilization=args.gpu_memory_utilization
    ) as llm:
        ex_seen = 0
        for ex_i, ex in enumerate(
            tqdm(read_jsonl(data_path), total=_total, desc="FactBench examples")
        ):
            if args.max_items and ex_i >= args.max_items:
                break
            ex_seen += 1

            prompt = ex.get("prompt", "") or ex.get("question", "") or ""
            sent_dict = ex.get("sentences", {}) or {}
            ordered_sents = iter_sentences_in_order(sent_dict)

            for sent_i, (sent_key, sent_obj) in enumerate(ordered_sents):
                sentence = clean_seg(
                    sent_obj.get("decontext") or sent_obj.get("text") or ""
                )
                if not sentence:
                    continue

                gold_supported = norm_gold_label(
                    sent_obj.get("sentence_factuality_label")
                )

                passages = build_candidate_passages_factbench(
                    prompt, sent_obj, args.max_passages, args.max_chars
                )
                bm25_index = build_bm25_index(passages) if passages else None
                sent_top_passages = select_topk_passages_bm25_indexed(
                    bm25_index, passages, query=f"{prompt}\n{sentence}", topk=args.topk
                )

                # Claims
                extraction_raw = None
                extraction_error = None
                row = {
                    "example_index": ex_i,
                    "sentence_key": sent_key,
                    "prompt": prompt,
                    "sentence": sentence,
                    "gold_supported": gold_supported,
                    "claims_source": args.claims_source,
                    "claims": [],
                    "total_claims": 0,
                    "extraction_raw": None,
                    "extraction_error": None,
                    "has_evidence_pool": bool(passages),
                    "top_evidence_sentence_query": [
                        p.get("snippet", "") for p in sent_top_passages[:3]
                    ],
                    "verification_details": [],
                }
                _init_eff_fields(row)
                row.setdefault("timing", {})
                row["timing"].setdefault("extract_s", 0.0)
                row["timing"].setdefault("verify_s", 0.0)
                row_idx = len(rows_out)
                rows_out.append(row)

                if args.claims_source == "sentence":
                    claims = [sentence]
                elif args.claims_source == "dataset":
                    raw_claims = sent_obj.get("claims", None)
                    claims = []
                    if isinstance(raw_claims, list):
                        claims = [
                            clean_seg(str(c)) for c in raw_claims if clean_seg(str(c))
                        ]
                    if not claims:
                        claims = [sentence]
                else:
                    # Claim extraction
                    snippet = build_extraction_snippet_with_window_factbench(
                        prompt, ordered_sents, sent_i, prev_n=3, next_n=1
                    )
                    user_extract = extraction_template.format(
                        snippet=snippet, sentence=sentence
                    )
                    extract_prompt = wrap_extract(user_extract)
                    t0 = time.perf_counter()
                    outs, err = vllm_generate_with_retries(
                        llm, [extract_prompt], sp_extract, max_tries=args.max_tries
                    )
                    dt = time.perf_counter() - t0
                    row["timing"]["extract_s"] += dt

                    if outs is not None and outs[0] is not None:
                        _add_vllm_tokens(row, outs[0], stage="extract")

                    if err is not None or outs is None:
                        extraction_error = err
                        claims = (
                            [sentence]
                            if args.fallback_to_sentence_on_extract_error
                            else []
                        )
                    else:
                        extraction_raw = (
                            outs[0].outputs[0].text if outs[0].outputs else ""
                        )
                        claims = dedup_claims(
                            parse_claims_from_extraction(extraction_raw, max_claims=32),
                            max_claims=args.max_claims,
                        )

                row["extraction_raw"] = extraction_raw
                row["extraction_error"] = extraction_error
                row["claims"] = claims
                row["total_claims"] = len(claims)

                sentence_claim_counts.append(len(claims))

                if not claims:
                    continue

                for c in claims:
                    claim_top = select_topk_passages_bm25_indexed(
                        bm25_index, passages, query=f"{prompt}\n{c}", topk=args.topk
                    )
                    if not claim_top:
                        rows_out[row_idx]["verification_details"].append(
                            {
                                "claim": c,
                                "label": "not_supported",
                                "score_not_supported": 1.0,
                                "llm_raw": None,
                                "note": "no_evidence_for_claim",
                                "top_evidence_claim_query": [],
                            }
                        )
                        continue

                    search_res_str = build_search_results_str(claim_top, k=args.topk)
                    tail = (
                        f"Your task:\n\nClaim: {c}\n\n{search_res_str}\n\n"
                        "Answer with exactly one of: ###Supported### or ###Unsupported###.\n\n"
                        "Your decision:"
                    )
                    user_verify = f"{prompt_initial_temp}\n\n{tail}"
                    full_prompt = wrap_verify(user_verify)

                    batch_prompts.append(full_prompt)
                    batch_meta.append(
                        (row_idx, c, [p.get("snippet", "") for p in claim_top[:3]])
                    )

                    if len(batch_prompts) >= args.batch_size:
                        flush_verify_batch(llm)

        flush_verify_batch(llm)

    # Finalize predictions + metrics
    for r in rows_out:
        _finalize_eff_row(r, args.model_params_b, args.flops_per_param)
        labels = [d.get("label") for d in (r.get("verification_details") or [])]
        pred_supported = strict_sentence_supported(
            labels, empty_claims_policy=args.empty_claims_policy
        )

        r["pred_supported"] = pred_supported
        r["pred_label"] = (
            "supported"
            if pred_supported is True
            else ("not_supported" if pred_supported is False else None)
        )

        r["supported_claims"] = sum(
            1
            for d in r["verification_details"]
            if (d.get("label") or "").lower() == "supported"
        )
        r["parsed_claims"] = sum(
            1
            for d in r["verification_details"]
            if d.get("label") in {"supported", "not_supported"}
        )
        r["risk_not_supported"] = sentence_risk_not_supported(r["verification_details"])
        r["score_not_supported"] = r["risk_not_supported"]

        gold_supported = r.get("gold_supported", None)
        if gold_supported is not None and pred_supported is not None:
            update_confusion_not_supported_positive(
                cm, bool(gold_supported), bool(pred_supported)
            )
            y_true.append(1 if (not bool(gold_supported)) else 0)
            y_score.append(float(r["score_not_supported"]))

    # VeriScore aggregation
    if args.veriscore_k_mode == "fixed":
        K = max(int(args.veriscore_k_fixed), 1)
    else:
        K = max(median_int(sentence_claim_counts), 1)

    veri_f1s: List[float] = []
    for r in rows_out:
        r["veriscore_f1_at_k"] = f1_at_k(
            int(r.get("supported_claims", 0)), int(r.get("total_claims", 0)), K
        )
        veri_f1s.append(r["veriscore_f1_at_k"])

    veriscore_mean = sum(veri_f1s) / len(veri_f1s) if veri_f1s else 0.0

    metrics = cm_to_macro_f1(cm)
    metrics["roc_auc_not_supported"] = (
        roc_auc_manual(y_true, y_score) if y_true else 0.0
    )
    metrics["n_scored_for_gold_metrics"] = len(y_true)
    metrics["veriscore"] = {
        "k_mode": args.veriscore_k_mode,
        "K": K,
        "n_sentences": len(rows_out),
        "mean_f1_at_k": veriscore_mean,
        "claim_count_median": (
            median_int(sentence_claim_counts) if sentence_claim_counts else 1
        ),
        "empty_claims_policy": args.empty_claims_policy,
    }
    eff_rows = [r for r in rows_out if (r.get("sentence") or "").strip()]
    eff = _agg_efficiency(eff_rows)

    compute = None
    if args.model_params_b and args.model_params_b > 0:
        compute = {
            "params_b": args.model_params_b,
            "flops_per_param": args.flops_per_param,
            **_agg_compute(eff_rows),
        }

    metrics["efficiency"] = eff
    metrics["compute"] = compute

    write_jsonl(rows_out, out_dir / "predictions.jsonl")
    write_json(metrics, out_dir / "metrics.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


# FELM runner
def run_felm(args) -> None:
    if load_from_disk is None:
        raise RuntimeError(
            "datasets is required for FELM. Install datasets or run factbench mode."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolve_veriscore_assets(args)
    ensure_assets_exist(
        args.extraction_template,
        args.verification_instruction_binary,
        args.fewshot_jsonl,
    )

    extraction_template = read_text(Path(args.extraction_template))
    verif_template = read_text(Path(args.verification_instruction_binary))
    fewshot_rows = load_fewshot_jsonl(Path(args.fewshot_jsonl))
    prompt_initial_temp = fill_verification_fewshot_template(
        verif_template, fewshot_rows
    )

    # Prompt wrapper selection
    alpaca_template_txt = ""
    tokenizer = None
    use_alpaca = bool(args.alpaca_template)
    if use_alpaca:
        alpaca_template_txt = read_text(Path(args.alpaca_template))
    else:
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required for tokenizer-based chat prompts. Install transformers or pass --alpaca_template."
            )
        tok_name = args.tokenizer_name or args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tok_name, trust_remote_code=args.trust_remote_code
        )

    wrap_extract, wrap_verify = build_prompt_wrappers(
        use_alpaca_template=use_alpaca,
        alpaca_template_txt=alpaca_template_txt,
        tokenizer=tokenizer,
        system_extract=args.system_extract,
        system_verify=args.system_verify,
    )

    # Stops
    stop_common = [s.strip() for s in args.stop_common.split(",") if s.strip()]
    stop_extract_extra = [
        s.encode("utf-8").decode("unicode_escape")
        for s in args.stop_extract_extra.split(",")
        if s.strip()
    ]
    stop_extract = stop_common + stop_extract_extra
    stop_verify = stop_common

    sp_extract, sp_verify = make_sampling_params(
        temperature=args.temperature,
        max_tokens_extract=args.max_tokens_extract,
        max_tokens_verify=args.max_tokens_verify,
        stop_extract=stop_extract,
        stop_verify=stop_verify,
    )

    # Load FELM
    ds_path = Path(args.felm_dir) / args.subset
    ds = load_from_disk(str(ds_path))[args.split]
    if args.max_examples and args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    # Fail accounting
    fail = {
        "empty_sentence": 0,
        "no_context": 0,
        "exception_extraction": 0,
        "no_verifiable_claim": 0,
        "exception_verification": 0,
        "unparsed_verification": 0,
        "no_evidence_for_claim": 0,
    }

    total_sentences_seen = 0
    skipped_no_context = 0

    cm_all = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    cm_eval = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    correct_all = 0
    correct_eval = 0
    n_all = 0
    n_eval = 0

    y_true_eval: List[int] = []
    y_score_eval: List[float] = []

    sentence_claim_counts: List[int] = []
    segments_out: List[Dict[str, Any]] = []

    # Verification batching across run
    batch_prompts: List[str] = []
    batch_meta: List[Tuple[int, str, List[str]]] = (
        []
    )  # (seg_row_idx, claim, top3_snips)

    def flush_verify_batch(llm):
        nonlocal batch_prompts, batch_meta
        if not batch_prompts:
            return

        t0 = time.perf_counter()
        outs, err = vllm_generate_with_retries(
            llm, batch_prompts, sp_verify, max_tries=args.max_tries
        )
        dt = time.perf_counter() - t0
        n = len(batch_prompts)
        per_item = dt / n if n else 0.0

        # If outs is None or error, mark all as failed
        if err is not None or outs is None:
            fail["exception_verification"] += len(batch_meta)
            for rid, claim, top3 in batch_meta:
                segments_out[rid]["timing"]["verify_s"] += per_item
                segments_out[rid]["verification_details"].append(
                    {
                        "claim": claim,
                        "label": None,
                        "score_not_supported": 1.0,
                        "llm_raw": None,
                        "llm_error": err,
                        "top_evidence_claim_query": top3,
                    }
                )
            batch_prompts, batch_meta = [], []
            return
        # Robust alignment (handle unexpected length mismatch)
        if len(outs) != len(batch_meta):
            mismatch_err = {
                "type": "length_mismatch",
                "message": f"vLLM returned {len(outs)} outs for {len(batch_meta)} prompts",
            }
        else:
            mismatch_err = None

        # Count tokens ONCE for the entire batch
        if outs and len(outs) > 0:
            # The first output has the shared prompt tokens (same for all in batch)
            shared_prompt_tokens = len(getattr(outs[0], "prompt_token_ids", []) or [])

            # Distribute prompt tokens evenly across claims
            n_items = len(batch_meta)
            per_claim_prompt = shared_prompt_tokens // n_items

            for j in range(len(batch_meta)):
                rid, claim, top3 = batch_meta[j]
                segments_out[rid]["timing"]["verify_s"] += per_item
                # Add prompt tokens (distributed from shared batch)
                segments_out[rid]["tokens"]["verify_prompt"] += per_claim_prompt

                out = outs[j] if j < len(outs) else None
                if out is not None:
                    # Count generation tokens for THIS specific claim
                    gen_tok = 0
                    for o in getattr(out, "outputs", None) or []:
                        gen_tok += len(getattr(o, "token_ids", []) or [])
                    segments_out[rid]["tokens"]["verify_gen"] += gen_tok

                    txt = out.outputs[0].text if out.outputs else ""
                    lab = (
                        parse_verdict_strict(txt)
                        if args.strict_verdict
                        else parse_verdict_with_fallback(txt)
                    )
                    segments_out[rid]["verification_details"].append(
                        {
                            "claim": claim,
                            "label": lab,
                            "score_not_supported": score_not_supported_from_label(lab),
                            "llm_raw": txt,
                            "top_evidence_claim_query": top3,
                            **({"llm_error": mismatch_err} if mismatch_err else {}),
                        }
                    )
                else:
                    segments_out[rid]["verification_details"].append(
                        {
                            "claim": claim,
                            "label": None,
                            "score_not_supported": 1.0,
                            "llm_raw": None,
                            "llm_error": mismatch_err or {"type": "missing_output"},
                            "top_evidence_claim_query": top3,
                        }
                    )

        batch_prompts, batch_meta = [], []

    with VLLMSession(
        args.model, gpu_memory_utilization=args.gpu_memory_utilization
    ) as llm:
        for ex in tqdm(ds, total=len(ds), desc="FELM examples"):
            question = clean_seg(ex.get("prompt") or "")
            segs = [clean_seg(s) for s in (ex.get("segmented_response") or [])]
            golds = [bool(x) for x in (ex.get("labels") or [])]
            ex_index = ex.get("index", None)

            # Evidence pool (example-level)
            base_topic = topic_from_ref_or_prompt(question, ex.get("ref") or [])
            passages = ref_text_to_passages(
                base_topic, ex.get("ref_text") or "", args.max_passages, args.max_chars
            )
            has_context = bool(passages)
            bm25_index = build_bm25_index(passages) if passages else None

            # Batch extraction per example
            do_extract = args.claims_source == "llm"
            extract_prompts: List[str] = []
            if do_extract:
                for i, s in enumerate(segs):
                    snippet = build_extraction_snippet_with_window_felm(
                        question, segs, i, prev_n=3, next_n=1
                    )
                    user_extract = extraction_template.format(
                        snippet=snippet, sentence=s
                    )
                    extract_prompts.append(wrap_extract(user_extract))

            per_sent_extract = 0.0
            extract_outs, extract_err = (None, None)
            if extract_prompts:
                t0 = time.perf_counter()
                extract_outs, extract_err = vllm_generate_with_retries(
                    llm, extract_prompts, sp_extract, max_tries=args.max_tries
                )
                dt_extract = time.perf_counter() - t0
                per_sent_extract = (
                    (dt_extract / len(extract_prompts)) if extract_prompts else 0.0
                )

            for i, (sentence, gold_supported) in enumerate(zip(segs, golds)):
                total_sentences_seen += 1
                gold_label = "supported" if gold_supported else "not_supported"

                if (not has_context) and args.skip_no_context:
                    skipped_no_context += 1
                    segments_out.append(
                        {
                            "subset": args.subset,
                            "split": args.split,
                            "example_index": ex_index,
                            "seg_id": i,
                            "prompt": question,
                            "sentence": sentence,
                            "gold_supported": gold_supported,
                            "gold_label": gold_label,
                            "skipped": True,
                            "skip_reason": "no_context",
                        }
                    )
                    sentence_claim_counts.append(0)
                    continue

                row = {
                    "subset": args.subset,
                    "split": args.split,
                    "example_index": ex_index,
                    "seg_id": i,
                    "prompt": question,
                    "sentence": sentence,
                    "gold_supported": gold_supported,
                    "gold_label": gold_label,
                    "has_context": has_context,
                    "n_passages_total": len(passages),
                    "claims_source": args.claims_source,
                    "claims": [],
                    "total_claims": 0,
                    "extraction_raw": None,
                    "extraction_error": None,
                    "verification_details": [],
                    "fail_reason": "",
                    "per_sent_extract": per_sent_extract,
                }
                _init_eff_fields(row)
                row.setdefault("timing", {})
                row["timing"].setdefault("extract_s", 0.0)
                row["timing"]["extract_s"] += float(per_sent_extract)
                rid = len(segments_out)
                segments_out.append(row)

                if (
                    extract_outs is not None
                    and i < len(extract_outs)
                    and extract_outs[i] is not None
                ):
                    _add_vllm_tokens(row, extract_outs[i], stage="extract")

                if not sentence:
                    fail["empty_sentence"] += 1
                    row["fail_reason"] = "empty_sentence"
                    sentence_claim_counts.append(0)
                    continue

                if not has_context:
                    fail["no_context"] += 1
                    row["fail_reason"] = "no_context"
                    sentence_claim_counts.append(0)
                    continue

                if do_extract and (extract_err is not None or extract_outs is None):
                    fail["exception_extraction"] += 1
                    row["fail_reason"] = "exception_extraction"
                    row["extraction_error"] = extract_err
                    sentence_claim_counts.append(0)
                    continue
                if do_extract:
                    try:
                        row["extraction_raw"] = (
                            extract_outs[i].outputs[0].text
                            if extract_outs[i].outputs
                            else ""
                        )
                    except Exception:
                        row["extraction_raw"] = ""
                else:
                    row["extraction_raw"] = None

                # Claims
                if args.claims_source == "sentence":
                    claims = [sentence]
                else:
                    claims = dedup_claims(
                        parse_claims_from_extraction(
                            row["extraction_raw"], max_claims=32
                        ),
                        max_claims=args.max_claims,
                    )

                row["claims"] = claims
                row["total_claims"] = len(claims)
                sentence_claim_counts.append(len(claims))

                if not claims:
                    fail["no_verifiable_claim"] += 1
                    row["no_verifiable_claim"] = True
                    continue

                # Verify each claim (batched across run)
                for c in claims:
                    claim_top = select_topk_passages_bm25_indexed(
                        bm25_index, passages, query=f"{question}\n{c}", topk=args.topk
                    )
                    if not claim_top:
                        fail["no_evidence_for_claim"] += 1
                        row["verification_details"].append(
                            {
                                "claim": c,
                                "label": "not_supported",
                                "score_not_supported": 1.0,
                                "llm_raw": None,
                                "note": "no_evidence_for_claim",
                                "top_evidence_claim_query": [],
                            }
                        )
                        continue

                    search_res_str = build_search_results_str(claim_top, k=args.topk)
                    tail = (
                        f"Your task:\n\nClaim: {c}\n\n{search_res_str}\n\n"
                        "Answer with exactly one of: ###Supported### or ###Unsupported###.\n\n"
                        "Your decision:"
                    )
                    user_verify = f"{prompt_initial_temp}\n\n{tail}"
                    full_prompt = wrap_verify(user_verify)

                    batch_prompts.append(full_prompt)
                    batch_meta.append(
                        (rid, c, [p.get("snippet", "") for p in claim_top[:3]])
                    )

                    if len(batch_prompts) >= args.batch_size:
                        flush_verify_batch(llm)

        flush_verify_batch(llm)

    # Pick K
    if args.veriscore_k_mode == "fixed":
        K = max(int(args.veriscore_k_fixed), 1)
    else:
        K = max(median_int(sentence_claim_counts), 1)

    veriscore_f1s: List[float] = []

    for r in segments_out:
        if not r.get("skipped"):
            _finalize_eff_row(r, args.model_params_b, args.flops_per_param)
        if r.get("skipped"):
            r["pred_supported"] = None
            r["pred_label"] = None
            r["supported_claims"] = 0
            r["parsed_claims"] = 0
            r["parsed_all_claims"] = False
            r["risk_not_supported"] = 0.0
            r["score_not_supported"] = 0.0
            r["evaluable"] = False
            r["veriscore_f1_at_k"] = 0.0
            continue

        details = r.get("verification_details") or []
        labels = [d.get("label") for d in details]

        pred_supported = strict_sentence_supported(
            labels, empty_claims_policy=args.empty_claims_policy
        )
        r["pred_supported"] = pred_supported
        r["pred_label"] = (
            "supported"
            if pred_supported is True
            else ("not_supported" if pred_supported is False else None)
        )

        r["supported_claims"] = sum(
            1 for d in details if (d.get("label") or "").lower() == "supported"
        )
        r["parsed_claims"] = sum(
            1 for d in details if d.get("label") in {"supported", "not_supported"}
        )
        r["parsed_all_claims"] = (
            bool(r.get("claims"))
            and bool(details)
            and all(lab in {"supported", "not_supported"} for lab in labels)
        )

        r["risk_not_supported"] = sentence_risk_not_supported(details)
        r["score_not_supported"] = r["risk_not_supported"]

        r["veriscore_f1_at_k"] = f1_at_k(
            int(r.get("supported_claims", 0)), int(r.get("total_claims", 0)), K
        )
        veriscore_f1s.append(r["veriscore_f1_at_k"])

        gold_supported = r.get("gold_supported", None)
        if gold_supported is None:
            r["evaluable"] = False
            continue

        evaluable = (
            (not r.get("fail_reason"))
            and bool(r.get("has_context"))
            and bool(r.get("claims"))
            and bool(r.get("parsed_all_claims"))
            and (pred_supported is not None)
        )
        r["evaluable"] = bool(evaluable)

        # ALL metrics (forced-wrong on failures/unparsed)
        n_all += 1
        has_failure = bool(r.get("fail_reason"))
        if (
            (not has_failure)
            and r.get("claims")
            and details
            and (not r.get("parsed_all_claims"))
        ):
            fail["unparsed_verification"] += 1
            r["fail_reason"] = "unparsed_verification"
            has_failure = True

        if has_failure or (pred_supported is None) or (not r.get("parsed_all_claims")):
            forced_pred_supported = not bool(gold_supported)  # forced wrong
            update_confusion_not_supported_positive(
                cm_all, bool(gold_supported), bool(forced_pred_supported)
            )
        else:
            update_confusion_not_supported_positive(
                cm_all, bool(gold_supported), bool(pred_supported)
            )
            correct_all += int(bool(pred_supported) == bool(gold_supported))

        # EVALUABLE only
        if evaluable:
            n_eval += 1
            update_confusion_not_supported_positive(
                cm_eval, bool(gold_supported), bool(pred_supported)
            )
            correct_eval += int(bool(pred_supported) == bool(gold_supported))
            y_true_eval.append(1 if (not bool(gold_supported)) else 0)
            y_score_eval.append(float(r["score_not_supported"]))

    m_all = cm_to_macro_f1(cm_all)
    m_eval = cm_to_macro_f1(cm_eval)

    eff_rows = [
        r
        for r in segments_out
        if (not r.get("skipped")) and (r.get("sentence") or "").strip()
    ]

    metrics = {
        "dataset": "FELM (offline ref_text)",
        "subset": args.subset,
        "split": args.split,
        "model_used": args.model,
        "tokenizer_used": (args.tokenizer_name or args.model),
        "claims_source": args.claims_source,
        "skip_no_context": bool(args.skip_no_context),
        "empty_claims_policy": args.empty_claims_policy,
        "total_sentences_seen": total_sentences_seen,
        "total_sentences_scored_all": n_all,
        "total_sentences_scored_evaluable": n_eval,
        "skipped_no_context": skipped_no_context,
        "fail_breakdown": fail,
        # ALL
        "acc_all": (correct_all / n_all) if n_all else 0.0,
        "f1_macro_all": m_all["f1_macro"],
        "confusion_matrix_all": m_all["confusion_matrix"],
        "precision_not_supported_all": m_all["precision_not_supported"],
        "recall_not_supported_all": m_all["recall_not_supported"],
        "f1_not_supported_all": m_all["f1_not_supported"],
        "precision_supported_all": m_all["precision_supported"],
        "recall_supported_all": m_all["recall_supported"],
        "f1_supported_all": m_all["f1_supported"],
        # EVALUABLE
        "acc_evaluable": (correct_eval / n_eval) if n_eval else 0.0,
        "f1_macro_evaluable": m_eval["f1_macro"],
        "confusion_matrix_evaluable": m_eval["confusion_matrix"],
        "roc_auc_not_supported_evaluable": (
            roc_auc_manual(y_true_eval, y_score_eval) if y_true_eval else 0.0
        ),
        # VeriScore
        "veriscore": {
            "k_mode": args.veriscore_k_mode,
            "K": K,
            "n_sentences_output": len(segments_out),
            "mean_f1_at_k": (
                (sum(veriscore_f1s) / len(veriscore_f1s)) if veriscore_f1s else 0.0
            ),
            "claim_count_median": (
                median_int(sentence_claim_counts) if sentence_claim_counts else 1
            ),
        },
    }

    metrics["efficiency"] = _agg_efficiency(eff_rows)
    metrics["compute"] = (
        None
        if args.model_params_b <= 0
        else {
            "params_b": args.model_params_b,
            "flops_per_param": args.flops_per_param,
            **_agg_compute(eff_rows),
        }
    )

    write_jsonl(segments_out, out_dir / "segments_with_veriscore.jsonl")
    write_json(metrics, out_dir / "metrics.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# ANAH runner
# ---------------------------------------------------------------------------
def _anah_gold_supported(hallucination_type: Any) -> Optional[bool]:
    """Map ANAH hallucination_type to gold_supported bool.

    'No Hallucination' -> True (supported)
    'No Fact'          -> None (skip – no verifiable fact)
    anything else      -> False (not supported)
    """
    t = str(hallucination_type or "").strip()
    if t == "No Hallucination":
        return True
    if t == "No Fact":
        return None
    return False  # Contradictory / Unverifiable Hallucination


def run_anah(args) -> None:
    """VeriScore-style pipeline on the ANAH (opencompass/anah) dataset.

    ANAH is a *sentence-level* dataset: each example is already one annotated
    sentence with its own reference fragment and hallucination label.
    We treat the ann_reference fragment as the evidence pool per sentence.
    """
    import sys as _sys, os as _os  # noqa: PLC0415
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from anah_utils import iter_anah_sentences  # noqa: PLC0415

    from datasets import load_dataset  # noqa: PLC0415

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolve_veriscore_assets(args)
    ensure_assets_exist(
        args.extraction_template,
        args.verification_instruction_binary,
        args.fewshot_jsonl,
    )

    extraction_template = read_text(Path(args.extraction_template))
    verif_template = read_text(Path(args.verification_instruction_binary))
    fewshot_rows = load_fewshot_jsonl(Path(args.fewshot_jsonl))
    prompt_initial_temp = fill_verification_fewshot_template(verif_template, fewshot_rows)

    # Prompt wrappers
    alpaca_template_txt = ""
    tokenizer = None
    use_alpaca = bool(args.alpaca_template)
    if use_alpaca:
        alpaca_template_txt = read_text(Path(args.alpaca_template))
    else:
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required for tokenizer-based chat prompts. "
                "Install transformers or pass --alpaca_template."
            )
        tok_name = args.tokenizer_name or args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tok_name, trust_remote_code=args.trust_remote_code
        )

    wrap_extract, wrap_verify = build_prompt_wrappers(
        use_alpaca_template=use_alpaca,
        alpaca_template_txt=alpaca_template_txt,
        tokenizer=tokenizer,
        system_extract=args.system_extract,
        system_verify=args.system_verify,
    )

    # Stops / sampling params
    stop_common = [s.strip() for s in args.stop_common.split(",") if s.strip()]
    stop_extract_extra = [
        s.encode("utf-8").decode("unicode_escape")
        for s in args.stop_extract_extra.split(",")
        if s.strip()
    ]
    stop_extract = stop_common + stop_extract_extra
    stop_verify = stop_common
    sp_extract, sp_verify = make_sampling_params(
        temperature=args.temperature,
        max_tokens_extract=args.max_tokens_extract,
        max_tokens_verify=args.max_tokens_verify,
        stop_extract=stop_extract,
        stop_verify=stop_verify,
    )

    # Load ANAH dataset
    if args.anah_sample_file:
        import json as _json
        with open(args.anah_sample_file, encoding="utf-8") as _f:
            _sample_rows = [_json.loads(l) for l in _f if l.strip()]
        _anah_iter = iter(_sample_rows)
        _anah_total = len(_sample_rows)
        print(f"ANAH: loaded {_anah_total} rows from sample file '{args.anah_sample_file}'", flush=True)
    else:
        ds = load_dataset("opencompass/anah", split=args.anah_split)
        _anah_total = (
            min(args.max_examples, len(ds)) if args.max_examples and args.max_examples > 0 else len(ds)
        )
        _anah_iter = iter_anah_sentences(ds, max_examples=args.max_examples or 0)

    # Fail accounting
    fail = {
        "empty_sentence": 0,
        "no_context": 0,
        "no_fact_skipped": 0,
        "exception_extraction": 0,
        "no_verifiable_claim": 0,
        "exception_verification": 0,
        "unparsed_verification": 0,
        "no_evidence_for_claim": 0,
    }

    total_sentences_seen = 0
    skipped_no_context = 0
    skipped_no_fact = 0

    cm_all = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    cm_eval = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    correct_all = 0
    correct_eval = 0
    n_all = 0
    n_eval = 0

    y_true_eval: List[int] = []
    y_score_eval: List[float] = []

    sentence_claim_counts: List[int] = []
    segments_out: List[Dict[str, Any]] = []

    # Verification batching
    batch_prompts: List[str] = []
    batch_meta: List[Tuple[int, str, List[str]]] = []

    def flush_verify_batch(llm):
        nonlocal batch_prompts, batch_meta
        if not batch_prompts:
            return

        t0 = time.perf_counter()
        outs, err = vllm_generate_with_retries(
            llm, batch_prompts, sp_verify, max_tries=args.max_tries
        )
        dt = time.perf_counter() - t0
        n = len(batch_prompts)
        per_item = dt / n if n else 0.0

        if err is not None or outs is None:
            fail["exception_verification"] += len(batch_meta)
            for rid, claim, top3 in batch_meta:
                segments_out[rid]["timing"]["verify_s"] += per_item
                segments_out[rid]["verification_details"].append(
                    {
                        "claim": claim,
                        "label": None,
                        "score_not_supported": 1.0,
                        "llm_raw": None,
                        "llm_error": err,
                        "top_evidence_claim_query": top3,
                    }
                )
            batch_prompts, batch_meta = [], []
            return

        if len(outs) != len(batch_meta):
            mismatch_err = {
                "type": "length_mismatch",
                "message": f"vLLM returned {len(outs)} outs for {len(batch_meta)} prompts",
            }
        else:
            mismatch_err = None

        if outs and len(outs) > 0:
            shared_prompt_tokens = len(getattr(outs[0], "prompt_token_ids", []) or [])
            n_items = len(batch_meta)
            per_claim_prompt = shared_prompt_tokens // n_items

            for j in range(len(batch_meta)):
                rid, claim, top3 = batch_meta[j]
                segments_out[rid]["timing"]["verify_s"] += per_item
                segments_out[rid]["tokens"]["verify_prompt"] += per_claim_prompt

                out = outs[j] if j < len(outs) else None
                if out is not None:
                    gen_tok = 0
                    for o in getattr(out, "outputs", None) or []:
                        gen_tok += len(getattr(o, "token_ids", []) or [])
                    segments_out[rid]["tokens"]["verify_gen"] += gen_tok

                    txt = out.outputs[0].text if out.outputs else ""
                    lab = (
                        parse_verdict_strict(txt)
                        if args.strict_verdict
                        else parse_verdict_with_fallback(txt)
                    )
                    segments_out[rid]["verification_details"].append(
                        {
                            "claim": claim,
                            "label": lab,
                            "score_not_supported": score_not_supported_from_label(lab),
                            "llm_raw": txt,
                            "top_evidence_claim_query": top3,
                            **({"llm_error": mismatch_err} if mismatch_err else {}),
                        }
                    )
                else:
                    segments_out[rid]["verification_details"].append(
                        {
                            "claim": claim,
                            "label": None,
                            "score_not_supported": 1.0,
                            "llm_raw": None,
                            "llm_error": mismatch_err or {"type": "missing_output"},
                            "top_evidence_claim_query": top3,
                        }
                    )

        batch_prompts, batch_meta = [], []

    with VLLMSession(args.model, gpu_memory_utilization=args.gpu_memory_utilization) as llm:
        for sent_row in tqdm(
            _anah_iter,
            total=_anah_total,
            desc="ANAH sentences",
        ):
            total_sentences_seen += 1

            gold_supported = sent_row["gold_supported"]

            # Skip "No Fact" rows (no verifiable claim)
            if gold_supported is None:
                skipped_no_fact += 1
                fail["no_fact_skipped"] += 1
                sentence_claim_counts.append(0)
                continue

            question = clean_seg(sent_row["question"])
            sentence = clean_seg(sent_row["sentence"])
            hallucination_type = sent_row["hallucination_type"]
            ex_i = sent_row["example_index"]
            # Use ann_reference (specific cited fragment) as per-sentence evidence
            reference = sent_row["ann_reference"]

            # Build passage pool from the reference fragment
            passages = ref_text_to_passages(
                topic=question[:80] or f"anah_{ex_i}_{sent_row['sentence_index']}",
                ref_text=reference,
                max_passages=args.max_passages,
                max_chars=args.max_chars,
            )
            has_context = bool(passages)
            bm25_index = build_bm25_index(passages) if passages else None

            gold_label = "supported" if gold_supported else "not_supported"

            row: Dict[str, Any] = {
                "example_index": ex_i,
                "answer_index": sent_row["answer_index"],
                "sentence_index": sent_row["sentence_index"],
                "question": question,
                "sentence": sentence,
                "hallucination_type": hallucination_type,
                "gold_supported": gold_supported,
                "gold_label": gold_label,
                "has_context": has_context,
                "claims_source": args.claims_source,
                "claims": [],
                "total_claims": 0,
                "extraction_raw": None,
                "extraction_error": None,
                "verification_details": [],
                "fail_reason": "",
            }
            _init_eff_fields(row)
            row.setdefault("timing", {})
            row["timing"].setdefault("extract_s", 0.0)
            row["timing"].setdefault("verify_s", 0.0)
            rid = len(segments_out)
            segments_out.append(row)

            if not sentence:
                fail["empty_sentence"] += 1
                row["fail_reason"] = "empty_sentence"
                sentence_claim_counts.append(0)
                continue

            if not has_context and args.skip_no_context:
                skipped_no_context += 1
                fail["no_context"] += 1
                row["fail_reason"] = "no_context"
                sentence_claim_counts.append(0)
                continue

            # Claim extraction / selection
            extraction_raw = None
            extraction_error = None

            if args.claims_source == "sentence":
                claims = [sentence]
            else:
                # Use ordered_sents = [(key, sentence)] to reuse FELM helpers
                ordered_sents = [("sentence0", {"text": sentence})]
                snippet = build_extraction_snippet_with_window_felm(
                    question, [sentence], 0, prev_n=0, next_n=0
                )
                user_extract = extraction_template.format(snippet=snippet, sentence=sentence)
                extract_prompt = wrap_extract(user_extract)
                t0 = time.perf_counter()
                outs, err = vllm_generate_with_retries(
                    llm, [extract_prompt], sp_extract, max_tries=args.max_tries
                )
                dt = time.perf_counter() - t0
                row["timing"]["extract_s"] += dt

                if outs is not None and outs[0] is not None:
                    _add_vllm_tokens(row, outs[0], stage="extract")

                if err is not None or outs is None:
                    extraction_error = err
                    claims = [sentence]
                else:
                    extraction_raw = outs[0].outputs[0].text if outs[0].outputs else ""
                    claims = dedup_claims(
                        parse_claims_from_extraction(extraction_raw, max_claims=32),
                        max_claims=args.max_claims,
                    )

            row["extraction_raw"] = extraction_raw
            row["extraction_error"] = extraction_error
            row["claims"] = claims
            row["total_claims"] = len(claims)
            sentence_claim_counts.append(len(claims))

            if not claims:
                fail["no_verifiable_claim"] += 1
                row["no_verifiable_claim"] = True
                continue

            # Verification per claim
            for c in claims:
                if has_context:
                    claim_top = select_topk_passages_bm25_indexed(
                        bm25_index, passages, query=f"{question}\n{c}", topk=args.topk
                    )
                else:
                    claim_top = []

                if not claim_top:
                    fail["no_evidence_for_claim"] += 1
                    row["verification_details"].append(
                        {
                            "claim": c,
                            "label": "not_supported",
                            "score_not_supported": 1.0,
                            "llm_raw": None,
                            "note": "no_evidence_for_claim",
                            "top_evidence_claim_query": [],
                        }
                    )
                    continue

                search_res_str = build_search_results_str(claim_top, k=args.topk)
                tail = (
                    f"Your task:\n\nClaim: {c}\n\n{search_res_str}\n\n"
                    "Answer with exactly one of: ###Supported### or ###Unsupported###.\n\n"
                    "Your decision:"
                )
                user_verify = f"{prompt_initial_temp}\n\n{tail}"
                full_prompt = wrap_verify(user_verify)

                batch_prompts.append(full_prompt)
                batch_meta.append((rid, c, [p.get("snippet", "") for p in claim_top[:3]]))

                if len(batch_prompts) >= args.batch_size:
                    flush_verify_batch(llm)

        flush_verify_batch(llm)

    # Finalize predictions + metrics
    if args.veriscore_k_mode == "fixed":
        K = max(int(args.veriscore_k_fixed), 1)
    else:
        K = max(median_int(sentence_claim_counts), 1)

    veri_f1s: List[float] = []

    for r in segments_out:
        if r.get("skipped"):
            r["pred_supported"] = None
            r["pred_label"] = None
            r["supported_claims"] = 0
            r["parsed_claims"] = 0
            r["parsed_all_claims"] = False
            r["risk_not_supported"] = 0.0
            r["score_not_supported"] = 0.0
            r["evaluable"] = False
            r["veriscore_f1_at_k"] = 0.0
            continue

        _finalize_eff_row(r, args.model_params_b, args.flops_per_param)
        details = r.get("verification_details") or []
        labels = [d.get("label") for d in details]

        pred_supported = strict_sentence_supported(
            labels, empty_claims_policy=args.empty_claims_policy
        )
        r["pred_supported"] = pred_supported
        r["pred_label"] = (
            "supported"
            if pred_supported is True
            else ("not_supported" if pred_supported is False else None)
        )
        r["supported_claims"] = sum(
            1 for d in details if (d.get("label") or "").lower() == "supported"
        )
        r["parsed_claims"] = sum(
            1 for d in details if d.get("label") in {"supported", "not_supported"}
        )
        r["parsed_all_claims"] = (
            bool(r.get("claims"))
            and bool(details)
            and all(lab in {"supported", "not_supported"} for lab in labels)
        )
        r["risk_not_supported"] = sentence_risk_not_supported(r["verification_details"])
        r["score_not_supported"] = r["risk_not_supported"]

        r["veriscore_f1_at_k"] = f1_at_k(
            int(r.get("supported_claims", 0)), int(r.get("total_claims", 0)), K
        )
        veri_f1s.append(r["veriscore_f1_at_k"])

        gold_supported = r.get("gold_supported")
        if gold_supported is None:
            r["evaluable"] = False
            continue

        evaluable = (
            (not r.get("fail_reason"))
            and bool(r.get("has_context"))
            and bool(r.get("claims"))
            and bool(r.get("parsed_all_claims"))
            and (pred_supported is not None)
        )
        r["evaluable"] = bool(evaluable)

        n_all += 1
        has_failure = bool(r.get("fail_reason"))
        if (
            (not has_failure)
            and r.get("claims")
            and details
            and (not r.get("parsed_all_claims"))
        ):
            fail["unparsed_verification"] += 1
            r["fail_reason"] = "unparsed_verification"
            has_failure = True

        if has_failure or (pred_supported is None) or (not r.get("parsed_all_claims")):
            forced_pred_supported = not bool(gold_supported)
            update_confusion_not_supported_positive(
                cm_all, bool(gold_supported), bool(forced_pred_supported)
            )
        else:
            update_confusion_not_supported_positive(
                cm_all, bool(gold_supported), bool(pred_supported)
            )
            correct_all += int(bool(pred_supported) == bool(gold_supported))

        if evaluable:
            n_eval += 1
            update_confusion_not_supported_positive(
                cm_eval, bool(gold_supported), bool(pred_supported)
            )
            correct_eval += int(bool(pred_supported) == bool(gold_supported))
            y_true_eval.append(1 if (not bool(gold_supported)) else 0)
            y_score_eval.append(float(r["score_not_supported"]))

    m_all = cm_to_macro_f1(cm_all)
    m_eval = cm_to_macro_f1(cm_eval)

    eff_rows = [
        r for r in segments_out
        if (not r.get("skipped")) and (r.get("sentence") or "").strip()
    ]

    metrics = {
        "dataset": "ANAH (opencompass/anah)",
        "split": args.anah_split,
        "model_used": args.model,
        "claims_source": args.claims_source,
        "skip_no_context": bool(args.skip_no_context),
        "empty_claims_policy": args.empty_claims_policy,
        "total_sentences_seen": total_sentences_seen,
        "skipped_no_fact": skipped_no_fact,
        "skipped_no_context": skipped_no_context,
        "total_sentences_scored_all": n_all,
        "total_sentences_scored_evaluable": n_eval,
        "fail_breakdown": fail,
        # ALL
        "acc_all": (correct_all / n_all) if n_all else 0.0,
        "f1_macro_all": m_all["f1_macro"],
        "confusion_matrix_all": m_all["confusion_matrix"],
        "precision_not_supported_all": m_all["precision_not_supported"],
        "recall_not_supported_all": m_all["recall_not_supported"],
        "f1_not_supported_all": m_all["f1_not_supported"],
        "precision_supported_all": m_all["precision_supported"],
        "recall_supported_all": m_all["recall_supported"],
        "f1_supported_all": m_all["f1_supported"],
        # EVALUABLE
        "acc_evaluable": (correct_eval / n_eval) if n_eval else 0.0,
        "f1_macro_evaluable": m_eval["f1_macro"],
        "confusion_matrix_evaluable": m_eval["confusion_matrix"],
        "roc_auc_not_supported_evaluable": (
            roc_auc_manual(y_true_eval, y_score_eval) if y_true_eval else 0.0
        ),
        # VeriScore
        "veriscore": {
            "k_mode": args.veriscore_k_mode,
            "K": K,
            "n_sentences_output": len(segments_out),
            "mean_f1_at_k": (
                (sum(veri_f1s) / len(veri_f1s)) if veri_f1s else 0.0
            ),
            "claim_count_median": (
                median_int(sentence_claim_counts) if sentence_claim_counts else 1
            ),
        },
        "efficiency": _agg_efficiency(eff_rows),
        "compute": (
            None
            if args.model_params_b <= 0
            else {
                "params_b": args.model_params_b,
                "flops_per_param": args.flops_per_param,
                **_agg_compute(eff_rows),
            }
        ),
    }

    write_jsonl(segments_out, out_dir / "segments_with_veriscore.jsonl")
    write_json(metrics, out_dir / "metrics.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_dir.resolve()}")


# CLI
def main():
    ap = argparse.ArgumentParser("veriscore_run.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common_flags(p):
        p.add_argument("--model", type=str, required=True)
        p.add_argument("--out_dir", type=str, required=True)

        p.add_argument(
            "--claims_source",
            type=str,
            default="llm",
            choices=["dataset", "llm", "sentence"],
        )
        p.add_argument("--max_claims", type=int, default=4)

        p.add_argument("--max_passages", type=int, default=80)
        p.add_argument("--max_chars", type=int, default=1200)
        p.add_argument("--topk", type=int, default=8)

        # Assets
        p.add_argument("--veriscore_assets_dir", type=str, default="")
        p.add_argument("--extraction_template", type=str, default="")
        p.add_argument("--verification_instruction_binary", type=str, default="")
        p.add_argument("--fewshot_jsonl", type=str, default="")

        # Prompt wrapping: either alpaca 2-slot template OR tokenizer chat template
        p.add_argument(
            "--alpaca_template",
            type=str,
            default="",
            help="If set, use 2-slot format(system,user).",
        )
        p.add_argument(
            "--tokenizer_name",
            type=str,
            default="",
            help="Optional tokenizer name (when not using alpaca_template).",
        )
        p.add_argument("--trust_remote_code", action="store_true")

        p.add_argument(
            "--system_extract", type=str, default="You are a helpful assistant."
        )
        p.add_argument(
            "--system_verify",
            type=str,
            default="You are a helpful assistant who can judge whether a claim is supported by the search results or not.",
        )

        # vLLM
        p.add_argument("--gpu_memory_utilization", type=float, default=0.55)
        p.add_argument("--temperature", type=float, default=0.01)
        p.add_argument("--max_tokens_extract", type=int, default=256)
        p.add_argument("--max_tokens_verify", type=int, default=64)
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--max_tries", type=int, default=3)

        # Compute / FLOPs (optional)
        p.add_argument(
            "--model_params_b",
            type=float,
            default=8.0,
            help="Model size in billions of parameters (for FLOPs estimate). Example: 8 for 8B.",
        )
        p.add_argument(
            "--flops_per_param",
            type=float,
            default=2.0,
            help="FLOPs per parameter per token multiplier (rough). Common: 2 or 6.",
        )

        # Stops
        p.add_argument("--stop_common", type=str, default="<|eot_id|>,</s>,<|im_end|>")
        p.add_argument(
            "--stop_extract_extra",
            type=str,
            default="\\nText:,\\n\\nText:,\\nQuestion:,\\n\\nQuestion:",
        )

        # Verdict parsing
        p.add_argument(
            "--strict_verdict",
            action="store_true",
            help="Only accept ###Supported###/###Unsupported### tags (no fallback).",
        )

        # Sentence aggregation policy for no-claim cases
        p.add_argument(
            "--empty_claims_policy",
            type=str,
            default="supported",
            choices=["supported", "not_supported", "skip"],
        )

        # VeriScore aggregation
        p.add_argument(
            "--veriscore_k_mode",
            type=str,
            default="median",
            choices=["fixed", "median"],
        )
        p.add_argument("--veriscore_k_fixed", type=int, default=1)

    # FactBench
    p_fb = sub.add_parser("factbench")
    add_common_flags(p_fb)
    p_fb.add_argument("--data_jsonl", type=str, required=True)
    p_fb.add_argument("--max_items", type=int, default=0)
    p_fb.add_argument("--fallback_to_sentence_on_extract_error", action="store_true")
    p_fb.set_defaults(func=run_factbench)

    # FELM
    p_felm = sub.add_parser("felm")
    add_common_flags(p_felm)
    p_felm.add_argument("--felm_dir", type=str, required=True)
    p_felm.add_argument("--subset", type=str, default="writing_rec")
    p_felm.add_argument("--split", type=str, default="test")
    p_felm.add_argument("--max_examples", type=int, default=0, help="0 = all")
    p_felm.add_argument("--skip_no_context", action="store_true")
    p_felm.set_defaults(func=run_felm)

    # ANAH
    p_anah = sub.add_parser("anah")
    add_common_flags(p_anah)
    p_anah.add_argument(
        "--anah_split",
        type=str,
        default="train",
        help="HuggingFace split to use. ANAH only has 'train'.",
    )
    p_anah.add_argument("--max_examples", type=int, default=0, help="0 = all")
    p_anah.add_argument("--skip_no_context", action="store_true")
    p_anah.add_argument(
        "--anah_sample_file",
        type=str,
        default="",
        help="Path to a pre-sampled ANAH jsonl. If set, skips HuggingFace download.",
    )
    p_anah.set_defaults(func=run_anah)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
