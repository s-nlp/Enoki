import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_from_disk
from factowl_utils import (
    apply_fail_policy,
    clean_seg,
    cm_to_macro_f1,
    debug_atomic_retry,
    flatten_evidence,
    flops_from_tokens,
    get_score_with_retries,
    iter_sentences_in_order,
    norm_gold_label,
    not_supported_risk_from_out,
    patch_factowl_atomic_extractor,
    pred_label_3class_from_out,
    ref_text_to_passages,
    roc_auc_manual,
    safe_div,
    save_json,
    save_jsonl,
    tflops_per_s,
    topic_from_prompt,
    topic_from_ref_or_prompt,
    update_confusion_not_supported_positive,
    vllm_session,
)

# FactOwl core
from factowl import FactScorerSpedUpVLLM as FactScorer


def _collect_factbench_evidence(
    sentence_row: Dict[str, Any], include_auto_evidence_url: bool
) -> List[str]:
    chunks: List[str] = []
    chunks += flatten_evidence(sentence_row.get("auto_evidence"))
    if include_auto_evidence_url:
        chunks += flatten_evidence(sentence_row.get("auto_evidence_url"))
    chunks += flatten_evidence(sentence_row.get("human_evidence"))
    return chunks


# FactBench runner
def run_factbench(args: argparse.Namespace) -> None:
    # Patch atomic extractor
    patch_factowl_atomic_extractor(
        template=args.atomic_template,
        set_examples=args.atomic_set_examples,
        verbose=args.verbose_patch,
    )

    # Load JSONL rows
    samples: List[Dict[str, Any]] = []
    with open(args.data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_samples and i >= args.max_samples:
                break
            samples.append(json.loads(line))

    # Build topic2passages. By default we keep evidence sentence-scoped to
    # match the Claimify sentence-level setup more closely.
    topic2passages: Dict[str, List[Dict[str, str]]] = {}
    sample_topics: List[Tuple[str, str]] = []

    for i, ex in enumerate(samples):
        base_topic = topic_from_prompt(ex.get("prompt", ""))
        sample_topic = f"{i}::{base_topic}"
        sent_items = iter_sentences_in_order(ex.get("sentences") or {})
        sample_topics.append((sample_topic, base_topic))

        if args.evidence_scope == "sample":
            ev_chunks: List[str] = []
            for _, s in sent_items:
                ev_chunks += _collect_factbench_evidence(
                    s, include_auto_evidence_url=args.include_auto_evidence_url
                )

            context_text = "\n\n".join(ev_chunks)
            topic2passages[sample_topic] = ref_text_to_passages(
                base_topic,
                context_text,
                max_passages=args.max_passages,
                max_chars=args.max_chars,
                wrap_long_paragraphs=True,
            )
            continue

        for sent_key, s in sent_items:
            sent_topic = f"{sample_topic}::{sent_key}"
            context_text = "\n\n".join(
                _collect_factbench_evidence(
                    s, include_auto_evidence_url=args.include_auto_evidence_url
                )
            )
            topic2passages[sent_topic] = ref_text_to_passages(
                base_topic,
                context_text,
                max_passages=args.max_passages,
                max_chars=args.max_chars,
                wrap_long_paragraphs=True,
            )

    # Metrics accumulators (ONLY on gold-defined sentences)
    total_segments = 0
    skipped_na = 0

    cnt_has_context = 0
    cnt_atoms = 0
    cnt_evaluable = 0

    fail = {
        "empty_segment": 0,
        "no_context": 0,
        "no_atoms_or_abstain": 0,
        "exception": 0,
    }

    cm_all = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    cm_evaluable = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    correct_all = 0
    correct_evaluable = 0

    sum_score = 0.0
    sum_nfpr = 0.0
    cnt_nfpr = 0
    cnt_score = 0

    segment_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    # For ROC-AUC (positive class = not_supported)
    y_true_all: List[int] = []
    y_score_all: List[float] = []
    y_true_eval: List[int] = []
    y_score_eval: List[float] = []

    with vllm_session(
        args.model, gpu_memory_utilization=args.gpu_memory_utilization
    ) as llm:
        fs = FactScorer(
            vllm_model=llm,
            model_name="retrieval+llama",
            context_type="wikipedia_api",
            use_this_topic2content_only=topic2passages,
            abstain_detection_type="generic",
            lang="en",
            verbose=False,
            debug=False,
        )
        fs.register_knowledge_source(name=args.knowledge_source)

        for i, ex in enumerate(samples):
            sample_topic, base_topic = sample_topics[i]
            sent_items = iter_sentences_in_order(ex.get("sentences") or {})

            factowl_pred3_list: List[str] = []
            factowl_pred2_list: List[str] = []
            factowl_fail_reason_list: List[str] = []
            factowl_rr_list: List[float] = []
            factowl_has_context_list: List[bool] = []
            gold_list: List[str] = []

            for sent_key, s in sent_items:
                dt = None
                seg_prompt_tok = 0
                seg_gen_tok = 0
                seg_total_tok = 0
                seg_flops = None
                gold_supported = norm_gold_label(s.get("sentence_factuality_label"))
                if gold_supported is None:
                    row_topic = (
                        sample_topic
                        if args.evidence_scope == "sample"
                        else f"{sample_topic}::{sent_key}"
                    )
                    skipped_na += 1
                    segment_rows.append(
                        {
                            "sample_id": i,
                            "topic": row_topic,
                            "sentence_key": sent_key,
                            "text": (s.get("decontext") or s.get("text") or "").strip(),
                            "gold": "NA",
                            "pred_binary": "SKIP",
                            "pred_3class": "SKIP",
                            "fail_reason": "gold_NA",
                            "has_context": len(topic2passages.get(row_topic, [])) > 0,
                            "factowl_error": None,
                            "atomic_retry": None,
                        }
                    )
                    continue

                total_segments += 1
                gold_label = "supported" if gold_supported else "not_supported"
                gold_list.append(gold_label)

                seg = (s.get("decontext") or s.get("text") or "").strip()
                topic = (
                    sample_topic
                    if args.evidence_scope == "sample"
                    else f"{sample_topic}::{sent_key}"
                )

                psgs = topic2passages.get(topic, [])
                has_context = len(psgs) > 0
                if has_context:
                    cnt_has_context += 1

                rr = 0.0
                pred3 = "ir"
                pred2_supported = None
                fail_reason = None
                out = None

                extra_debug = {"factowl_error": None, "atomic_retry": None}

                if not seg:
                    fail_reason = "empty_segment"
                    fail["empty_segment"] += 1

                elif not has_context:
                    fail_reason = "no_context"
                    fail["no_context"] += 1

                else:
                    usage_snap = llm.snapshot()
                    t0 = time.perf_counter()
                    out, err_info = get_score_with_retries(
                        fs,
                        topic=topic,
                        seg=seg,
                        knowledge_source=args.knowledge_source,
                        max_tries=args.max_tries,
                        base_sleep=args.base_sleep,
                    )

                    if err_info is not None:
                        fail_reason = "exception"
                        fail["exception"] += 1
                        extra_debug["factowl_error"] = err_info

                    else:
                        rr = float(out.get("respond_ratio", 0.0) or 0.0)
                        pred3 = pred_label_3class_from_out(out)

                        decisions = out.get("decisions") or []
                        has_atoms = (rr > 0.0) and any(d.get("atom") for d in decisions)
                        if has_atoms:
                            cnt_atoms += 1

                        if (rr == 0.0) or (pred3 == "ir") or (not has_atoms):
                            atomic_dbg = debug_atomic_retry(llm, seg)
                            fail_reason = "no_atoms_or_abstain"
                            fail["no_atoms_or_abstain"] += 1
                            extra_debug["atomic_retry"] = atomic_dbg
                        else:
                            cnt_evaluable += 1
                            pred2_supported = pred3 == "supported"

                            if out.get("score") is not None:
                                sum_score += float(out["score"])
                                cnt_score += 1
                            if out.get("num_facts_per_response") is not None:
                                sum_nfpr += float(out["num_facts_per_response"])
                                cnt_nfpr += 1

                    dt = time.perf_counter() - t0
                    usage_delta = llm.delta(usage_snap)
                    seg_prompt_tok = int(usage_delta.prompt_tokens)
                    seg_gen_tok = int(usage_delta.gen_tokens)
                    seg_total_tok = seg_prompt_tok + seg_gen_tok
                    seg_flops = None
                    if args.model_params_b and args.model_params_b > 0:
                        seg_flops = flops_from_tokens(
                            seg_total_tok, args.model_params_b, args.flops_per_param
                        )
                # Update confusion + accuracy
                if fail_reason is not None or pred2_supported is None:
                    forced_pred_supported = apply_fail_policy(
                        gold_supported, args.fail_policy
                    )
                    update_confusion_not_supported_positive(
                        cm_all, gold_supported, forced_pred_supported
                    )
                    pred2_label = "FAIL"
                    correct_all += int(forced_pred_supported == gold_supported)
                    pred2_supported = forced_pred_supported
                    pred2_label = (
                        "supported" if forced_pred_supported else "not_supported"
                    )
                else:
                    update_confusion_not_supported_positive(
                        cm_all, gold_supported, pred2_supported
                    )
                    update_confusion_not_supported_positive(
                        cm_evaluable, gold_supported, pred2_supported
                    )
                    correct_all += int(pred2_supported == gold_supported)
                    correct_evaluable += int(pred2_supported == gold_supported)
                    pred2_label = "supported" if pred2_supported else "not_supported"

                factowl_pred3_list.append(pred3)
                factowl_pred2_list.append(pred2_label)
                factowl_fail_reason_list.append(fail_reason or "")
                factowl_rr_list.append(rr)
                factowl_has_context_list.append(has_context)

                segment_rows.append(
                    {
                        "sample_id": i,
                        "topic": topic,
                        "prompt": ex.get("prompt"),
                        "sentence_key": sent_key,
                        "text": seg,
                        "gold": gold_label,
                        "gold_supported": bool(gold_supported),
                        "has_context": has_context,
                        "respond_ratio": rr,
                        "pred_3class": pred3,
                        "pred_binary": pred2_label,
                        "fail_reason": fail_reason or "",
                        "score": (
                            float(out.get("score"))
                            if out and out.get("score") is not None
                            else None
                        ),
                        "num_facts_per_response": (
                            float(out.get("num_facts_per_response"))
                            if out and out.get("num_facts_per_response") is not None
                            else None
                        ),
                        "decisions": out.get("decisions", []) if out else [],
                        "factowl_error": extra_debug["factowl_error"],
                        "atomic_retry": extra_debug["atomic_retry"],
                        "time_s": dt,
                        "prompt_tokens": seg_prompt_tok,
                        "gen_tokens": seg_gen_tok,
                        "total_tokens": seg_total_tok,
                        "flops": seg_flops,
                    }
                )

                # AUC (pos = not_supported)
                y_true = 0 if gold_supported else 1
                risk = not_supported_risk_from_out(out)

                if fail_reason is not None or risk is None:
                    # count FAIL as worst risk=1.0
                    y_true_all.append(y_true)
                    y_score_all.append(1.0)
                else:
                    y_true_all.append(y_true)
                    y_score_all.append(risk)
                    y_true_eval.append(y_true)
                    y_score_eval.append(risk)

            ex_out = dict(ex)
            ex_out["factowl_topic"] = sample_topic
            ex_out["factowl_base_topic"] = base_topic
            ex_out["gold_binary"] = gold_list
            ex_out["factowl_pred_3class"] = factowl_pred3_list
            ex_out["factowl_pred_binary"] = factowl_pred2_list
            ex_out["factowl_fail_reason"] = factowl_fail_reason_list
            ex_out["factowl_respond_ratio"] = factowl_rr_list
            ex_out["factowl_has_context"] = factowl_has_context_list
            ex_out["factowl_model_used"] = args.model
            example_rows.append(ex_out)

    # Final metrics
    coverage_context = safe_div(cnt_has_context, total_segments)
    coverage_atoms = safe_div(cnt_atoms, total_segments)
    coverage_evaluable = safe_div(cnt_evaluable, total_segments)

    m_all = cm_to_macro_f1(cm_all)
    m_eval = cm_to_macro_f1(cm_evaluable)

    acc_all = safe_div(correct_all, total_segments)
    acc_evaluable = safe_div(correct_evaluable, cnt_evaluable)

    auc_all = roc_auc_manual(y_true_all, y_score_all)
    auc_eval = roc_auc_manual(y_true_eval, y_score_eval)

    sum_time = sum(
        (r.get("time_s") or 0.0)
        for r in segment_rows
        if r.get("gold") not in {"NA"} and r.get("gold") is not None
    )
    sum_prompt_tok = sum(int(r.get("prompt_tokens") or 0) for r in segment_rows)
    sum_gen_tok = sum(int(r.get("gen_tokens") or 0) for r in segment_rows)
    sum_total_tok = sum_prompt_tok + sum_gen_tok
    sum_flops = sum(
        (r.get("flops") or 0.0) for r in segment_rows if r.get("flops") is not None
    )

    metrics = {
        "dataset": "factcheck-GPT-benchmark (FactBench)",
        "model_used": args.model,
        "evidence_scope": args.evidence_scope,
        "include_auto_evidence_url": bool(args.include_auto_evidence_url),
        "total_samples": len(samples),
        "total_segments_with_gold": total_segments,
        "skipped_segments_gold_NA": skipped_na,
        "coverage_context": coverage_context,
        "coverage_atoms": coverage_atoms,
        "coverage_evaluable": coverage_evaluable,
        "fail_breakdown": fail,
        "acc_all": acc_all,
        "precision_not_supported_all": m_all["precision_not_supported"],
        "recall_not_supported_all": m_all["recall_not_supported"],
        "f1_not_supported_all": m_all["f1_not_supported"],
        "precision_supported_all": m_all["precision_supported"],
        "recall_supported_all": m_all["recall_supported"],
        "f1_supported_all": m_all["f1_supported"],
        "f1_macro_all": m_all["f1_macro"],
        "confusion_matrix_all": m_all["confusion_matrix"],
        "acc_evaluable": acc_evaluable,
        "f1_macro_evaluable": m_eval["f1_macro"],
        "confusion_matrix_evaluable": m_eval["confusion_matrix"],
        "mean_score_evaluable": safe_div(sum_score, cnt_score),
        "mean_num_facts_per_response_evaluable": safe_div(sum_nfpr, cnt_nfpr),
        "auc_roc_all": auc_all,
        "auc_roc_evaluable": auc_eval,
        "auc_roc_n_all": len(y_true_all),
        "auc_roc_n_evaluable": len(y_true_eval),
        "fail_policy": args.fail_policy,
        "knowledge_source": args.knowledge_source,
        "compute": (
            None
            if args.model_params_b <= 0
            else {
                "params_b": args.model_params_b,
                "flops_per_param": args.flops_per_param,
                "sum_prompt_tokens": sum_prompt_tok,
                "sum_gen_tokens": sum_gen_tok,
                "sum_total_tokens": sum_total_tok,
                "sum_total_flops": sum_flops,
                "sum_time_s": sum_time,
                "total_tflops_per_s_agg": tflops_per_s(sum_flops, sum_time),
            }
        ),
    }

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, out_dir / "metrics.json")
    save_jsonl(segment_rows, out_dir / "segments_with_factowl.jsonl")
    save_jsonl(example_rows, out_dir / "examples_with_factowl.jsonl")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_dir.resolve()}")


# FELM runner
def run_felm(args: argparse.Namespace) -> None:
    # Patch atomic extractor
    patch_factowl_atomic_extractor(
        template=args.atomic_template,
        set_examples=args.atomic_set_examples,
        verbose=args.verbose_patch,
    )

    # Load FELM
    ds_path = Path(args.felm_dir) / args.subset
    ds = load_from_disk(str(ds_path))[args.split]
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Build topic2passages
    topic2passages: Dict[str, List[Dict[str, str]]] = {}
    rows: List[Tuple[str, Dict[str, Any], str]] = []
    for ex in ds:
        base_topic = topic_from_ref_or_prompt(ex.get("prompt"), ex.get("ref") or [])
        topic = f"{base_topic} :: {ex['index']}"
        psgs = ref_text_to_passages(
            base_topic,
            ex.get("ref_text") or "",
            max_passages=args.max_passages,
            max_chars=args.max_chars,
            wrap_long_paragraphs=args.wrap_long_paragraphs,
        )
        topic2passages[topic] = psgs
        rows.append((topic, dict(ex), base_topic))

    # Metrics accumulators
    total_segments = 0
    cnt_has_context = 0
    cnt_atoms = 0
    cnt_evaluable = 0

    fail = {
        "empty_segment": 0,
        "no_context": 0,
        "no_atoms_or_abstain": 0,
        "exception": 0,
    }

    cm_all = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    cm_evaluable = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    correct_all = 0
    correct_evaluable = 0

    sum_score = 0.0
    sum_nfpr = 0.0
    cnt_nfpr = 0
    cnt_score = 0

    segment_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    with vllm_session(
        args.model, gpu_memory_utilization=args.gpu_memory_utilization
    ) as llm:
        fs = FactScorer(
            vllm_model=llm,
            model_name="retrieval+llama",
            context_type="wikipedia_api",
            use_this_topic2content_only=topic2passages,
            abstain_detection_type="generic",
            lang="en",
            verbose=False,
            debug=False,
        )
        fs.register_knowledge_source(name=args.knowledge_source)

        for topic, ex, base_topic in rows:
            segs = ex["segmented_response"]
            labs = ex["labels"]  # bool list

            factowl_pred3_list: List[str] = []
            factowl_pred2_list: List[str] = []
            factowl_fail_reason_list: List[str] = []
            factowl_rr_list: List[float] = []
            factowl_has_context_list: List[bool] = []

            for seg_id, (seg, lab) in enumerate(zip(segs, labs)):
                dt = None
                seg_prompt_tok = 0
                seg_gen_tok = 0
                seg_total_tok = 0
                seg_flops = None
                total_segments += 1
                seg = clean_seg(seg)

                gold_supported = bool(lab)
                gold_label = "supported" if gold_supported else "not_supported"

                psgs = topic2passages.get(topic, [])
                has_context = len(psgs) > 0
                if has_context:
                    cnt_has_context += 1

                rr = 0.0
                pred3 = "ir"
                pred2_supported = None
                fail_reason = None
                out = None

                extra_debug = {"factowl_error": None, "atomic_retry": None}

                if not seg:
                    fail_reason = "empty_segment"
                    fail["empty_segment"] += 1

                elif not has_context:
                    fail_reason = "no_context"
                    fail["no_context"] += 1

                else:
                    usage_snap = llm.snapshot()
                    t0 = time.perf_counter()
                    out, err_info = get_score_with_retries(
                        fs,
                        topic=topic,
                        seg=seg,
                        knowledge_source=args.knowledge_source,
                        max_tries=args.max_tries,
                        base_sleep=args.base_sleep,
                    )

                    if err_info is not None:
                        fail_reason = "exception"
                        fail["exception"] += 1
                        extra_debug["factowl_error"] = err_info
                    else:
                        rr = float(out.get("respond_ratio", 0.0) or 0.0)
                        pred3 = pred_label_3class_from_out(out)

                        decisions = out.get("decisions") or []
                        has_atoms = (rr > 0.0) and any(d.get("atom") for d in decisions)
                        if has_atoms:
                            cnt_atoms += 1

                        if (rr == 0.0) or (pred3 == "ir") or (not has_atoms):
                            atomic_dbg = debug_atomic_retry(llm, seg)
                            fail_reason = "no_atoms_or_abstain"
                            fail["no_atoms_or_abstain"] += 1
                            extra_debug["atomic_retry"] = atomic_dbg
                        else:
                            cnt_evaluable += 1
                            pred2_supported = pred3 == "supported"

                            if out.get("score") is not None:
                                sum_score += float(out["score"])
                                cnt_score += 1
                            if out.get("num_facts_per_response") is not None:
                                sum_nfpr += float(out["num_facts_per_response"])
                                cnt_nfpr += 1

                    dt = time.perf_counter() - t0
                    usage_delta = llm.delta(usage_snap)
                    seg_prompt_tok = int(usage_delta.prompt_tokens)
                    seg_gen_tok = int(usage_delta.gen_tokens)
                    seg_total_tok = seg_prompt_tok + seg_gen_tok
                    seg_flops = None
                    if args.model_params_b and args.model_params_b > 0:
                        seg_flops = flops_from_tokens(
                            seg_total_tok, args.model_params_b, args.flops_per_param
                        )

                # FAIL policy for extraction/abstention cases.
                if fail_reason is not None or pred2_supported is None:
                    forced_pred_supported = apply_fail_policy(
                        gold_supported, args.fail_policy
                    )
                    update_confusion_not_supported_positive(
                        cm_all, gold_supported, forced_pred_supported
                    )
                    pred2_label = "FAIL"
                    correct_all += int(forced_pred_supported == gold_supported)
                    pred2_supported = forced_pred_supported
                    pred2_label = (
                        "supported" if forced_pred_supported else "not_supported"
                    )
                else:
                    update_confusion_not_supported_positive(
                        cm_all, gold_supported, pred2_supported
                    )
                    update_confusion_not_supported_positive(
                        cm_evaluable, gold_supported, pred2_supported
                    )
                    correct_all += int(pred2_supported == gold_supported)
                    correct_evaluable += int(pred2_supported == gold_supported)
                    pred2_label = "supported" if pred2_supported else "not_supported"

                factowl_pred3_list.append(pred3)
                factowl_pred2_list.append(pred2_label)
                factowl_fail_reason_list.append(fail_reason or "")
                factowl_rr_list.append(rr)
                factowl_has_context_list.append(has_context)

                segment_rows.append(
                    {
                        "subset": args.subset,
                        "split": args.split,
                        "example_index": ex["index"],
                        "topic": topic,
                        "base_topic": base_topic,
                        "seg_id": seg_id,
                        "text": seg,
                        "gold": gold_label,
                        "gold_supported": gold_supported,
                        "has_context": has_context,
                        "n_passages": len(psgs),
                        "respond_ratio": rr,
                        "pred_3class": pred3,
                        "pred_binary": pred2_label,
                        "fail_reason": fail_reason or "",
                        "score": (
                            float(out.get("score"))
                            if out and out.get("score") is not None
                            else None
                        ),
                        "num_facts_per_response": (
                            float(out.get("num_facts_per_response"))
                            if out and out.get("num_facts_per_response") is not None
                            else None
                        ),
                        "decisions": out.get("decisions", []) if out else [],
                        "factowl_error": extra_debug["factowl_error"],
                        "atomic_retry": extra_debug["atomic_retry"],
                        "time_s": dt,
                        "prompt_tokens": seg_prompt_tok,
                        "gen_tokens": seg_gen_tok,
                        "total_tokens": seg_total_tok,
                        "flops": seg_flops,
                    }
                )

            ex_out = dict(ex)
            ex_out["factowl_topic"] = topic
            ex_out["factowl_base_topic"] = base_topic
            ex_out["factowl_pred_3class"] = factowl_pred3_list
            ex_out["factowl_pred_binary"] = factowl_pred2_list
            ex_out["factowl_fail_reason"] = factowl_fail_reason_list
            ex_out["factowl_respond_ratio"] = factowl_rr_list
            ex_out["factowl_has_context"] = factowl_has_context_list
            ex_out["factowl_model_used"] = args.model
            example_rows.append(ex_out)

    # Metrics
    coverage_context = safe_div(cnt_has_context, total_segments)
    coverage_atoms = safe_div(cnt_atoms, total_segments)
    coverage_evaluable = safe_div(cnt_evaluable, total_segments)

    m_all = cm_to_macro_f1(cm_all)
    m_eval = cm_to_macro_f1(cm_evaluable)

    acc_all = safe_div(correct_all, total_segments)
    acc_evaluable = safe_div(correct_evaluable, cnt_evaluable)

    sum_time = sum(
        (r.get("time_s") or 0.0)
        for r in segment_rows
        if r.get("gold") not in {"NA"} and r.get("gold") is not None
    )
    sum_prompt_tok = sum(int(r.get("prompt_tokens") or 0) for r in segment_rows)
    sum_gen_tok = sum(int(r.get("gen_tokens") or 0) for r in segment_rows)
    sum_total_tok = sum_prompt_tok + sum_gen_tok
    sum_flops = sum(
        (r.get("flops") or 0.0) for r in segment_rows if r.get("flops") is not None
    )

    metrics = {
        "dataset": "FELM (offline ref_text)",
        "subset": args.subset,
        "split": args.split,
        "model_used": args.model,
        "total_examples": len(rows),
        "total_segments": total_segments,
        "coverage_context": coverage_context,
        "coverage_atoms": coverage_atoms,
        "coverage_evaluable": coverage_evaluable,
        "fail_breakdown": fail,
        "acc_all": acc_all,
        "precision_not_supported_all": m_all["precision_not_supported"],
        "recall_not_supported_all": m_all["recall_not_supported"],
        "f1_not_supported_all": m_all["f1_not_supported"],
        "precision_supported_all": m_all["precision_supported"],
        "recall_supported_all": m_all["recall_supported"],
        "f1_supported_all": m_all["f1_supported"],
        "f1_macro_all": m_all["f1_macro"],
        "confusion_matrix_all": m_all["confusion_matrix"],
        "acc_evaluable": acc_evaluable,
        "f1_macro_evaluable": m_eval["f1_macro"],
        "confusion_matrix_evaluable": m_eval["confusion_matrix"],
        "mean_score_evaluable": safe_div(sum_score, cnt_score),
        "mean_num_facts_per_response_evaluable": safe_div(sum_nfpr, cnt_nfpr),
        "fail_policy": args.fail_policy,
        "knowledge_source": args.knowledge_source,
        "compute": (
            None
            if args.model_params_b <= 0
            else {
                "params_b": args.model_params_b,
                "flops_per_param": args.flops_per_param,
                "sum_prompt_tokens": sum_prompt_tok,
                "sum_gen_tokens": sum_gen_tok,
                "sum_total_tokens": sum_total_tok,
                "sum_total_flops": sum_flops,
                "sum_time_s": sum_time,
                "total_tflops_per_s_agg": tflops_per_s(sum_flops, sum_time),
            }
        ),
    }

    out_dir = Path(args.out_root) / args.subset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, out_dir / "metrics.json")
    save_jsonl(segment_rows, out_dir / "segments_with_factowl.jsonl")
    save_jsonl(example_rows, out_dir / "examples_with_factowl.jsonl")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# ANAH runner
# ---------------------------------------------------------------------------
def run_anah(args: argparse.Namespace) -> None:
    """Run FactOwl on the ANAH (opencompass/anah) dataset.

    ANAH is a sentence-level hallucination annotation dataset.  Each example
    contains annotated sentences aligned with their reference fragments.
    We use anah_utils to parse the dataset correctly, skip 'No Fact' rows,
    and treat the ann_reference fragment as the per-sentence evidence pool.
    """
    import sys as _sys, os as _os  # noqa: PLC0415
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from anah_utils import iter_anah_sentences  # noqa: PLC0415

    try:
        from datasets import load_dataset as _load_anah  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError(
            "datasets is not installed, but 'anah' subcommand requires it: "
            "pip install datasets"
        ) from exc

    patch_factowl_atomic_extractor(
        template=args.atomic_template,
        set_examples=args.atomic_set_examples,
        verbose=args.verbose_patch,
    )

    if args.anah_sample_file:
        import json as _json
        with open(args.anah_sample_file, encoding="utf-8") as _f:
            _sample_rows = [_json.loads(l) for l in _f if l.strip()]
        _anah_iter = iter(_sample_rows)
        _n_total = len(_sample_rows)
        print(f"ANAH: loaded {_n_total} rows from sample file '{args.anah_sample_file}'", flush=True)
    else:
        ds = _load_anah("opencompass/anah", split=args.anah_split)
        _anah_iter = iter_anah_sentences(ds, max_examples=args.max_samples or 0)
        _n_total = len(ds)

    # Pre-build topic2passages (keyed by unique sentence key)
    topic2passages: Dict[str, List[Dict[str, str]]] = {}
    anah_rows: List[Tuple[str, Dict[str, Any], str]] = []

    skipped_no_fact = 0
    for sent_row in _anah_iter:
        if sent_row["gold_supported"] is None:
            skipped_no_fact += 1
            continue  # No Fact – skip

        question = clean_seg(sent_row["question"])
        sentence = clean_seg(sent_row["sentence"])
        hallucination_type = sent_row["hallucination_type"]
        ex_i = sent_row["example_index"]
        # Use ann_reference (specific cited fragment) as per-sentence evidence
        reference = sent_row["ann_reference"]
        gold_supported: bool = bool(sent_row["gold_supported"])

        base_topic = question[:80] or f"anah_{ex_i}"
        topic = f"{base_topic}::{ex_i}_{sent_row['answer_index']}_{sent_row['sentence_index']}"

        psgs = ref_text_to_passages(
            base_topic,
            reference,
            max_passages=args.max_passages,
            max_chars=args.max_chars,
            wrap_long_paragraphs=getattr(args, "wrap_long_paragraphs", False),
        )
        topic2passages[topic] = psgs

        row_ex = {
            "index": ex_i,
            "answer_index": sent_row["answer_index"],
            "sentence_index": sent_row["sentence_index"],
            "question": question,
            "sentence": sentence,
            "hallucination_type": hallucination_type,
            "gold_supported": gold_supported,
        }
        anah_rows.append((topic, row_ex, base_topic))

    # Metrics accumulators
    total_segments = 0
    cnt_has_context = 0
    cnt_atoms = 0
    cnt_evaluable = 0

    fail = {
        "empty_segment": 0,
        "no_context": 0,
        "no_atoms_or_abstain": 0,
        "exception": 0,
    }

    cm_all = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    cm_evaluable = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    correct_all = 0
    correct_evaluable = 0

    sum_score = 0.0
    sum_nfpr = 0.0
    cnt_nfpr = 0
    cnt_score = 0

    segment_rows: List[Dict[str, Any]] = []

    with vllm_session(args.model, gpu_memory_utilization=args.gpu_memory_utilization) as llm:
        fs = FactScorer(
            vllm_model=llm,
            model_name="retrieval+llama",
            context_type="wikipedia_api",
            use_this_topic2content_only=topic2passages,
            abstain_detection_type="generic",
            lang="en",
            verbose=False,
            debug=False,
        )
        fs.register_knowledge_source(name=args.knowledge_source)

        for topic, ex, base_topic in anah_rows:
            seg = clean_seg(ex["sentence"])
            gold_supported = bool(ex["gold_supported"])
            gold_label = "supported" if gold_supported else "not_supported"

            dt = None
            seg_prompt_tok = 0
            seg_gen_tok = 0
            seg_total_tok = 0
            seg_flops = None
            total_segments += 1

            psgs = topic2passages.get(topic, [])
            has_context = len(psgs) > 0
            if has_context:
                cnt_has_context += 1

            rr = 0.0
            pred3 = "ir"
            pred2_supported = None
            fail_reason = None
            out = None
            extra_debug = {"factowl_error": None, "atomic_retry": None}

            if not seg:
                fail_reason = "empty_segment"
                fail["empty_segment"] += 1
            elif not has_context:
                fail_reason = "no_context"
                fail["no_context"] += 1
            else:
                usage_snap = llm.snapshot()
                t0 = time.perf_counter()
                out, err_info = get_score_with_retries(
                    fs,
                    topic=topic,
                    seg=seg,
                    knowledge_source=args.knowledge_source,
                    max_tries=args.max_tries,
                    base_sleep=args.base_sleep,
                )

                if err_info is not None:
                    fail_reason = "exception"
                    fail["exception"] += 1
                    extra_debug["factowl_error"] = err_info
                else:
                    rr = float(out.get("respond_ratio", 0.0) or 0.0)
                    pred3 = pred_label_3class_from_out(out)

                    decisions = out.get("decisions") or []
                    has_atoms = (rr > 0.0) and any(d.get("atom") for d in decisions)
                    if has_atoms:
                        cnt_atoms += 1

                    if (rr == 0.0) or (pred3 == "ir") or (not has_atoms):
                        atomic_dbg = debug_atomic_retry(llm, seg)
                        fail_reason = "no_atoms_or_abstain"
                        fail["no_atoms_or_abstain"] += 1
                        extra_debug["atomic_retry"] = atomic_dbg
                    else:
                        cnt_evaluable += 1
                        pred2_supported = pred3 == "supported"

                        if out.get("score") is not None:
                            sum_score += float(out["score"])
                            cnt_score += 1
                        if out.get("num_facts_per_response") is not None:
                            sum_nfpr += float(out["num_facts_per_response"])
                            cnt_nfpr += 1

                dt = time.perf_counter() - t0
                usage_delta = llm.delta(usage_snap)
                seg_prompt_tok = int(usage_delta.prompt_tokens)
                seg_gen_tok = int(usage_delta.gen_tokens)
                seg_total_tok = seg_prompt_tok + seg_gen_tok
                if args.model_params_b and args.model_params_b > 0:
                    seg_flops = flops_from_tokens(
                        seg_total_tok, args.model_params_b, args.flops_per_param
                    )

            if fail_reason is not None or pred2_supported is None:
                forced_pred_supported = apply_fail_policy(gold_supported, args.fail_policy)
                update_confusion_not_supported_positive(
                    cm_all, gold_supported, forced_pred_supported
                )
                correct_all += int(forced_pred_supported == gold_supported)
                pred2_supported = forced_pred_supported
                pred2_label = "supported" if forced_pred_supported else "not_supported"
            else:
                update_confusion_not_supported_positive(
                    cm_all, gold_supported, pred2_supported
                )
                update_confusion_not_supported_positive(
                    cm_evaluable, gold_supported, pred2_supported
                )
                correct_all += int(pred2_supported == gold_supported)
                correct_evaluable += int(pred2_supported == gold_supported)
                pred2_label = "supported" if pred2_supported else "not_supported"

            segment_rows.append(
                {
                    "split": args.anah_split,
                    "example_index": ex["index"],
                    "answer_index": ex.get("answer_index"),
                    "sentence_index": ex.get("sentence_index"),
                    "topic": topic,
                    "base_topic": base_topic,
                    "question": ex["question"],
                    "hallucination_type": ex["hallucination_type"],
                    "text": seg,
                    "gold": gold_label,
                    "gold_supported": gold_supported,
                    "has_context": has_context,
                    "n_passages": len(psgs),
                    "respond_ratio": rr,
                    "pred_3class": pred3,
                    "pred_binary": pred2_label,
                    "fail_reason": fail_reason or "",
                    "score": (
                        float(out.get("score"))
                        if out and out.get("score") is not None
                        else None
                    ),
                    "num_facts_per_response": (
                        float(out.get("num_facts_per_response"))
                        if out and out.get("num_facts_per_response") is not None
                        else None
                    ),
                    "decisions": out.get("decisions", []) if out else [],
                    "factowl_error": extra_debug["factowl_error"],
                    "atomic_retry": extra_debug["atomic_retry"],
                    "time_s": dt,
                    "prompt_tokens": seg_prompt_tok,
                    "gen_tokens": seg_gen_tok,
                    "total_tokens": seg_total_tok,
                    "flops": seg_flops,
                }
            )

    # Metrics
    coverage_context = safe_div(cnt_has_context, total_segments)
    coverage_atoms = safe_div(cnt_atoms, total_segments)
    coverage_evaluable = safe_div(cnt_evaluable, total_segments)

    m_all = cm_to_macro_f1(cm_all)
    m_eval = cm_to_macro_f1(cm_evaluable)

    acc_all = safe_div(correct_all, total_segments)
    acc_evaluable = safe_div(correct_evaluable, cnt_evaluable)

    sum_time = sum((r.get("time_s") or 0.0) for r in segment_rows)
    sum_prompt_tok = sum(int(r.get("prompt_tokens") or 0) for r in segment_rows)
    sum_gen_tok = sum(int(r.get("gen_tokens") or 0) for r in segment_rows)
    sum_total_tok = sum_prompt_tok + sum_gen_tok
    sum_flops = sum(
        (r.get("flops") or 0.0) for r in segment_rows if r.get("flops") is not None
    )

    metrics = {
        "dataset": "ANAH (opencompass/anah)",
        "split": args.anah_split,
        "model_used": args.model,
        "total_examples_in_split": _n_total,
        "skipped_no_fact": skipped_no_fact,
        "total_segments": total_segments,
        "coverage_context": coverage_context,
        "coverage_atoms": coverage_atoms,
        "coverage_evaluable": coverage_evaluable,
        "fail_breakdown": fail,
        "acc_all": acc_all,
        "precision_not_supported_all": m_all["precision_not_supported"],
        "recall_not_supported_all": m_all["recall_not_supported"],
        "f1_not_supported_all": m_all["f1_not_supported"],
        "precision_supported_all": m_all["precision_supported"],
        "recall_supported_all": m_all["recall_supported"],
        "f1_supported_all": m_all["f1_supported"],
        "f1_macro_all": m_all["f1_macro"],
        "confusion_matrix_all": m_all["confusion_matrix"],
        "acc_evaluable": acc_evaluable,
        "f1_macro_evaluable": m_eval["f1_macro"],
        "confusion_matrix_evaluable": m_eval["confusion_matrix"],
        "mean_score_evaluable": safe_div(sum_score, cnt_score),
        "mean_num_facts_per_response_evaluable": safe_div(sum_nfpr, cnt_nfpr),
        "fail_policy": args.fail_policy,
        "knowledge_source": args.knowledge_source,
        "compute": (
            None
            if args.model_params_b <= 0
            else {
                "params_b": args.model_params_b,
                "flops_per_param": args.flops_per_param,
                "sum_prompt_tokens": sum_prompt_tok,
                "sum_gen_tokens": sum_gen_tok,
                "sum_total_tokens": sum_total_tok,
                "sum_total_flops": sum_flops,
                "sum_time_s": sum_time,
                "total_tflops_per_s_agg": tflops_per_s(sum_flops, sum_time),
            }
        ),
    }

    out_dir = Path(args.out_root) / "anah" / args.anah_split
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, out_dir / "metrics.json")
    save_jsonl(segment_rows, out_dir / "segments_with_factowl.jsonl")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_dir.resolve()}")


# CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("factowl_run.py")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # Shared-ish flags helper
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out_root", type=str, default="cachedir_factowl")
        p.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
        p.add_argument("--gpu_memory_utilization", type=float, default=0.5)
        p.add_argument("--max_passages", type=int, default=60)
        p.add_argument("--max_chars", type=int, default=1200)
        p.add_argument("--max_samples", type=int, default=0, help="0 = all")

        # FLOPs estimate
        p.add_argument(
            "--model_params_b",
            type=float,
            default=8.0,
            help="Model size in billions of params (e.g., 8 for 8B). 0 disables FLOPs.",
        )
        p.add_argument(
            "--flops_per_param",
            type=float,
            default=2.0,
            help="FLOPs per param per token multiplier (rough). Common: 2 or 6.",
        )

        # retries for fs.get_score
        p.add_argument("--max_tries", type=int, default=3)
        p.add_argument("--base_sleep", type=float, default=1.0)

        # factowl patch controls
        p.add_argument(
            "--atomic_template",
            type=str,
            default="sentence",
            choices=["sentence", "entity"],
        )
        p.add_argument(
            "--atomic_set_examples",
            action="store_true",
            help="If set, override FACT_GENERATION_EXAMPLES",
        )
        p.add_argument("--verbose_patch", action="store_true")

        # FAIL policy
        p.add_argument(
            "--fail_policy",
            type=str,
            default=None,
            choices=["not_supported", "supported", "opposite", "same"],
            help=(
                "What to predict when FAIL happens. "
                "Use 'not_supported' for reported metrics; "
                "'same'/'opposite' are gold-conditioned debug modes."
            ),
        )

        # knowledge source name used in fs.register_knowledge_source and fs.get_score
        p.add_argument("--knowledge_source", type=str, default=None)

    # FactBench
    p_fb = sub.add_parser(
        "factbench", help="Run FactOwl on factcheck-GPT-benchmark.jsonl"
    )
    add_common(p_fb)
    p_fb.add_argument(
        "--data", type=str, required=True, help="Path to factcheck-GPT-benchmark.jsonl"
    )
    p_fb.add_argument(
        "--evidence_scope",
        type=str,
        default="sentence",
        choices=["sentence", "sample"],
        help="Use sentence-only evidence or concatenate evidence across the sample.",
    )
    p_fb.add_argument(
        "--include_auto_evidence_url",
        action="store_true",
        help="Also include auto_evidence_url when building FactBench evidence passages.",
    )
    p_fb.set_defaults(_runner=run_factbench)

    # FELM
    p_felm = sub.add_parser("felm", help="Run FactOwl on FELM dataset (load_from_disk)")
    add_common(p_felm)
    p_felm.add_argument(
        "--felm_dir",
        type=str,
        required=True,
        help="Path to felm_with_ref_text (directory with subsets)",
    )
    p_felm.add_argument("--subset", type=str, default="writing_rec")
    p_felm.add_argument("--split", type=str, default="test")
    p_felm.add_argument(
        "--wrap_long_paragraphs",
        action="store_true",
        help="If set, wrap each paragraph to max_chars (FactBench-like).",
    )
    p_felm.set_defaults(_runner=run_felm)

    # ANAH
    p_anah = sub.add_parser(
        "anah", help="Run FactOwl on ANAH dataset (opencompass/anah via HuggingFace)"
    )
    add_common(p_anah)
    p_anah.add_argument(
        "--anah_split",
        type=str,
        default="train",
        help="HuggingFace split for ANAH. Only 'train' exists.",
    )
    p_anah.add_argument(
        "--anah_sample_file",
        type=str,
        default="",
        help="Path to a pre-sampled ANAH jsonl. If set, skips HuggingFace download.",
    )
    p_anah.add_argument(
        "--wrap_long_paragraphs",
        action="store_true",
        help="If set, wrap each paragraph to max_chars.",
    )
    p_anah.set_defaults(_runner=run_anah)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    # set dataset-specific defaults if user didn't override
    if args.cmd == "factbench":
        if args.fail_policy is None:
            args.fail_policy = "not_supported"
        if args.knowledge_source is None:
            args.knowledge_source = "factbench"
    elif args.cmd == "felm":
        if args.fail_policy is None:
            args.fail_policy = "not_supported"
        if args.knowledge_source is None:
            args.knowledge_source = "felm"
    elif args.cmd == "anah":
        if args.fail_policy is None:
            args.fail_policy = "not_supported"
        if args.knowledge_source is None:
            args.knowledge_source = "anah"

    if args.fail_policy in {"same", "opposite"}:
        print(
            "[warn] Using a gold-conditioned fail policy for FAIL cases. "
            "This is useful for debugging, but it should not be reported as a "
            "real evaluation setting.",
            flush=True,
        )

    args._runner(args)


if __name__ == "__main__":
    main()
