import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from safe_utils import (
    ATOMIC_PROMPT,
    RELEVANCE_PROMPT,
    REVISE_PROMPT,
    VERIFY_PROMPT,
    VLLMChat,
    _safe_tflops_per_s,
    _sum_flops,
    _sum_time,
    add_flops,
    clean_seg,
    cm_to_macro_f1,
    extract_code_block,
    finalize_row_times_and_tokens,
    flatten_evidence,
    iter_sentences_in_order,
    merge_list_markers_with_labels,
    norm_gold_label,
    not_supported_risk_from_verdicts,
    parse_atoms,
    parse_relevance,
    parse_supported_not_supported,
    passages_to_knowledge,
    ref_text_to_passages,
    roc_auc_manual,
    safe_div,
    save_json,
    save_jsonl,
    topic_from_prompt,
    topic_from_ref_or_prompt,
    update_confusion_not_supported_positive,
    vllm_session,
)

# Optional import for FELM only
try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None


def norm_gold_label_felm(x: Any) -> Optional[bool]:
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


def run_pipeline_on_sentences(
    *,
    chat: VLLMChat,
    question: str,
    full_answer: str,
    knowledge: str,
    sent_keys: List[str],
    sent_texts: List[str],
    gold_supported_list: List[Optional[bool]],
    model_name: str,
    args: argparse.Namespace,
    is_factbench: bool,
) -> Dict[str, Any]:
    """
    Shared SAFE-like loop. Returns:
      - segment_rows
      - pred3_list/pred2_list/fail_list/rr_list/has_context_list
      - metrics accumulators (cm, counts, auc arrays if factbench)
    """
    has_context = bool((knowledge or "").strip())

    # accumulators
    total_segments_with_gold = 0
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

    # AUC (FactBench script had it; keep it only there by default)
    y_true_all: List[int] = []
    y_score_all: List[float] = []
    y_true_eval: List[int] = []
    y_score_eval: List[float] = []

    # output per-example lists
    pred3_list: List[str] = []
    pred2_list: List[str] = []
    fail_list: List[str] = []
    rr_list: List[float] = []
    has_context_list: List[bool] = []

    segment_rows: List[Dict[str, Any]] = []

    # Per-sentence accumulators for timing/tokens, aligned with sent_texts indices
    timing_by_i = []
    tokens_by_i = []
    for _ in sent_texts:
        timing_by_i.append(
            {
                "atomic_s": 0.0,
                "revise_s": 0.0,
                "relevance_s": 0.0,
                "verify_s": 0.0,
                "extract_s": 0.0,
                "total_s": 0.0,
            }
        )
        tokens_by_i.append(
            {
                "atomic_prompt": 0,
                "atomic_gen": 0,
                "revise_prompt": 0,
                "revise_gen": 0,
                "relevance_prompt": 0,
                "relevance_gen": 0,
                "verify_prompt": 0,
                "verify_gen": 0,
                "total_prompt": 0,
                "total_gen": 0,
            }
        )

    # 1) atomic prompts for all sentences
    atomic_prompts = []
    for t in sent_texts:
        u = ATOMIC_PROMPT.format(question=question, sentence=t)
        atomic_prompts.append(chat.render("You extract atomic facts as bullets.", u))

    atomic_outs: List[str] = []
    bs = max(1, int(args.batch_size_atomic))
    for start in range(0, len(atomic_prompts), bs):
        chunk = atomic_prompts[start : start + bs]
        br = chat.generate_with_usage(
            chunk,
            temperature=0.0,
            max_tokens=args.atomic_max_tokens,
        )
        atomic_outs.extend(br.texts)
        # attribute equal per-item time + per-item usage to each sentence in this chunk
        for j in range(len(chunk)):
            i_sent = start + j
            timing_by_i[i_sent]["atomic_s"] += br.per_item_time_s
            tokens_by_i[i_sent]["atomic_prompt"] += br.usage[j].prompt_tokens
            tokens_by_i[i_sent]["atomic_gen"] += br.usage[j].gen_tokens

    # 2) per-sentence pipeline
    for sent_idx, (sent_key, sentence, gold_supported, atomic_raw) in enumerate(
        zip(sent_keys, sent_texts, gold_supported_list, atomic_outs)
    ):
        # FactBench: skip gold == NA
        if is_factbench and gold_supported is None:
            skipped_na += 1
            pred3_list.append("SKIP")
            pred2_list.append("SKIP")
            fail_list.append("gold_NA")
            rr_list.append(0.0)
            has_context_list.append(has_context)

            segment_rows.append(
                {
                    "sentence_key": sent_key,
                    "text": sentence,
                    "gold": "NA",
                    "pred_3class": "SKIP",
                    "pred_binary": "SKIP",
                    "fail_reason": "gold_NA",
                    "has_context": has_context,
                    "respond_ratio": 0.0,
                    "num_facts": 0,
                    "num_relevant_facts": 0,
                    "risk_not_supported": None,
                    "decisions": [],
                    "raw_atomic": atomic_raw,
                    "model_used": model_name,
                    "timing": timing_by_i[sent_idx],
                    "tokens": tokens_by_i[sent_idx],
                }
            )
            continue

        # gold must exist for felm; for factbench here it exists too
        assert gold_supported is not None, "gold_supported must be set here"

        total_segments_with_gold += 1
        gold_label = "supported" if gold_supported else "not_supported"
        if has_context:
            cnt_has_context += 1

        rr = 0.0
        pred3 = "ir"
        pred2_supported: Optional[bool] = None
        fail_reason: Optional[str] = None
        decisions: List[Dict[str, Any]] = []

        num_facts = 0
        num_rel_facts = 0
        risk: Optional[float] = None

        if not (sentence or "").strip():
            fail_reason = "empty_segment"
            fail["empty_segment"] += 1
        elif not has_context:
            fail_reason = "no_context"
            fail["no_context"] += 1
        else:
            try:
                facts = parse_atoms(atomic_raw)
                num_facts = len(facts)

                if num_facts > 0:
                    rr = 1.0
                    cnt_atoms += 1

                if not facts:
                    fail_reason = "no_atoms_or_abstain"
                    fail["no_atoms_or_abstain"] += 1
                else:
                    # revise
                    revise_prompts = []
                    for f in facts:
                        u = REVISE_PROMPT.format(
                            question=question, full_answer=full_answer, fact=f
                        )
                        revise_prompts.append(
                            chat.render("Rewrite as self-contained fact in ``` ```.", u)
                        )

                    revised_raws: List[str] = []
                    b2 = max(1, int(args.batch_size_small))
                    for start in range(0, len(revise_prompts), b2):
                        chunk = revise_prompts[start : start + b2]
                        br = chat.generate_with_usage(
                            chunk,
                            temperature=0.0,
                            max_tokens=args.revise_max_tokens,
                        )
                        revised_raws.extend(br.texts)
                        # all revise prompts belong to this sentence
                        timing_by_i[sent_idx]["revise_s"] += br.per_item_time_s * len(
                            chunk
                        )
                        for u1 in br.usage[: len(chunk)]:
                            tokens_by_i[sent_idx]["revise_prompt"] += u1.prompt_tokens
                            tokens_by_i[sent_idx]["revise_gen"] += u1.gen_tokens

                    revised_facts = []
                    for orig, rrw in zip(facts, revised_raws):
                        cb = extract_code_block(rrw)
                        revised_facts.append(cb if cb else orig)

                    # relevance
                    rel_prompts = []
                    for rf in revised_facts:
                        u = RELEVANCE_PROMPT.format(
                            question=question, full_answer=full_answer, fact=rf
                        )
                        rel_prompts.append(
                            chat.render("Return exactly [Foo] or [Not Foo].", u)
                        )

                    rel_raws: List[str] = []
                    for start in range(0, len(rel_prompts), b2):
                        chunk = rel_prompts[start : start + b2]
                        br = chat.generate_with_usage(
                            chunk,
                            temperature=0.0,
                            max_tokens=args.relevance_max_tokens,
                        )
                        rel_raws.extend(br.texts)
                        timing_by_i[sent_idx][
                            "relevance_s"
                        ] += br.per_item_time_s * len(chunk)
                        for u1 in br.usage[: len(chunk)]:
                            tokens_by_i[sent_idx][
                                "relevance_prompt"
                            ] += u1.prompt_tokens
                            tokens_by_i[sent_idx]["relevance_gen"] += u1.gen_tokens

                    relevant: List[str] = []
                    for rf, relr in zip(revised_facts, rel_raws):
                        is_rel = parse_relevance(relr)
                        decisions.append(
                            {
                                "atom": rf,
                                "stage": "relevance",
                                "raw": relr,
                                "is_relevant": is_rel,
                            }
                        )
                        if is_rel:
                            relevant.append(rf)

                    num_rel_facts = len(relevant)

                    if not relevant:
                        # IMPORTANT: irrelevant => IR, NOT FAIL
                        pred3 = "ir"
                        pred2_supported = None
                        risk = None
                    else:
                        # verify
                        ver_prompts = []
                        for rf in relevant:
                            u = VERIFY_PROMPT.format(knowledge=knowledge, statement=rf)
                            ver_prompts.append(
                                chat.render(
                                    "Return final answer in [Supported]/[Not Supported].",
                                    u,
                                )
                            )

                        ver_raws: List[str] = []
                        for start in range(0, len(ver_prompts), b2):
                            chunk = ver_prompts[start : start + b2]
                            br = chat.generate_with_usage(
                                chunk,
                                temperature=0.0,
                                max_tokens=args.verify_max_tokens,
                            )
                            ver_raws.extend(br.texts)
                            timing_by_i[sent_idx][
                                "verify_s"
                            ] += br.per_item_time_s * len(chunk)
                            for u1 in br.usage[: len(chunk)]:
                                tokens_by_i[sent_idx][
                                    "verify_prompt"
                                ] += u1.prompt_tokens
                                tokens_by_i[sent_idx]["verify_gen"] += u1.gen_tokens

                        verdicts: List[str] = []
                        for rf, vr in zip(relevant, ver_raws):
                            lab2 = parse_supported_not_supported(vr)
                            verdicts.append(lab2)
                            decisions.append(
                                {
                                    "atom": rf,
                                    "stage": "verify",
                                    "raw": vr,
                                    "verdict": lab2,
                                }
                            )

                        risk = not_supported_risk_from_verdicts(verdicts)

                        if any(v == "not_supported" for v in verdicts):
                            pred3 = "not_supported"
                        elif any(v == "supported" for v in verdicts):
                            pred3 = "supported"
                        else:
                            pred3 = "ir"

                        if pred3 in {"supported", "not_supported"}:
                            cnt_evaluable += 1
                            pred2_supported = pred3 == "supported"
                        else:
                            pred2_supported = None

            except Exception:
                fail_reason = "exception"
                fail["exception"] += 1
                decisions.append(
                    {"stage": "exception", "traceback": traceback.format_exc()}
                )

        # Metrics update
        y_true = 0 if gold_supported else 1  # pos = not_supported
        if fail_reason is not None:
            # forced wrong
            forced_pred_supported = not gold_supported
            update_confusion_not_supported_positive(
                cm_all, gold_supported, forced_pred_supported
            )
            pred2_label = "FAIL"

            if is_factbench:
                # AUC_all counts FAIL as worst
                y_true_all.append(y_true)
                y_score_all.append(1.0)
        else:
            if pred2_supported is None:
                pred2_label = "ir"
                # Keep acc_all and cm_all consistent: count IR as incorrect in cm_all
                forced_pred_supported = not gold_supported  # always wrong
                update_confusion_not_supported_positive(
                    cm_all, gold_supported, forced_pred_supported
                )
                if is_factbench and risk is not None:
                    y_true_all.append(y_true)
                    y_score_all.append(float(risk))
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

                if is_factbench:
                    if risk is not None:
                        y_true_all.append(y_true)
                        y_score_all.append(float(risk))
                        y_true_eval.append(y_true)
                        y_score_eval.append(float(risk))
                    else:
                        y_true_all.append(y_true)
                        y_score_all.append(1.0)

        pred3_list.append(pred3)
        pred2_list.append(pred2_label)
        fail_list.append(fail_reason or "")
        rr_list.append(rr)
        has_context_list.append(has_context)

        segment_rows.append(
            {
                "sentence_key": sent_key,
                "text": sentence,
                "gold": gold_label,
                "gold_supported": bool(gold_supported),
                "has_context": has_context,
                "respond_ratio": rr,
                "pred_3class": pred3,
                "pred_binary": pred2_label,
                "fail_reason": fail_reason or "",
                "num_facts": int(num_facts),
                "num_relevant_facts": int(num_rel_facts),
                "risk_not_supported": risk,
                "decisions": decisions,
                "raw_atomic": atomic_raw,
                "model_used": model_name,
                "timing": timing_by_i[sent_idx],
                "tokens": tokens_by_i[sent_idx],
            }
        )

    return {
        "total_segments_with_gold": total_segments_with_gold,
        "skipped_na": skipped_na,
        "cnt_has_context": cnt_has_context,
        "cnt_atoms": cnt_atoms,
        "cnt_evaluable": cnt_evaluable,
        "fail": fail,
        "cm_all": cm_all,
        "cm_evaluable": cm_evaluable,
        "correct_all": correct_all,
        "correct_evaluable": correct_evaluable,
        "y_true_all": y_true_all,
        "y_score_all": y_score_all,
        "y_true_eval": y_true_eval,
        "y_score_eval": y_score_eval,
        "pred3_list": pred3_list,
        "pred2_list": pred2_list,
        "fail_list": fail_list,
        "rr_list": rr_list,
        "has_context_list": has_context_list,
        "segment_rows": segment_rows,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, choices=["factbench", "felm", "anah", "ragtruth"], required=True)

    # FactBench args
    ap.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to factcheck-GPT-benchmark.jsonl (FactBench)",
    )

    # FELM args
    ap.add_argument("--felm_dir", type=str, default="")
    ap.add_argument("--subset", nargs="+", default=["writing_rec"])
    ap.add_argument("--split", type=str, default="test")

    # ANAH args
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
        help="Path to a pre-sampled ANAH jsonl. If set, skips HuggingFace download.",
    )

    # RAGTruth args
    ap.add_argument(
        "--ragtruth_sample_file",
        type=str,
        default="",
        help="Path to a pre-sampled RAGTruth jsonl (e.g. ragtruth_250_sample.jsonl).",
    )

    # common model args
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    ap.add_argument("--max_model_len", type=int, default=0)

    # compute / FLOPs estimate (Claimify-style)
    ap.add_argument(
        "--model_params_b",
        type=float,
        default=8.0,
        help="Model size in billions of parameters (for FLOPs estimate). Example: 8 for 8B.",
    )
    ap.add_argument(
        "--flops_per_param",
        type=float,
        default=2.0,
        help="FLOPs per parameter per token multiplier (rough). Common: 2 or 6.",
    )

    # knowledge packing
    ap.add_argument("--max_passages", type=int, default=60)
    ap.add_argument("--passage_max_chars", type=int, default=1200)
    ap.add_argument("--max_knowledge_chars", type=int, default=12000)
    ap.add_argument("--max_knowledge_passages", type=int, default=20)

    # batching
    ap.add_argument("--batch_size_atomic", type=int, default=32)
    ap.add_argument("--batch_size_small", type=int, default=64)

    # generation limits
    # Note: thinking models (e.g. Qwen3) emit a <think>…</think> block before the
    # actual answer.  strip_think_tags() in safe_utils removes that block, so the
    # parser only sees the real output.  However the <think> block still consumes
    # tokens during generation, so defaults are set high enough to leave room for
    # the actual answer after the reasoning.  Override on the CLI if needed.
    ap.add_argument("--atomic_max_tokens", type=int, default=2048)
    ap.add_argument("--revise_max_tokens", type=int, default=512)
    ap.add_argument("--relevance_max_tokens", type=int, default=256)
    ap.add_argument("--verify_max_tokens", type=int, default=512)

    # misc
    ap.add_argument("--out_root", type=str, default="./cachedir_safe_like")
    ap.add_argument(
        "--max_samples", type=int, default=0, help="FactBench only: 0 = all"
    )
    ap.add_argument(
        "--debug_n_examples", type=int, default=0, help="FELM only: 0 = all"
    )

    args = ap.parse_args()

    max_model_len = None if not args.max_model_len else int(args.max_model_len)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "felm":
        if load_from_disk is None:
            raise RuntimeError(
                "datasets is not installed, but --dataset felm requires it: pip install datasets"
            )
        if not args.felm_dir:
            raise ValueError("--felm_dir is required for --dataset felm")

    if args.dataset == "anah":
        try:
            from datasets import load_dataset as _load_dataset  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError(
                "datasets is not installed, but --dataset anah requires it: pip install datasets"
            ) from exc

    with vllm_session(
        args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
    ) as llm:
        chat = VLLMChat(args.model, llm)

        if args.dataset == "factbench":
            if not args.data:
                raise ValueError("--data is required for --dataset factbench")

            # Load JSONL
            samples: List[Dict[str, Any]] = []
            with open(args.data, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if args.max_samples and i >= args.max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    samples.append(json.loads(line))

            segment_rows_all: List[Dict[str, Any]] = []
            example_rows: List[Dict[str, Any]] = []

            # Global accumulators
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

            y_true_all: List[int] = []
            y_score_all: List[float] = []
            y_true_eval: List[int] = []
            y_score_eval: List[float] = []

            # Build items and run
            for i, ex in enumerate(samples):
                prompt = (ex.get("prompt") or "").strip()
                base_topic = topic_from_prompt(prompt)
                topic = f"{i}::{base_topic}"

                sent_items = iter_sentences_in_order(ex.get("sentences") or {})

                sent_keys: List[str] = []
                sent_texts: List[str] = []
                gold_supported_list: List[Optional[bool]] = []

                # full answer is concat of sentence texts
                tmp_full = []
                for sent_key, s in sent_items:
                    sent_keys.append(sent_key)
                    # Follow the Claimify paper's SAFE setup: decomposition runs on
                    # the original sentence, while decontextualization gets the full answer.
                    t = (s.get("text") or s.get("decontext") or "").strip()
                    sent_texts.append(t)
                    if t:
                        tmp_full.append(t)
                    gold_supported_list.append(
                        norm_gold_label(s.get("sentence_factuality_label"))
                    )

                full_answer = " ".join(tmp_full).strip()

                # evidence pooling across all sentences
                ev_chunks: List[str] = []
                for _, s in sent_items:
                    ev_chunks += flatten_evidence(s.get("auto_evidence"))
                    ev_chunks += flatten_evidence(s.get("human_evidence"))

                context_text = "\n\n".join([c for c in ev_chunks if str(c).strip()])
                passages = ref_text_to_passages(
                    base_topic,
                    context_text,
                    max_passages=args.max_passages,
                    max_chars=args.passage_max_chars,
                    felm_clean=False,
                )
                knowledge = passages_to_knowledge(
                    passages,
                    max_chars=args.max_knowledge_chars,
                    max_passages=args.max_knowledge_passages,
                )

                res = run_pipeline_on_sentences(
                    chat=chat,
                    question=prompt,
                    full_answer=full_answer,
                    knowledge=knowledge,
                    sent_keys=sent_keys,
                    sent_texts=sent_texts,
                    gold_supported_list=gold_supported_list,
                    model_name=args.model,
                    args=args,
                    is_factbench=True,
                )

                # add identifiers to segment rows
                for r in res["segment_rows"]:
                    r.update({"sample_id": i, "topic": topic, "prompt": prompt})
                    # finalize row totals + add FLOPs
                    finalize_row_times_and_tokens(r)
                    add_flops(r, args.model_params_b, args.flops_per_param)
                segment_rows_all.extend(res["segment_rows"])

                ex_out = dict(ex)
                ex_out["safe_like_topic"] = topic
                ex_out["safe_like_base_topic"] = base_topic
                ex_out["safe_like_pred_3class"] = res["pred3_list"]
                ex_out["safe_like_pred_binary"] = res["pred2_list"]
                ex_out["safe_like_fail_reason"] = res["fail_list"]
                ex_out["safe_like_respond_ratio"] = res["rr_list"]
                ex_out["safe_like_has_context"] = res["has_context_list"]
                ex_out["safe_like_model_used"] = args.model
                example_rows.append(ex_out)

                # aggregate
                total_segments += res["total_segments_with_gold"]
                skipped_na += res["skipped_na"]
                cnt_has_context += res["cnt_has_context"]
                cnt_atoms += res["cnt_atoms"]
                cnt_evaluable += res["cnt_evaluable"]

                for k in fail:
                    fail[k] += res["fail"][k]

                for k in cm_all:
                    cm_all[k] += res["cm_all"][k]
                    cm_evaluable[k] += res["cm_evaluable"][k]

                correct_all += res["correct_all"]
                correct_evaluable += res["correct_evaluable"]

                y_true_all.extend(res["y_true_all"])
                y_score_all.extend(res["y_score_all"])
                y_true_eval.extend(res["y_true_eval"])
                y_score_eval.extend(res["y_score_eval"])

            coverage_context = safe_div(cnt_has_context, total_segments)
            coverage_atoms = safe_div(cnt_atoms, total_segments)
            coverage_evaluable = safe_div(cnt_evaluable, total_segments)

            m_all = cm_to_macro_f1(cm_all)
            m_eval = cm_to_macro_f1(cm_evaluable)

            acc_all = safe_div(correct_all, total_segments)
            acc_evaluable = safe_div(correct_evaluable, cnt_evaluable)

            auc_all = roc_auc_manual(y_true_all, y_score_all)
            auc_eval = roc_auc_manual(y_true_eval, y_score_eval)

            # Only rows with gold participate in efficiency/compute averages
            eval_rows = [
                r for r in segment_rows_all if r.get("gold_supported") is not None
            ]
            _sum_ext_s = _sum_time(eval_rows, "extract_s")
            _sum_ver_s = _sum_time(eval_rows, "verify_s")
            _sum_tot_s = _sum_time(eval_rows, "total_s")
            metrics = {
                "dataset": "factcheck-GPT-benchmark (FactBench)",
                "model_used": args.model,
                "total_samples": len(samples),
                "total_segments_with_gold": total_segments,
                "skipped_segments_gold_NA": skipped_na,
                "coverage_context": coverage_context,
                "coverage_atoms": coverage_atoms,
                "coverage_evaluable": coverage_evaluable,
                "fail_breakdown": fail,
                # Canonical sentence-level metric blocks (mirrors Claimify)
                "all_sentences": {"n": total_segments, **m_all},
                "all_sentences_evaluable": {"n": cnt_evaluable, **m_eval},
                "roc_auc_not_supported": auc_all,
                "roc_auc_not_supported_evaluable": auc_eval,
                "n_scored_for_auc": len(y_true_all),
                "n_scored_for_auc_evaluable": len(y_true_eval),
                "efficiency": {
                    "n_eval_rows": total_segments,
                    "avg_extract_time_s_per_sentence": safe_div(_sum_ext_s, total_segments),
                    "avg_verify_time_s_per_sentence": safe_div(_sum_ver_s, total_segments),
                    "avg_total_time_s_per_sentence": safe_div(_sum_tot_s, total_segments),
                    "sum_extract_time_s": _sum_ext_s,
                    "sum_verify_time_s": _sum_ver_s,
                    "sum_total_time_s": _sum_tot_s,
                },
                "compute": (
                    None
                    if args.model_params_b <= 0
                    else {
                        "params_b": args.model_params_b,
                        # FLOPs = flops_per_param * params * tokens (vLLM token ids per stage)
                        # extract = atomic + revise + relevance; verify = verification calls
                        "flops_per_param": args.flops_per_param,
                        "sum_extract_flops": _sum_flops(eval_rows, "extract_flops"),
                        "sum_verify_flops": _sum_flops(eval_rows, "verify_flops"),
                        "sum_total_flops": _sum_flops(eval_rows, "total_flops"),
                        "extract_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "extract_flops"), _sum_ext_s
                        ),
                        "verify_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "verify_flops"), _sum_ver_s
                        ),
                        "total_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "total_flops"), _sum_tot_s
                        ),
                        "sum_extract_time_s": _sum_ext_s,
                        "sum_verify_time_s": _sum_ver_s,
                        "sum_total_time_s": _sum_tot_s,
                    }
                ),
            }

            out_dir = out_root / "factbench"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json(metrics, out_dir / "metrics.json")
            save_jsonl(segment_rows_all, out_dir / "segments_with_safe_like.jsonl")
            save_jsonl(example_rows, out_dir / "examples_with_safe_like.jsonl")

            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            print(f"\nSaved to: {out_dir.resolve()}")

        elif args.dataset == "felm":
            # FELM
            felm_dir = Path(args.felm_dir)

            for subset in args.subset:
                ds_path = felm_dir / subset
                if not ds_path.exists():
                    raise FileNotFoundError(f"Subset folder not found: {ds_path}")

                ds = load_from_disk(str(ds_path))[args.split]
                if args.debug_n_examples and args.debug_n_examples > 0:
                    ds = ds.select(range(min(len(ds), args.debug_n_examples)))

                subset_out = out_root / "felm" / subset / args.split
                subset_out.mkdir(parents=True, exist_ok=True)

                segment_rows_all: List[Dict[str, Any]] = []
                example_rows: List[Dict[str, Any]] = []

                # Global accumulators
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

                # AUC arrays — needed for canonical roc_auc_not_supported output
                y_true_all: List[int] = []
                y_score_all: List[float] = []
                y_true_eval: List[int] = []
                y_score_eval: List[float] = []

                # build and run
                for ex in ds:
                    ex_idx = int(ex["index"])
                    question = (ex.get("prompt") or "").strip()

                    raw_segs = ex.get("segmented_response") or []
                    raw_labs = ex.get("labels") or []
                    segs, labs = merge_list_markers_with_labels(raw_segs, raw_labs)

                    sent_keys = [str(i) for i in range(len(segs))]
                    sent_texts = [clean_seg(s) for s in segs]
                    gold_supported_list = [norm_gold_label_felm(x) for x in labs]

                    full_answer = " ".join([t for t in sent_texts if t]).strip()

                    base_topic = topic_from_ref_or_prompt(
                        ex.get("prompt"), ex.get("ref") or []
                    )
                    topic = f"{base_topic}::{ex_idx}"

                    passages = ref_text_to_passages(
                        base_topic,
                        ex.get("ref_text") or "",
                        max_passages=args.max_passages,
                        max_chars=args.passage_max_chars,
                        felm_clean=True,
                    )
                    knowledge = passages_to_knowledge(
                        passages,
                        max_chars=args.max_knowledge_chars,
                        max_passages=args.max_knowledge_passages,
                    )

                    res = run_pipeline_on_sentences(
                        chat=chat,
                        question=question,
                        full_answer=full_answer,
                        knowledge=knowledge,
                        sent_keys=sent_keys,
                        sent_texts=sent_texts,
                        gold_supported_list=gold_supported_list,
                        model_name=args.model,
                        args=args,
                        is_factbench=False,
                    )

                    # tag segment rows
                    for r in res["segment_rows"]:
                        r.update(
                            {
                                "subset": subset,
                                "split": args.split,
                                "example_index": ex_idx,
                                "topic": topic,
                            }
                        )
                        finalize_row_times_and_tokens(r)
                        add_flops(r, args.model_params_b, args.flops_per_param)
                    segment_rows_all.extend(res["segment_rows"])

                    ex_out = dict(ex)
                    ex_out["safe_like_topic"] = topic
                    ex_out["safe_like_pred_3class"] = res["pred3_list"]
                    ex_out["safe_like_pred_binary"] = res["pred2_list"]
                    ex_out["safe_like_fail_reason"] = res["fail_list"]
                    ex_out["safe_like_respond_ratio"] = res["rr_list"]
                    ex_out["safe_like_has_context"] = res["has_context_list"]
                    ex_out["safe_like_model_used"] = args.model
                    example_rows.append(ex_out)

                    # aggregate
                    total_segments += res["total_segments_with_gold"]
                    cnt_has_context += res["cnt_has_context"]
                    cnt_atoms += res["cnt_atoms"]
                    cnt_evaluable += res["cnt_evaluable"]

                    for k in fail:
                        fail[k] += res["fail"][k]

                    for k in cm_all:
                        cm_all[k] += res["cm_all"][k]
                        cm_evaluable[k] += res["cm_evaluable"][k]

                    correct_all += res["correct_all"]
                    correct_evaluable += res["correct_evaluable"]

                    # Accumulate AUC arrays from pipeline result (FELM is_factbench=False,
                    # so run_pipeline_on_sentences does NOT populate y_true/y_score —
                    # we build them here from segment rows directly)
                    for r in res["segment_rows"]:
                        gs = r.get("gold_supported")
                        if gs is None:
                            continue
                        y_true_val = 0 if gs else 1
                        risk = r.get("risk_not_supported")
                        fail_r = r.get("fail_reason") or ""
                        if fail_r:
                            y_true_all.append(y_true_val)
                            y_score_all.append(1.0)
                        elif risk is not None:
                            y_true_all.append(y_true_val)
                            y_score_all.append(float(risk))
                            pred3 = r.get("pred_3class", "ir")
                            if pred3 in {"supported", "not_supported"}:
                                y_true_eval.append(y_true_val)
                                y_score_eval.append(float(risk))
                        else:
                            # ir / no risk: worst-case for all, skip for eval
                            y_true_all.append(y_true_val)
                            y_score_all.append(1.0)

                coverage_context = safe_div(cnt_has_context, total_segments)
                coverage_atoms = safe_div(cnt_atoms, total_segments)
                coverage_evaluable = safe_div(cnt_evaluable, total_segments)

                m_all = cm_to_macro_f1(cm_all)
                m_eval = cm_to_macro_f1(cm_evaluable)

                auc_all = roc_auc_manual(y_true_all, y_score_all)
                auc_eval = roc_auc_manual(y_true_eval, y_score_eval)

                # Only rows with gold participate in efficiency/compute averages
                eval_rows = [
                    r for r in segment_rows_all if r.get("gold_supported") is not None
                ]
                _sum_ext_s = _sum_time(eval_rows, "extract_s")
                _sum_ver_s = _sum_time(eval_rows, "verify_s")
                _sum_tot_s = _sum_time(eval_rows, "total_s")
                metrics = {
                    "dataset": "FELM (offline ref_text)",
                    "subset": subset,
                    "split": args.split,
                    "model_used": args.model,
                    "total_examples": len(ds),
                    "total_segments": total_segments,
                    "coverage_context": coverage_context,
                    "coverage_atoms": coverage_atoms,
                    "coverage_evaluable": coverage_evaluable,
                    "fail_breakdown": fail,
                    # Canonical sentence-level metric blocks
                    "all_sentences": {"n": total_segments, **m_all},
                    "all_sentences_evaluable": {"n": cnt_evaluable, **m_eval},
                    "roc_auc_not_supported": auc_all,
                    "roc_auc_not_supported_evaluable": auc_eval,
                    "n_scored_for_auc": len(y_true_all),
                    "n_scored_for_auc_evaluable": len(y_true_eval),
                    "efficiency": {
                        "n_eval_rows": total_segments,
                        "avg_extract_time_s_per_sentence": safe_div(_sum_ext_s, total_segments),
                        "avg_verify_time_s_per_sentence": safe_div(_sum_ver_s, total_segments),
                        "avg_total_time_s_per_sentence": safe_div(_sum_tot_s, total_segments),
                        "sum_extract_time_s": _sum_ext_s,
                        "sum_verify_time_s": _sum_ver_s,
                        "sum_total_time_s": _sum_tot_s,
                    },
                    "compute": (
                        None
                        if args.model_params_b <= 0
                        else {
                            "params_b": args.model_params_b,
                            # FLOPs = flops_per_param * params * tokens (vLLM token ids per stage)
                            "flops_per_param": args.flops_per_param,
                            "sum_extract_flops": _sum_flops(eval_rows, "extract_flops"),
                            "sum_verify_flops": _sum_flops(eval_rows, "verify_flops"),
                            "sum_total_flops": _sum_flops(eval_rows, "total_flops"),
                            "extract_tflops_per_s_agg": _safe_tflops_per_s(
                                _sum_flops(eval_rows, "extract_flops"), _sum_ext_s
                            ),
                            "verify_tflops_per_s_agg": _safe_tflops_per_s(
                                _sum_flops(eval_rows, "verify_flops"), _sum_ver_s
                            ),
                            "total_tflops_per_s_agg": _safe_tflops_per_s(
                                _sum_flops(eval_rows, "total_flops"), _sum_tot_s
                            ),
                            "sum_extract_time_s": _sum_ext_s,
                            "sum_verify_time_s": _sum_ver_s,
                            "sum_total_time_s": _sum_tot_s,
                        }
                    ),
                }

                save_json(metrics, subset_out / "metrics.json")
                save_jsonl(
                    segment_rows_all, subset_out / "segments_with_safe_like.jsonl"
                )
                save_jsonl(example_rows, subset_out / "examples_with_safe_like.jsonl")

                print(json.dumps(metrics, indent=2, ensure_ascii=False))
                print(f"\nSaved to: {subset_out.resolve()}")

        elif args.dataset == "ragtruth":
            import sys as _sys, os as _os  # noqa: PLC0415
            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
            from ragtruth_utils import load_ragtruth_rows  # noqa: PLC0415

            if not args.ragtruth_sample_file:
                raise ValueError("--ragtruth_sample_file is required for --dataset ragtruth")

            ragtruth_rows, n_examples_total = load_ragtruth_rows(args.ragtruth_sample_file)
            split_label = _os.path.splitext(_os.path.basename(args.ragtruth_sample_file))[0]

            ragtruth_out = out_root / "ragtruth" / split_label
            ragtruth_out.mkdir(parents=True, exist_ok=True)

            segment_rows_all: List[Dict[str, Any]] = []

            # Global accumulators
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

            y_true_all: List[int] = []
            y_score_all: List[float] = []
            y_true_eval: List[int] = []
            y_score_eval: List[float] = []

            for sent_row in ragtruth_rows:
                gold_supported: Optional[bool] = sent_row["gold_supported"]
                if gold_supported is None:
                    continue

                question = clean_seg(sent_row["question"])
                sentence = clean_seg(sent_row["sentence"])
                hallucination_type = sent_row["hallucination_type"]
                ex_i = sent_row["example_index"]
                reference = sent_row["context"]
                full_answer = clean_seg(sent_row.get("answer_text") or "")
                if not full_answer:
                    full_answer = " ".join(
                        clean_seg(s)
                        for s in (sent_row.get("answer_sentences") or [])
                        if clean_seg(s)
                    ).strip()
                if not full_answer:
                    full_answer = sentence

                topic = question[:80] or f"ragtruth_{ex_i}_{sent_row['sentence_index']}"
                passages = ref_text_to_passages(
                    topic,
                    reference,
                    max_passages=args.max_passages,
                    max_chars=args.passage_max_chars,
                    felm_clean=False,
                )
                knowledge = passages_to_knowledge(
                    passages,
                    max_chars=args.max_knowledge_chars,
                    max_passages=args.max_knowledge_passages,
                )

                res = run_pipeline_on_sentences(
                    chat=chat,
                    question=question,
                    full_answer=full_answer,
                    knowledge=knowledge,
                    sent_keys=["sentence0"],
                    sent_texts=[sentence],
                    gold_supported_list=[gold_supported],
                    model_name=args.model,
                    args=args,
                    is_factbench=False,
                )

                for r in res["segment_rows"]:
                    r.update(
                        {
                            "example_index": ex_i,
                            "row_id": sent_row.get("row_id", ""),
                            "task_type": sent_row.get("task_type", ""),
                            "sentence_index": sent_row["sentence_index"],
                            "hallucination_type": hallucination_type,
                            "sample_file": args.ragtruth_sample_file,
                        }
                    )
                    finalize_row_times_and_tokens(r)
                    add_flops(r, args.model_params_b, args.flops_per_param)
                segment_rows_all.extend(res["segment_rows"])

                total_segments += res["total_segments_with_gold"]
                cnt_has_context += res["cnt_has_context"]
                cnt_atoms += res["cnt_atoms"]
                cnt_evaluable += res["cnt_evaluable"]

                for k in fail:
                    fail[k] += res["fail"][k]
                for k in cm_all:
                    cm_all[k] += res["cm_all"][k]
                    cm_evaluable[k] += res["cm_evaluable"][k]
                correct_all += res["correct_all"]
                correct_evaluable += res["correct_evaluable"]
                # Accumulate AUC arrays from each sentence's segment row
                for r in res["segment_rows"]:
                    gs = r.get("gold_supported")
                    if gs is None:
                        continue
                    y_true = 0 if gs else 1
                    risk = r.get("risk_not_supported")
                    fail_r = r.get("fail_reason") or ""
                    if fail_r:
                        y_true_all.append(y_true)
                        y_score_all.append(1.0)
                    elif risk is not None:
                        y_true_all.append(y_true)
                        y_score_all.append(float(risk))
                        pred3 = r.get("pred_3class", "ir")
                        if pred3 in {"supported", "not_supported"}:
                            y_true_eval.append(y_true)
                            y_score_eval.append(float(risk))
                    else:
                        y_true_all.append(y_true)
                        y_score_all.append(1.0)

            coverage_context = safe_div(cnt_has_context, total_segments)
            coverage_atoms = safe_div(cnt_atoms, total_segments)
            coverage_evaluable = safe_div(cnt_evaluable, total_segments)

            m_all = cm_to_macro_f1(cm_all)
            m_eval = cm_to_macro_f1(cm_evaluable)

            acc_all = safe_div(correct_all, total_segments)
            acc_evaluable = safe_div(correct_evaluable, cnt_evaluable)

            auc_all = roc_auc_manual(y_true_all, y_score_all)
            auc_eval = roc_auc_manual(y_true_eval, y_score_eval)

            eval_rows = [
                r for r in segment_rows_all if r.get("gold_supported") is not None
            ]
            _sum_ext_s = _sum_time(eval_rows, "extract_s")
            _sum_ver_s = _sum_time(eval_rows, "verify_s")
            _sum_tot_s = _sum_time(eval_rows, "total_s")
            metrics = {
                "dataset": "RAGTruth (wandb/RAGTruth-processed)",
                "sample_file": args.ragtruth_sample_file,
                "model_used": args.model,
                "total_sentences": total_segments,
                "coverage_context": coverage_context,
                "coverage_atoms": coverage_atoms,
                "coverage_evaluable": coverage_evaluable,
                "fail_breakdown": fail,
                # Canonical sentence-level metric blocks
                "all_sentences": {"n": total_segments, **m_all},
                "all_sentences_evaluable": {"n": cnt_evaluable, **m_eval},
                "roc_auc_not_supported": auc_all,
                "roc_auc_not_supported_evaluable": auc_eval,
                "n_scored_for_auc": len(y_true_all),
                "n_scored_for_auc_evaluable": len(y_true_eval),
                "efficiency": {
                    "n_eval_rows": total_segments,
                    "avg_extract_time_s_per_sentence": safe_div(_sum_ext_s, total_segments),
                    "avg_verify_time_s_per_sentence": safe_div(_sum_ver_s, total_segments),
                    "avg_total_time_s_per_sentence": safe_div(_sum_tot_s, total_segments),
                    "sum_extract_time_s": _sum_ext_s,
                    "sum_verify_time_s": _sum_ver_s,
                    "sum_total_time_s": _sum_tot_s,
                },
                "compute": (
                    None
                    if args.model_params_b <= 0
                    else {
                        "params_b": args.model_params_b,
                        # FLOPs = flops_per_param * params * tokens (vLLM token ids per stage)
                        "flops_per_param": args.flops_per_param,
                        "sum_extract_flops": _sum_flops(eval_rows, "extract_flops"),
                        "sum_verify_flops": _sum_flops(eval_rows, "verify_flops"),
                        "sum_total_flops": _sum_flops(eval_rows, "total_flops"),
                        "extract_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "extract_flops"), _sum_ext_s
                        ),
                        "verify_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "verify_flops"), _sum_ver_s
                        ),
                        "total_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "total_flops"), _sum_tot_s
                        ),
                        "sum_extract_time_s": _sum_ext_s,
                        "sum_verify_time_s": _sum_ver_s,
                        "sum_total_time_s": _sum_tot_s,
                    }
                ),
            }

            save_json(metrics, ragtruth_out / "metrics.json")
            save_jsonl(segment_rows_all, ragtruth_out / "segments_with_safe_like.jsonl")

            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            print(f"\nSaved to: {ragtruth_out.resolve()}")

        else:  # anah
            import sys as _sys, os as _os  # noqa: PLC0415
            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
            from anah_utils import iter_anah_sentences  # noqa: PLC0415

            if args.anah_sample_file:
                with open(args.anah_sample_file, encoding="utf-8") as _f:
                    _sample_rows = [json.loads(l) for l in _f if l.strip()]
                anah_iter = iter(_sample_rows)
                n_examples_total = len(_sample_rows)
                split_label = _os.path.splitext(_os.path.basename(args.anah_sample_file))[0]
            else:
                from datasets import load_dataset as _load_dataset  # noqa: PLC0415
                ds = _load_dataset("opencompass/anah", split=args.anah_split)
                anah_iter = iter_anah_sentences(ds, max_examples=args.anah_max_examples or 0)
                n_examples_total = len(ds)
                split_label = args.anah_split

            anah_out = out_root / "anah" / split_label
            anah_out.mkdir(parents=True, exist_ok=True)

            segment_rows_all: List[Dict[str, Any]] = []

            # Global accumulators
            total_segments = 0
            skipped_no_fact = 0
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

            y_true_all: List[int] = []
            y_score_all: List[float] = []
            y_true_eval: List[int] = []
            y_score_eval: List[float] = []

            for sent_row in anah_iter:
                gold_supported: Optional[bool] = sent_row["gold_supported"]
                if gold_supported is None:
                    skipped_no_fact += 1
                    continue  # No Fact – skip entirely

                question = clean_seg(sent_row["question"])
                sentence = clean_seg(sent_row["sentence"])
                hallucination_type = sent_row["hallucination_type"]
                ex_i = sent_row["example_index"]
                # Use ann_reference (specific cited fragment) as per-sentence evidence
                reference = sent_row["ann_reference"]
                full_answer = clean_seg(sent_row.get("answer_text") or "")
                if not full_answer:
                    full_answer = " ".join(
                        clean_seg(s)
                        for s in (sent_row.get("answer_sentences") or [])
                        if clean_seg(s)
                    ).strip()
                if not full_answer:
                    full_answer = sentence

                topic = question[:80] or f"anah_{ex_i}_{sent_row['sentence_index']}"
                passages = ref_text_to_passages(
                    topic,
                    reference,
                    max_passages=args.max_passages,
                    max_chars=args.passage_max_chars,
                    felm_clean=False,
                )
                knowledge = passages_to_knowledge(
                    passages,
                    max_chars=args.max_knowledge_chars,
                    max_passages=args.max_knowledge_passages,
                )

                # ANAH rows are individual sentences – wrap as single-sentence list
                res = run_pipeline_on_sentences(
                    chat=chat,
                    question=question,
                    full_answer=full_answer,
                    knowledge=knowledge,
                    sent_keys=["sentence0"],
                    sent_texts=[sentence],
                    gold_supported_list=[gold_supported],
                    model_name=args.model,
                    args=args,
                    is_factbench=False,
                )

                for r in res["segment_rows"]:
                    r.update(
                        {
                            "example_index": ex_i,
                            "answer_index": sent_row["answer_index"],
                            "sentence_index": sent_row["sentence_index"],
                            "hallucination_type": hallucination_type,
                            "split": args.anah_split,
                        }
                    )
                    finalize_row_times_and_tokens(r)
                    add_flops(r, args.model_params_b, args.flops_per_param)
                segment_rows_all.extend(res["segment_rows"])

                total_segments += res["total_segments_with_gold"]
                cnt_has_context += res["cnt_has_context"]
                cnt_atoms += res["cnt_atoms"]
                cnt_evaluable += res["cnt_evaluable"]

                for k in fail:
                    fail[k] += res["fail"][k]
                for k in cm_all:
                    cm_all[k] += res["cm_all"][k]
                    cm_evaluable[k] += res["cm_evaluable"][k]
                correct_all += res["correct_all"]
                correct_evaluable += res["correct_evaluable"]
                # Accumulate AUC arrays from each sentence's segment row
                for r in res["segment_rows"]:
                    gs = r.get("gold_supported")
                    if gs is None:
                        continue
                    y_true = 0 if gs else 1
                    risk = r.get("risk_not_supported")
                    fail_r = r.get("fail_reason") or ""
                    if fail_r:
                        y_true_all.append(y_true)
                        y_score_all.append(1.0)
                    elif risk is not None:
                        y_true_all.append(y_true)
                        y_score_all.append(float(risk))
                        pred3 = r.get("pred_3class", "ir")
                        if pred3 in {"supported", "not_supported"}:
                            y_true_eval.append(y_true)
                            y_score_eval.append(float(risk))
                    else:
                        y_true_all.append(y_true)
                        y_score_all.append(1.0)

            coverage_context = safe_div(cnt_has_context, total_segments)
            coverage_atoms = safe_div(cnt_atoms, total_segments)
            coverage_evaluable = safe_div(cnt_evaluable, total_segments)

            m_all = cm_to_macro_f1(cm_all)
            m_eval = cm_to_macro_f1(cm_evaluable)

            acc_all = safe_div(correct_all, total_segments)
            acc_evaluable = safe_div(correct_evaluable, cnt_evaluable)

            auc_all = roc_auc_manual(y_true_all, y_score_all)
            auc_eval = roc_auc_manual(y_true_eval, y_score_eval)

            eval_rows = [
                r for r in segment_rows_all if r.get("gold_supported") is not None
            ]
            _sum_ext_s = _sum_time(eval_rows, "extract_s")
            _sum_ver_s = _sum_time(eval_rows, "verify_s")
            _sum_tot_s = _sum_time(eval_rows, "total_s")
            metrics = {
                "dataset": "ANAH (opencompass/anah)",
                "split": split_label,
                "model_used": args.model,
                "total_examples_in_split": n_examples_total,
                "skipped_no_fact": skipped_no_fact,
                "total_segments": total_segments,
                "coverage_context": coverage_context,
                "coverage_atoms": coverage_atoms,
                "coverage_evaluable": coverage_evaluable,
                "fail_breakdown": fail,
                # Canonical sentence-level metric blocks
                "all_sentences": {"n": total_segments, **m_all},
                "all_sentences_evaluable": {"n": cnt_evaluable, **m_eval},
                "roc_auc_not_supported": auc_all,
                "roc_auc_not_supported_evaluable": auc_eval,
                "n_scored_for_auc": len(y_true_all),
                "n_scored_for_auc_evaluable": len(y_true_eval),
                "efficiency": {
                    "n_eval_rows": total_segments,
                    "avg_extract_time_s_per_sentence": safe_div(_sum_ext_s, total_segments),
                    "avg_verify_time_s_per_sentence": safe_div(_sum_ver_s, total_segments),
                    "avg_total_time_s_per_sentence": safe_div(_sum_tot_s, total_segments),
                    "sum_extract_time_s": _sum_ext_s,
                    "sum_verify_time_s": _sum_ver_s,
                    "sum_total_time_s": _sum_tot_s,
                },
                "compute": (
                    None
                    if args.model_params_b <= 0
                    else {
                        "params_b": args.model_params_b,
                        # FLOPs = flops_per_param * params * tokens (vLLM token ids per stage)
                        "flops_per_param": args.flops_per_param,
                        "sum_extract_flops": _sum_flops(eval_rows, "extract_flops"),
                        "sum_verify_flops": _sum_flops(eval_rows, "verify_flops"),
                        "sum_total_flops": _sum_flops(eval_rows, "total_flops"),
                        "extract_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "extract_flops"), _sum_ext_s
                        ),
                        "verify_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "verify_flops"), _sum_ver_s
                        ),
                        "total_tflops_per_s_agg": _safe_tflops_per_s(
                            _sum_flops(eval_rows, "total_flops"), _sum_tot_s
                        ),
                        "sum_extract_time_s": _sum_ext_s,
                        "sum_verify_time_s": _sum_ver_s,
                        "sum_total_time_s": _sum_tot_s,
                    }
                ),
            }

            save_json(metrics, anah_out / "metrics.json")
            save_jsonl(segment_rows_all, anah_out / "segments_with_safe_like.jsonl")

            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            print(f"\nSaved to: {anah_out.resolve()}")


if __name__ == "__main__":
    main()
