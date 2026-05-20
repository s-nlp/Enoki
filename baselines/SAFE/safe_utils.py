import gc
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# Evidence / passages
def topic_from_prompt(prompt: str) -> str:
    return (prompt or "").strip()[:120] or "unknown_topic"


def topic_from_ref_or_prompt(prompt: str, ref: List[str]) -> str:
    """FELM helper: use wiki title if available."""
    from urllib.parse import unquote

    if ref and len(ref) and isinstance(ref[0], str) and "wikipedia.org/wiki/" in ref[0]:
        return unquote(ref[0].split("/wiki/")[-1]).replace("_", " ")
    return (prompt or "").strip()[:120] or "unknown_topic"


def flatten_evidence(x: Any) -> List[str]:
    """
    FactBench fields:
      - auto_evidence: usually List[List[str]] (nested)
      - auto_evidence_url: usually List[List[str]] (nested)
      - human_evidence: List[str]
    Accept any nesting and stringify non-strings safely.
    """
    out: List[str] = []

    def rec(v):
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif isinstance(v, (list, tuple)):
            for t in v:
                rec(t)
        elif isinstance(v, dict):
            out.append(json.dumps(v, ensure_ascii=False))
        else:
            out.append(str(v))

    rec(x)
    return out


def ref_text_to_passages(
    topic: str,
    ref_text: Any,
    max_passages: int = 60,
    max_chars: int = 1200,
    felm_clean: bool = False,
) -> List[Dict[str, str]]:
    """
    - FactBench: ref_text is typically one big string (we split by newlines).
    - FELM: ref_text may be list or contains <s> </s> markers (optional felm_clean).
    """
    if ref_text is None:
        return []
    if isinstance(ref_text, list):
        ref_text = "\n".join([str(x) for x in ref_text])

    txt = str(ref_text)
    if not txt.strip():
        return []

    if felm_clean:
        txt = txt.replace("<s>", " ").replace("</s>", "\n")

    parts = [p.strip() for p in txt.split("\n") if p.strip()]

    passages: List[Dict[str, str]] = []
    buf = ""

    for p in parts:
        if not felm_clean:
            # FactBench/ANAH-style: keep paragraph boundaries, but never drop
            # long blocks; chunk them deterministically by max_chars.
            if len(p) <= max_chars:
                passages.append({"title": topic, "text": p})
            else:
                for i in range(0, len(p), max_chars):
                    chunk = p[i : i + max_chars].strip()
                    if chunk:
                        passages.append({"title": topic, "text": chunk})
                    if len(passages) >= max_passages:
                        break
        else:
            # FELM-style packing (also ok for long noisy refs)
            if len(buf) + 1 + len(p) <= max_chars:
                buf = (buf + " " + p).strip()
            else:
                if buf.strip():
                    passages.append({"title": topic, "text": buf})
                buf = p
        if len(passages) >= max_passages:
            break

    if felm_clean and buf.strip() and len(passages) < max_passages:
        passages.append({"title": topic, "text": buf})

    return passages[:max_passages]


def passages_to_knowledge(
    passages: List[Dict[str, str]], max_chars: int = 12000, max_passages: int = 20
) -> str:
    if not passages:
        return ""
    buf = []
    for p in passages[:max_passages]:
        title = (p.get("title") or "").strip()
        txt = (p.get("text") or "").strip()
        if not txt:
            continue
        if title:
            buf.append(f"[{title}]\n{txt}")
        else:
            buf.append(txt)
    s = "\n\n".join(buf)
    return s[:max_chars]


# -------------------------
# FactBench sentence iteration + gold
# -------------------------

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


def norm_gold_label(x) -> Optional[bool]:
    # dataset is bool or "NA"
    if x is True or x is False:
        return bool(x)
    if x is None:
        return None
    if isinstance(x, str) and x.strip().upper() == "NA":
        return None
    return None


# -------------------------
# FELM segment hygiene
# -------------------------

NUM_MARK_RE = re.compile(r"^\s*\d+\s*[\.\)]?\s*$")


def clean_seg(seg: str) -> str:
    seg = (seg or "").strip()
    seg = seg.replace('\\"', '"').strip()
    return seg


def merge_list_markers_with_labels(
    segs: List[str], labs: List[Any]
) -> Tuple[List[str], List[Any]]:
    """
    Merge pure numeric list-marker segments like '1.' / '2)' into the next segment,
    and merge labels accordingly.

    Rule:
    - If seg[i] is a marker and seg[i+1] exists:
        new_seg = seg[i] + " " + seg[i+1]
        new_lab = labs[i+1]   (marker's label is dropped)
    - If marker is dangling: drop it and its label.
    """
    if len(segs) != len(labs):
        m = min(len(segs), len(labs))
        segs, labs = segs[:m], labs[:m]

    segs = [clean_seg(s) for s in segs]
    out_segs, out_labs = [], []
    i = 0
    while i < len(segs):
        s = (segs[i] or "").strip()

        if NUM_MARK_RE.match(s):
            if i + 1 < len(segs):
                nxt = (segs[i + 1] or "").strip()
                if nxt:
                    out_segs.append(f"{s} {nxt}")
                    out_labs.append(labs[i + 1])
                    i += 2
                    continue
            i += 1
            continue

        if re.search(r"\n\s*\d+\s*[\.\)]\s*$", s):
            s = re.sub(r"\n\s*(\d+\s*[\.\)])\s*$", r" \1", s)

        out_segs.append(s)
        out_labs.append(labs[i])
        i += 1

    return out_segs, out_labs


# -------------------------
# Metrics
# -------------------------


def safe_div(a, b):
    return a / b if b else 0.0


def prf1(tp, fp, fn):
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def update_confusion_not_supported_positive(
    cm, gold_supported: bool, pred_supported: bool
):
    """Positive class is not_supported."""
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


def roc_auc_manual(y_true: List[int], y_score: List[float]) -> float:
    """
    ROC-AUC using rank statistic (Mann–Whitney U), tie-safe.
    Returns 0.0 if undefined (all positives or all negatives).
    """
    assert len(y_true) == len(y_score)
    n = len(y_true)
    if n == 0:
        return 0.0
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    pairs = sorted(
        [(s, y, i) for i, (s, y) in enumerate(zip(y_score, y_true))],
        key=lambda t: t[0],
    )
    ranks = [0.0] * n
    i = 0
    rank = 1
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            _, _, orig_idx = pairs[k]
            ranks[orig_idx] = avg_rank
        rank += j - i
        i = j

    sum_ranks_pos = sum(r for r, y in zip(ranks, y_true) if y == 1)
    u_pos = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = u_pos / (n_pos * n_neg)
    return float(auc)


def not_supported_risk_from_verdicts(verdicts: List[str]) -> Optional[float]:
    """
    risk in [0,1] = fraction of relevant facts that are NOT_SUPPORTED.
    None if no verdicts.
    """
    if not verdicts:
        return None
    vals = [
        1.0 if v == "not_supported" else 0.0
        for v in verdicts
        if v in {"supported", "not_supported"}
    ]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


@dataclass
class Usage:
    prompt_tokens: int = 0
    gen_tokens: int = 0


@dataclass
class BatchGenResult:
    texts: List[str]
    usage: List[Usage]
    # per-prompt wall time (batch_wall / n_prompts)
    per_item_time_s: float = 0.0


# -------------------------
# Save helpers
# -------------------------


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# -------------------------
# vLLM wrapper + retries
# -------------------------


def is_transient_error(e: Exception) -> bool:
    msg = str(e).lower()
    transient_markers = [
        "timeout",
        "timed out",
        "temporarily",
        "try again",
        "connection",
        "broken pipe",
        "cuda error",
        "cublas",
        "nccl",
        "out of memory",
        "oom",
        "engine is dead",
        "engine stopped",
        "queue",
        "overloaded",
    ]
    return any(m in msg for m in transient_markers)


@contextmanager
def vllm_session(
    model_name: str,
    gpu_memory_utilization: float = 0.5,
    max_model_len: Optional[int] = None,
):
    llm = None
    try:
        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        yield llm
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class VLLMChat:
    def __init__(self, model: str, llm: LLM):
        self.llm = llm
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = model
        self.has_chat_template = hasattr(self.tok, "apply_chat_template")

    def render(self, system: str, user: str) -> str:
        if self.has_chat_template:
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            try:
                return self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return f"{system}\n\n{user}"
        return f"{system}\n\n{user}"

    def generate(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int,
        max_tries: int = 3,
        base_sleep: float = 1.0,
    ) -> List[str]:
        sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                outs = self.llm.generate(prompts, sp)
                res = []
                for o in outs:
                    res.append(((o.outputs[0].text if o.outputs else "") or "").strip())
                return res
            except Exception as e:
                last_err = e
                if attempt < max_tries and is_transient_error(e):
                    time.sleep(base_sleep * (2 ** (attempt - 1)))
                    continue
                raise
        raise last_err

    def generate_with_usage(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int,
        max_tries: int = 3,
        base_sleep: float = 1.0,
    ) -> BatchGenResult:
        """
        Like generate(), but returns token counts and wall-time.
        Token counting strategy:
          - prompt_tokens: vLLM request prompt_token_ids if present, else tokenizer encode length
          - gen_tokens: sum of output token_ids if present, else tokenizer encode length of text
        """
        sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                t0 = time.perf_counter()
                outs = self.llm.generate(prompts, sp)
                dt = time.perf_counter() - t0
                n = len(prompts) or 1
                per_item = dt / n

                texts: List[str] = []
                usages: List[Usage] = []

                # vLLM typically returns outputs aligned with prompts
                for i, o in enumerate(outs):
                    txt = ((o.outputs[0].text if o.outputs else "") or "").strip()
                    texts.append(txt)

                    # prompt tokens
                    pt = len(getattr(o, "prompt_token_ids", []) or [])
                    if pt == 0:
                        try:
                            pt = len(
                                self.tok.encode(prompts[i], add_special_tokens=False)
                            )
                        except Exception:
                            pt = 0

                    # gen tokens
                    gt = 0
                    if o.outputs:
                        for out1 in o.outputs:
                            gt += len(getattr(out1, "token_ids", []) or [])
                    if gt == 0:
                        try:
                            gt = len(self.tok.encode(txt, add_special_tokens=False))
                        except Exception:
                            gt = 0

                    usages.append(Usage(prompt_tokens=int(pt), gen_tokens=int(gt)))

                # In rare cases vLLM can return fewer outputs; pad to keep alignment
                while len(texts) < len(prompts):
                    texts.append("")
                    usages.append(Usage())

                return BatchGenResult(
                    texts=texts, usage=usages, per_item_time_s=float(per_item)
                )

            except Exception as e:
                last_err = e
                if attempt < max_tries and is_transient_error(e):
                    time.sleep(base_sleep * (2 ** (attempt - 1)))
                    continue
                raise
        raise last_err


# SAFE-like prompts
ATOMIC_PROMPT = """You are given ONE sentence from a model answer to a question.
Extract ALL atomic, verifiable factual claims from the sentence.

Rules:
- Use ONLY information explicitly present in the sentence.
- Each claim must be ONE short, checkable statement (subject + predicate + object).
- Break compound sentences into multiple atomic claims.
- Output ONLY a bullet list; each line must start with "- ".
- Most sentences contain at least one verifiable claim — extract it even if it seems obvious.
- Output EXACTLY "- NONE" ONLY if the sentence is purely a greeting, transition phrase, or contains zero factual content (e.g. "Let me explain.", "In conclusion,").

Question:
{question}

Sentence:
{sentence}
"""

REVISE_PROMPT = """You are given an ATOMIC FACT extracted from a longer answer.
Rewrite it to be fully self-contained and unambiguous, using the FULL ANSWER for context.

Rules:
- Replace pronouns/vague references with explicit entities from the FULL ANSWER.
- Do NOT add new information. Do NOT change meaning.
- Output ONLY the revised fact inside a markdown code block.

Question:
{question}

FULL ANSWER:
{full_answer}

ATOMIC FACT:
{fact}
"""

RELEVANCE_PROMPT = """Decide whether the FACT is relevant to answering the QUESTION, given the FULL ANSWER.

Return EXACTLY one of:
[Foo]      - the fact is relevant to the question
[Not Foo]  - the fact is not relevant

Question:
{question}

FULL ANSWER:
{full_answer}

FACT:
{fact}
"""

VERIFY_PROMPT = """Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE.
2. Determine whether the STATEMENT is supported by the KNOWLEDGE.
3. Your final answer MUST be either [Supported] or [Not Supported].

KNOWLEDGE:
{knowledge}

STATEMENT:
{statement}
"""


# -------------------------
# Parsing helpers
# -------------------------

BULLET_OR_ENUM_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)\s*(.*\S)\s*$")


def parse_atoms(text: str) -> List[str]:
    atoms = []
    for ln in (text or "").splitlines():
        m = BULLET_OR_ENUM_RE.match(ln)
        if not m:
            continue
        s = m.group(1).strip()
        s_up = s.upper().strip().rstrip(".")
        if s_up in {"NONE", "NO FACTS", "N/A", "NULL"}:
            continue
        if s in {"[]", "[ ]"}:
            continue
        atoms.append(s)

    seen = set()
    out = []
    for a in atoms:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def extract_code_block(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    fence = "```"
    if fence not in t:
        return None
    parts = t.split(fence)
    if len(parts) < 3:
        return None
    mid = parts[1]
    mid_lines = mid.splitlines()
    if len(mid_lines) >= 2 and mid_lines[0].strip().isalpha():
        content = "\n".join(mid_lines[1:]).strip()
    else:
        content = mid.strip()
    return content.strip() if content.strip() else None


def extract_first_square_brackets(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    i = s.find("[")
    j = s.find("]", i + 1) if i != -1 else -1
    if i != -1 and j != -1:
        return s[i + 1 : j].strip()
    return ""


def parse_relevance(text: str) -> bool:
    ans = extract_first_square_brackets(text).lower()
    if "not" in ans:
        return False
    if "foo" in ans:
        return True
    # SAFE-ish default
    return True


def parse_supported_not_supported(text: str) -> str:
    ans = extract_first_square_brackets(text).lower()
    if "not supported" in ans or "notsupported" in ans:
        return "not_supported"
    if "supported" in ans:
        return "supported"
    return "not_supported"


# Compute / FLOPs helpers (Claimify-style)
def finalize_row_times_and_tokens(r: Dict[str, Any]):
    # "extract" = atomic + revise + relevance
    t = r.get("timing") or {}
    t["extract_s"] = (
        float(t.get("atomic_s", 0.0) or 0.0)
        + float(t.get("revise_s", 0.0) or 0.0)
        + float(t.get("relevance_s", 0.0) or 0.0)
    )
    t["total_s"] = float(t.get("extract_s", 0.0) or 0.0) + float(
        t.get("verify_s", 0.0) or 0.0
    )
    r["timing"] = t

    tok = r.get("tokens") or {}
    tok["total_prompt"] = (
        int(tok.get("atomic_prompt", 0) or 0)
        + int(tok.get("revise_prompt", 0) or 0)
        + int(tok.get("relevance_prompt", 0) or 0)
        + int(tok.get("verify_prompt", 0) or 0)
    )
    tok["total_gen"] = (
        int(tok.get("atomic_gen", 0) or 0)
        + int(tok.get("revise_gen", 0) or 0)
        + int(tok.get("relevance_gen", 0) or 0)
        + int(tok.get("verify_gen", 0) or 0)
    )
    r["tokens"] = tok


def add_flops(r: Dict[str, Any], params_b: float, flops_per_param: float):
    """
    Rough estimate: FLOPs ≈ k * P * tokens, where:
      P = params_b * 1e9
      tokens = (prompt + gen)
    """
    if not params_b:
        r["flops"] = None
        return

    P = float(params_b) * 1e9
    k = float(flops_per_param)

    tok = r.get("tokens") or {}
    extract_tok = (
        int(tok.get("atomic_prompt", 0) or 0)
        + int(tok.get("atomic_gen", 0) or 0)
        + int(tok.get("revise_prompt", 0) or 0)
        + int(tok.get("revise_gen", 0) or 0)
        + int(tok.get("relevance_prompt", 0) or 0)
        + int(tok.get("relevance_gen", 0) or 0)
    )
    verify_tok = int(tok.get("verify_prompt", 0) or 0) + int(
        tok.get("verify_gen", 0) or 0
    )
    total_tok = int(extract_tok + verify_tok)

    def flops_for(tok_n: int) -> float:
        return k * P * float(tok_n)

    extract_flops = flops_for(extract_tok)
    verify_flops = flops_for(verify_tok)
    total_flops = flops_for(total_tok)

    t = r.get("timing") or {}
    extract_s = float(t.get("extract_s", 0.0) or 0.0)
    verify_s = float(t.get("verify_s", 0.0) or 0.0)
    total_s = float(t.get("total_s", 0.0) or 0.0)

    def tflops_per_s(flops: float, seconds: float) -> Optional[float]:
        if seconds <= 0:
            return None
        return flops / seconds / 1e12

    r["flops"] = {
        "params_b": float(params_b),
        "k": float(k),
        "extract_tokens": int(extract_tok),
        "verify_tokens": int(verify_tok),
        "total_tokens": int(total_tok),
        "extract_flops": float(extract_flops),
        "verify_flops": float(verify_flops),
        "total_flops": float(total_flops),
        "extract_tflops_per_s": tflops_per_s(extract_flops, extract_s),
        "verify_tflops_per_s": tflops_per_s(verify_flops, verify_s),
        "total_tflops_per_s": tflops_per_s(total_flops, total_s),
    }


def _sum_time(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(float((r.get("timing") or {}).get(key, 0.0) or 0.0) for r in rows)


def _sum_flops(rows: List[Dict[str, Any]], key: str) -> float:
    s = 0.0
    for r in rows:
        f = (r.get("flops") or {}).get(key, None)
        if isinstance(f, (int, float)):
            s += float(f)
    return s


def _safe_tflops_per_s(total_flops: float, total_s: float) -> Optional[float]:
    if total_s <= 0:
        return None
    return (total_flops / total_s) / 1e12
