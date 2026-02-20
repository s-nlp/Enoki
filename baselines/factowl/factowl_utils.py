import gc
import json
import re
import textwrap
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams


@dataclass
class VLLMUsageTotals:
    prompt_tokens: int = 0
    gen_tokens: int = 0
    n_generate_calls: int = 0


class VLLMUsageWrapper:
    """
    Proxy around vllm.LLM that counts prompt/gen tokens across ALL internal generate() calls.
    Works with FactOwl because it passes vllm_model directly and internally calls vllm_model.generate(...).
    """

    def __init__(self, llm: LLM):
        self._llm = llm
        self.totals = VLLMUsageTotals()

    def snapshot(self) -> VLLMUsageTotals:
        return VLLMUsageTotals(
            prompt_tokens=self.totals.prompt_tokens,
            gen_tokens=self.totals.gen_tokens,
            n_generate_calls=self.totals.n_generate_calls,
        )

    def delta(self, snap: VLLMUsageTotals) -> VLLMUsageTotals:
        return VLLMUsageTotals(
            prompt_tokens=self.totals.prompt_tokens - snap.prompt_tokens,
            gen_tokens=self.totals.gen_tokens - snap.gen_tokens,
            n_generate_calls=self.totals.n_generate_calls - snap.n_generate_calls,
        )

    def generate(self, prompts, *args, **kwargs):
        outs = self._llm.generate(prompts, *args, **kwargs)
        self.totals.n_generate_calls += 1

        # vLLM RequestOutput has prompt_token_ids and outputs[*].token_ids
        for o in outs or []:
            pt = len(getattr(o, "prompt_token_ids", []) or [])
            gt = 0
            for x in getattr(o, "outputs", []) or []:
                gt += len(getattr(x, "token_ids", []) or [])
            self.totals.prompt_tokens += int(pt)
            self.totals.gen_tokens += int(gt)
        return outs

    def __getattr__(self, name):
        # delegate everything else (llm_engine, tokenizer config, etc.)
        return getattr(self._llm, name)


def flops_from_tokens(
    total_tokens: int, params_b: float, flops_per_param: float
) -> float:
    """
    Rough FLOPs estimate:
      FLOPs ~= flops_per_param * (#params) * (#tokens)
    where params_b is in billions.
    """
    if not params_b or params_b <= 0:
        return 0.0
    P = params_b * 1e9
    return float(flops_per_param) * float(P) * float(total_tokens)


def tflops_per_s(flops: float, seconds: float):
    if seconds <= 0:
        return None
    return float(flops) / float(seconds) / 1e12


# Generic helpers
def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def update_confusion_not_supported_positive(
    cm: Dict[str, int], gold_supported: bool, pred_supported: bool
) -> None:
    """
    POSITIVE class = not_supported
    TP: gold not_supported & pred not_supported
    FP: gold supported     & pred not_supported
    FN: gold not_supported & pred supported
    TN: gold supported     & pred supported
    """
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
    # not_supported class
    p_ns, r_ns, f1_ns = prf1(cm["TP"], cm["FP"], cm["FN"])
    # supported class (swap)
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
    ROC-AUC using rank statistic (Mann–Whitney U).
    Handles ties by using average ranks.
    Returns 0.0 if AUC is undefined (all positives or all negatives).
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
        [(s, y, i) for i, (s, y) in enumerate(zip(y_score, y_true))], key=lambda t: t[0]
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


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# Evidence / topic helpers
def topic_from_prompt(prompt: str) -> str:
    return (prompt or "").strip()[:120] or "unknown_topic"


def topic_from_ref_or_prompt(prompt: str, ref: List[str]) -> str:
    from urllib.parse import unquote

    if ref and isinstance(ref[0], str) and "wikipedia.org/wiki/" in ref[0]:
        return unquote(ref[0].split("/wiki/")[-1]).replace("_", " ")
    return (prompt or "").strip()[:120] or "unknown_topic"


def flatten_evidence(x: Any) -> List[str]:
    """
    FactBench fields can be nested:
      - auto_evidence: usually List[List[str]]
      - auto_evidence_url: usually List[List[str]]
      - human_evidence: List[str]
    We accept any nesting and stringify non-strings safely.
    """
    out: List[str] = []

    def rec(v: Any) -> None:
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
    wrap_long_paragraphs: bool = True,
) -> List[Dict[str, str]]:
    """
    Unified ref_text -> list of {title,text} passages.

    - If ref_text is list -> join.
    - Replaces <s>, </s> like in FELM preprocessing.
    - Splits on newlines; then either wraps long paras or packs lines into <=max_chars chunks.
    """
    if ref_text is None:
        return []

    if isinstance(ref_text, list):
        ref_text = "\n".join([str(x) for x in ref_text])

    txt = str(ref_text).strip()
    if not txt:
        return []

    txt = txt.replace("<s>", " ").replace("</s>", "\n")
    parts = [p.strip() for p in txt.split("\n") if p.strip()]

    passages: List[Dict[str, str]] = []
    if wrap_long_paragraphs:
        for p in parts:
            for chunk in textwrap.wrap(
                p,
                width=max_chars,
                break_long_words=False,
                replace_whitespace=False,
            ):
                passages.append({"title": topic, "text": chunk})
                if len(passages) >= max_passages:
                    return passages
        return passages

    # alternative: pack lines into a buffer
    buf = ""
    for p in parts:
        if len(buf) + 1 + len(p) <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                passages.append({"title": topic, "text": buf})
            buf = p
            if len(passages) >= max_passages:
                return passages
    if buf:
        passages.append({"title": topic, "text": buf})
    return passages[:max_passages]


# vLLM session + retries
@contextmanager
def vllm_session(model_name: str, gpu_memory_utilization: float = 0.5):
    llm = None
    try:
        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        yield VLLMUsageWrapper(llm)
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass


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


def vllm_generate_with_retries(
    llm: LLM,
    prompts: List[str],
    sampling_params: Optional[SamplingParams] = None,
    use_tqdm: bool = False,
    max_tries: int = 3,
    base_sleep: float = 1.0,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_tries + 1):
        try:
            if sampling_params is None:
                return llm.generate(prompts, use_tqdm=use_tqdm), None
            return llm.generate(prompts, sampling_params, use_tqdm=use_tqdm), None
        except Exception as e:
            last_err = e
            err_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "attempt": attempt,
                "max_tries": max_tries,
            }
            if attempt < max_tries and is_transient_error(e):
                time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue
            return None, err_info

    return None, {
        "type": type(last_err).__name__ if last_err else "UnknownError",
        "message": str(last_err) if last_err else "unknown",
        "traceback": traceback.format_exc(),
        "attempt": max_tries,
        "max_tries": max_tries,
    }


def get_score_with_retries(
    fs: Any,
    topic: str,
    seg: str,
    knowledge_source: str,
    max_tries: int = 3,
    base_sleep: float = 1.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Calls FactOwl fs.get_score with retries.
    Returns: (out, err_info)
      - out: dict or None
      - err_info: None or dict with traceback
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_tries + 1):
        try:
            out = fs.get_score(
                [topic],
                [seg],
                gamma=0,
                knowledge_source=knowledge_source,
                verbose=False,
            )
            return out, None
        except Exception as e:
            last_err = e
            err_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "attempt": attempt,
                "max_tries": max_tries,
            }
            if attempt < max_tries and is_transient_error(e):
                time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue
            return None, err_info

    return None, {
        "type": type(last_err).__name__ if last_err else "UnknownError",
        "message": str(last_err) if last_err else "unknown",
        "traceback": traceback.format_exc(),
        "attempt": max_tries,
        "max_tries": max_tries,
    }


# FactOwl parsing / debug
def pred_label_3class_from_out(out: Dict[str, Any]) -> str:
    """supported / not_supported / ir (ir = no atoms/abstain)"""
    decisions = out.get("decisions") or []
    ds = [d for d in decisions if d.get("atom")]
    if not ds:
        return "ir"
    if any(d.get("is_supported") is False for d in ds):
        return "not_supported"
    return "supported"


def not_supported_risk_from_out(out: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Continuous risk score in [0,1] where higher => more likely NOT_SUPPORTED.
    None if cannot compute.
    """
    if not out:
        return None

    decisions = out.get("decisions") or []
    atom_ds = [d for d in decisions if d.get("atom")]
    if atom_ds:
        vals: List[float] = []
        for d in atom_ds:
            if "is_supported" in d and d["is_supported"] is not None:
                vals.append(0.0 if d["is_supported"] else 1.0)
        if vals:
            return float(sum(vals) / len(vals))

    # fallback: use overall score if present (assume higher score => more supported)
    if out.get("score") is not None:
        try:
            s = float(out["score"])
            s = max(0.0, min(1.0, s))
            return float(1.0 - s)
        except Exception:
            return None

    return None


_ATOM_BULLET_RE = re.compile(r"^\s*-\s*(?:-\s*)?(.*\S)\s*$")


def debug_atomic_retry(
    llm: LLM,
    seg: str,
    max_new_tokens: int = 256,
    max_tries: int = 3,
) -> Dict[str, Any]:
    """
    Second-pass debugging: ask the model to output atomic facts in EXACT '- ' bullet format.
    Returns dict with raw_output + parsed list.
    """
    prompt = (
        "Extract atomic factual statements from the sentence below.\n"
        "Rules:\n"
        "1) Use ONLY information explicitly present in the sentence.\n"
        "2) Do NOT infer unstated details.\n"
        "3) Output ONLY bullets; each line must start with '- '.\n"
        "4) If there are no factual statements, output EXACTLY: - NONE\n\n"
        f"Sentence:\n{seg}\n"
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)

    res, err = vllm_generate_with_retries(
        llm, [prompt], sampling_params=sp, max_tries=max_tries, base_sleep=1.0
    )
    if err is not None:
        return {"ok": False, "error": err, "raw_output": None, "parsed_atoms": []}

    raw = res[0].outputs[0].text if res and res[0].outputs else ""

    atoms: List[str] = []
    for ln in raw.splitlines():
        m = _ATOM_BULLET_RE.match(ln)
        if not m:
            continue
        a = m.group(1).strip()
        if a.upper() == "NONE":
            continue
        atoms.append(a)

    atoms_norm = [a for a in atoms if a and a.upper() != "NONE"]
    return {"ok": True, "raw_output": raw, "parsed_atoms": atoms_norm}


def clean_seg(seg: str) -> str:
    seg = (seg or "").strip()
    seg = seg.replace('\\"', '"').strip()
    seg = seg.strip(' "\u201c\u201d')
    return seg


def apply_fail_policy(gold_supported: bool, policy: str) -> bool:
    """
    Return forced_pred_supported for FAIL cases.
    policy:
      - "not_supported": always predict not_supported (pred_supported=False)
      - "supported":     always predict supported (pred_supported=True)
      - "opposite":      always wrong vs gold (pred_supported = not gold_supported)
      - "same":          copy gold (pred_supported = gold_supported)
    """
    policy = (policy or "").lower()
    if policy == "not_supported":
        return False
    if policy == "supported":
        return True
    if policy == "opposite":
        return not gold_supported
    if policy == "same":
        return gold_supported
    # default fallback
    return False


# Patch FactOwl atomic extractor
def patch_factowl_atomic_extractor(
    template: str = "sentence",
    set_examples: bool = True,
    verbose: bool = True,
) -> None:
    """
    Patches factowl.atomic_facts_sped_up_vllm prompts & parser, similarly to your scripts.

    template:
      - "sentence": SAMPLE_PROMPT_TEMPLATE = "Sentence:\\n<sample_text>\\n"
      - "entity":   SAMPLE_PROMPT_TEMPLATE = "Entity: <sample_topic>\\nText:\\n<sample_text>\\n"
    """
    import factowl.atomic_facts_sped_up_vllm as af

    af.DEFAULT_ATOMIZATION_PROMPTS["en"] = (
        "You are an expert fact extraction assistant.\n"
        "Task: extract atomic factual claims from the given sentence.\n\n"
        "Rules:\n"
        "1) Only extract claims that are stated in the sentence (no external knowledge).\n"
        "2) Each claim must be a standalone factual statement.\n"
        "3) Do NOT include opinions, advice, questions, or meta statements (e.g., 'the sentence is false').\n"
        "4) Do NOT add explanations or notes.\n"
        "5) Output ONLY a list of claims. Each claim must be on its own line.\n"
        "6) Each line MUST start with '- ' (dash + space).\n"
        "7) If there are no factual claims, output exactly: - NONE"
    )

    if template == "entity":
        af.SAMPLE_PROMPT_TEMPLATE = "Entity: <sample_topic>\nText:\n<sample_text>\n"
    else:
        af.SAMPLE_PROMPT_TEMPLATE = "Sentence:\n<sample_text>\n"

    if set_examples:
        af.FACT_GENERATION_EXAMPLES = [
            {
                "topic": "unused",
                "query": "Barack Obama was born in Honolulu, Hawaii, to a mother from Kansas and a father from Kenya.",
                "facts": [
                    "Barack Obama was born in Honolulu, Hawaii.",
                    "Barack Obama's mother was from Kansas.",
                    "Barack Obama's father was from Kenya.",
                ],
            },
            {
                "topic": "unused",
                "query": "Is there something else you would like to know about the gender of the first female US President?",
                "facts": ["NONE"],
            },
        ]

    atom_line_re = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)\s*(.*\S)\s*$")

    def text_to_sentences_v2(text: str) -> List[str]:
        out: List[str] = []
        for ln in (text or "").splitlines():
            m = atom_line_re.match(ln)
            if not m:
                continue
            s = m.group(1).strip()
            s_upper = s.strip().upper().rstrip(".")
            if s_upper in {"NONE", "NO FACTS", "N/A"}:
                continue
            if not s:
                continue
            if not s.endswith("."):
                s += "."
            out.append(s)
        return out

    af.text_to_sentences = text_to_sentences_v2

    if verbose:
        print("[patch] atomic module:", af.__file__)
        print("[patch] using patched text_to_sentences:", af.text_to_sentences.__name__)
        print("[patch] SAMPLE_PROMPT_TEMPLATE:", af.SAMPLE_PROMPT_TEMPLATE)
        print("[patch] ATOM_PROMPT_HEAD:", af.DEFAULT_ATOMIZATION_PROMPTS["en"][:200])
        try:
            print(
                "[patch] N_EXAMPLES:", len(getattr(af, "FACT_GENERATION_EXAMPLES", []))
            )
        except Exception:
            pass


# FactBench parsing helpers
_SENT_RE = re.compile(r"sentence(\d+)$")


def iter_sentences_in_order(
    sent_dict: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[int, str, Dict[str, Any]]] = []
    for k, v in (sent_dict or {}).items():
        m = _SENT_RE.search(k)
        idx = int(m.group(1)) if m else 10**9
        items.append((idx, k, v))
    items.sort(key=lambda t: t[0])
    return [(k, v) for _, k, v in items]


def norm_gold_label(x: Any) -> Optional[bool]:
    # dataset is bool or "NA"
    if x is True or x is False:
        return bool(x)
    if x is None:
        return None
    if isinstance(x, str) and x.strip().upper() == "NA":
        return None
    return None
