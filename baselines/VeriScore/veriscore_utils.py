import gc
import json
import math
import re
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from vllm import LLM, SamplingParams

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# Regex / basic helpers
_SENT_RE = re.compile(r"sentence(\d+)$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)

TAG_RE = re.compile(r"###\s*(Supported|Unsupported)\.?\s*###", re.I)
VERDICT_FALLBACK_RE = re.compile(r"^\s*(supported|unsupported)\b", re.I)

_BAD_PREFIX_RE = re.compile(
    r"^(here are|facts\b|fact\b|text\b|sentence to be focused on\b|extract\b|"
    r"no verifiable claim\b|your decision\b|claim\b|search result\b|title\b|link\b|content\b|"
    r"references?\b|reference\(s\)\b)\s*[:\-]?",
    re.I,
)

_BAD_CONTAINS_RE = re.compile(
    r"(?i)\b("
    r"no\s+verifiable\s+claim|"
    r"verifiable\s+claim|"
    r"verifiable\s+facts?|"
    r"extracted\s+verifiable|"
    r"verifiable\s+atomic|"
    r"atomic\s+facts?|"
    r"sentence\s+to\s+be\s+focused\s+on|"
    r"your\s+task|"
    r"your\s+decision|"
    r"search\s+result|"
    r"title\s*:|"
    r"link\s*:|"
    r"content\s*:|"
    r"facts\s*:|"
    r"text\s*:"
    r")\b"
)


def clean_seg(seg: str) -> str:
    seg = (seg or "").strip()
    seg = seg.replace('\\"', '"').strip()
    seg = seg.strip(' "\u201c\u201d')
    return seg


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def cm_to_macro_f1(cm: Dict[str, int]) -> Dict[str, Any]:
    # positive class = NOT_SUPPORTED
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


def update_confusion_not_supported_positive(
    cm: Dict[str, int], gold_supported: bool, pred_supported: bool
) -> None:
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


def roc_auc_manual(y_true: List[int], y_score: List[float]) -> float:
    # y_true: 1 for NOT_SUPPORTED, 0 for SUPPORTED
    if len(y_true) != len(y_score):
        return 0.0
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
    return float(u_pos / (n_pos * n_neg))


# VeriScore metrics
def f1_at_k(supported: int, total_claims: int, k: int) -> float:
    if total_claims <= 0 or supported <= 0:
        return 0.0
    p = supported / total_claims
    r = min(supported / max(k, 1), 1.0)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def median_int(xs: List[int]) -> int:
    xs = sorted([int(x) for x in xs if x is not None and int(x) > 0])
    if not xs:
        return 1
    m = len(xs) // 2
    if len(xs) % 2 == 1:
        return max(xs[m], 1)
    return max(int(round((xs[m - 1] + xs[m]) / 2.0)), 1)


def sentence_risk_not_supported(details: List[Dict[str, Any]]) -> float:
    """
    risk = 1 - (#supported / #claims)
    Treat None as not_supported (supported=0) because only explicit 'supported' counts.
    If no claims/details => 0.0 (don't penalize abstention in AUC)
    """
    if not details:
        return 0.0
    total = len(details)
    supp = sum(1 for d in details if (d.get("label") or "").lower() == "supported")
    return 1.0 - (supp / total)


# Tokenization / BM25
def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25OkapiLite:
    def __init__(
        self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75
    ):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_len = [len(x) for x in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)

        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            seen = set(doc)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        self.idf = {
            t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

        self.tf: List[Dict[str, int]] = []
        for doc in corpus_tokens:
            d: Dict[str, int] = {}
            for t in doc:
                d[t] = d.get(t, 0) + 1
            self.tf.append(d)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_len[i]
            denom_const = self.k1 * (1 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
            tf_i = self.tf[i]
            s = 0.0
            for t in query_tokens:
                if t not in tf_i:
                    continue
                f = tf_i[t]
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + denom_const)
            scores[i] = s
        return scores


def build_bm25_index(passages: List[Dict[str, str]]) -> Optional[BM25OkapiLite]:
    if not passages:
        return None
    toks = [tokenize(p.get("snippet", "")) for p in passages]
    return BM25OkapiLite(toks)


def select_topk_passages_bm25_indexed(
    bm25: Optional[BM25OkapiLite],
    passages: List[Dict[str, str]],
    query: str,
    topk: int,
) -> List[Dict[str, str]]:
    if not bm25 or not passages:
        return []
    scores = bm25.get_scores(tokenize(query))
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        : max(topk, 1)
    ]
    return [passages[i] for i in idxs]


# Evidence flattening / passage pool
def looks_like_url(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://") or s.startswith("www.")


def is_numeric_junk(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    if len(t) < 30:
        return t.isdigit()
    digits = sum(ch.isdigit() for ch in t)
    letters = sum(ch.isalpha() for ch in t)
    return digits > 2 * max(letters, 1)


def flatten_evidence(x: Any) -> List[str]:
    TEXT_KEYS = {
        "text",
        "content",
        "snippet",
        "evidence",
        "sentence",
        "passage",
        "title",
        "abstract",
    }
    out: List[str] = []

    def rec(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
            return
        if isinstance(v, (list, tuple)):
            for t in v:
                rec(t)
            return
        if isinstance(v, dict):
            for k in TEXT_KEYS:
                if k in v and isinstance(v[k], str) and v[k].strip():
                    out.append(v[k].strip())
            for vv in v.values():
                rec(vv)
            return
        s = str(v).strip()
        if s:
            out.append(s)

    rec(x)

    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def topic_from_prompt(prompt: str) -> str:
    return (prompt or "").strip()[:120] or "unknown_topic"


def topic_from_ref_or_prompt(prompt: str, ref: List[str]) -> str:
    from urllib.parse import unquote

    if (
        ref
        and isinstance(ref, list)
        and isinstance(ref[0], str)
        and "wikipedia.org/wiki/" in ref[0]
    ):
        return unquote(ref[0].split("/wiki/")[-1]).replace("_", " ")
    return (prompt or "").strip()[:120] or "unknown_topic"


def ref_text_to_passages(
    topic: str, ref_text: Any, max_passages: int = 60, max_chars: int = 1200
) -> List[Dict[str, str]]:
    if not ref_text or not str(ref_text).strip():
        return []
    if isinstance(ref_text, list):
        ref_text = "\n".join([str(x) for x in ref_text])

    txt = str(ref_text).replace("<s>", " ").replace("</s>", "\n")
    paras = [p.strip() for p in txt.split("\n") if p.strip()]

    passages: List[Dict[str, str]] = []
    for p in paras:
        for chunk in textwrap.wrap(
            p, width=max_chars, break_long_words=False, replace_whitespace=False
        ):
            passages.append({"title": topic, "snippet": chunk, "link": ""})
            if len(passages) >= max_passages:
                return passages
    return passages


def build_candidate_passages_factbench(
    prompt: str,
    sent_obj: Dict[str, Any],
    max_passages: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    topic = topic_from_prompt(prompt)

    ev_blobs = []
    for key in [
        "auto_evidence",
        "human_evidence",
        "evidence",
        "references",
        "ref_text",
    ]:
        if key in sent_obj and sent_obj[key] is not None:
            ev_blobs.append(sent_obj[key])

    ev_texts = flatten_evidence(ev_blobs)

    clean: List[str] = []
    for s in ev_texts:
        if looks_like_url(s):
            continue
        if is_numeric_junk(s):
            continue
        clean.append(s)

    joined = "\n".join(clean).strip()
    if not joined:
        return []

    return ref_text_to_passages(
        topic=topic, ref_text=joined, max_passages=max_passages, max_chars=max_chars
    )


# FactBench sentence iteration / gold normalization
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
    if x is True or x is False:
        return bool(x)
    if x is None:
        return None
    if isinstance(x, str) and x.strip().upper() == "NA":
        return None
    return None


# Claim extraction parsing
def _norm_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def dedup_claims(claims: List[str], max_claims: int) -> List[str]:
    out, seen = [], set()
    for c in claims:
        k = _norm_key(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= max_claims:
            break
    return out


def parse_claims_from_extraction(text: str, max_claims: int = 32) -> List[str]:
    if not text:
        return []
    low = text.strip().lower()
    if "no verifiable claim" in low:
        return []

    claims: List[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue

        ln = re.sub(r"^\s*[-*\u2022]+\s*", "", ln)
        ln = re.sub(r"^\s*\(?\d+[\).:]\s*", "", ln).strip()
        if not ln:
            continue

        if _BAD_PREFIX_RE.match(ln):
            continue
        if _BAD_CONTAINS_RE.search(ln):
            continue

        if len(ln) < 8:
            continue
        if sum(ch.isalpha() for ch in ln) < 3:
            continue
        if not re.search(r"[A-Za-z]", ln):
            continue

        # Drop URLs / markdown links
        if "http://" in ln or "https://" in ln:
            continue
        if re.match(r"^\[[^\]]+\]\((https?://|www\.)", ln):
            continue

        claims.append(ln)

    uniq, seen = [], set()
    for c in claims:
        c = c.strip()
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq[:max_claims]


# Window snippet (Claimify/VeriScore style)
def sentence_text_for_window(sent_obj: Dict[str, Any]) -> str:
    return clean_seg(sent_obj.get("decontext") or sent_obj.get("text") or "")


def build_extraction_snippet_with_window_factbench(
    prompt: str,
    ordered_sents: List[Tuple[str, Dict[str, Any]]],
    cur_idx: int,
    prev_n: int = 3,
    next_n: int = 1,
) -> str:
    lo = max(0, cur_idx - prev_n)
    hi = min(len(ordered_sents), cur_idx + 1 + next_n)
    parts: List[str] = []
    for j in range(lo, hi):
        s_txt = sentence_text_for_window(ordered_sents[j][1])
        if not s_txt:
            continue
        if j == cur_idx:
            parts.append(f"<SOS>{s_txt}<EOS>")
        else:
            parts.append(s_txt)
    return "\n".join(parts).strip()


def build_extraction_snippet_with_window_felm(
    question: str,
    sentences: List[str],
    cur_idx: int,
    prev_n: int = 3,
    next_n: int = 1,
) -> str:
    lo = max(0, cur_idx - prev_n)
    hi = min(len(sentences), cur_idx + 1 + next_n)
    parts: List[str] = []
    for j in range(lo, hi):
        s_txt = clean_seg(sentences[j])
        if not s_txt:
            continue
        if j == cur_idx:
            parts.append(f"<SOS>{s_txt}<EOS>")
        else:
            parts.append(s_txt)
    return "\n".join(parts).strip()


# Verification prompt formatting
def build_search_results_str(passages: List[Dict[str, str]], k: int) -> str:
    out = []
    for i, p in enumerate(passages[:k], start=1):
        title = (p.get("title") or "").strip()
        link = (p.get("link") or "").strip()
        snippet = (p.get("snippet") or "").strip()
        out.append(
            f"Search result {i}\n"
            f"Title: {title}\n"
            f"Link: {link}\n"
            f"Content: {snippet}\n"
        )
    return "\n".join(out).strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_fewshot_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def fill_verification_fewshot_template(
    verification_instruction_binary_txt: str,
    fewshot_rows: List[Dict[str, Any]],
) -> str:
    need = verification_instruction_binary_txt.count("{}")
    if need % 3 != 0:
        raise RuntimeError(
            f"verification_instruction_binary.txt has {need} '{{}}' placeholders (not divisible by 3)."
        )
    n_slots = need // 3

    triples = []
    for ex in fewshot_rows:
        claim = (ex.get("claim") or "").strip()
        sr = (ex.get("search_result") or "").strip()
        human = (ex.get("human_label") or ex.get("label") or "").strip().lower()
        if not claim or not sr or not human:
            continue
        lab = (
            "Supported"
            if human in {"support", "supported", "true", "entailed"}
            else "Unsupported"
        )
        triples.append((claim, sr, lab))

    if not triples:
        raise RuntimeError("No valid few-shot triples found in few_shot_examples.jsonl")

    if len(triples) >= n_slots:
        triples = triples[:n_slots]
    else:
        last = triples[-1]
        while len(triples) < n_slots:
            triples.append(last)

    flat_args: List[str] = []
    for c, r, l in triples:
        flat_args.extend([c, r, l])

    return verification_instruction_binary_txt.format(*flat_args)


def wrap_two_slot_template_checked(
    template_txt: str,
    system_msg: str,
    user_msg: str,
    must_contain_system: str = "",
) -> str:
    out = template_txt.format(system_msg or "", user_msg or "")
    if must_contain_system and must_contain_system not in out:
        raise RuntimeError(
            "Chat template seems wrong (system content not found after wrapping). "
            "Likely swapped placeholders or wrong template file."
        )
    return out


def build_chat_prompt_with_tokenizer(tokenizer, system_msg: str, user_msg: str) -> str:
    """
    Uses tokenizer.apply_chat_template if present.
    Falls back to a simple concat (still better than wrong alpaca template).
    """
    msgs = []
    if system_msg is not None and str(system_msg).strip():
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    if system_msg and str(system_msg).strip():
        return f"{system_msg}\n\n{user_msg}\n"
    return f"{user_msg}\n"


# Verdict parsing / scoring
def parse_verdict_strict(text: str) -> Optional[str]:
    """
    Return 'supported' / 'not_supported' / None
    using last ###Supported### / ###Unsupported###, preferring after last 'Your decision:' marker.
    """
    if not text:
        return None
    t = text.strip()

    idx = t.lower().rfind("your decision")
    if idx >= 0:
        tail = t[idx:]
        m_all = list(TAG_RE.finditer(tail))
        if m_all:
            lab = m_all[-1].group(1).strip().lower()
            return "supported" if lab == "supported" else "not_supported"

    m_all = list(TAG_RE.finditer(t))
    if m_all:
        lab = m_all[-1].group(1).strip().lower()
        return "supported" if lab == "supported" else "not_supported"

    return None


def parse_verdict_with_fallback(text: str) -> Optional[str]:
    lab = parse_verdict_strict(text)
    if lab is not None:
        return lab
    if not text:
        return None

    first = text.strip().splitlines()[0].strip()
    m = VERDICT_FALLBACK_RE.search(first)
    if m:
        return "supported" if m.group(1).lower() == "supported" else "not_supported"

    low = text.lower()
    if "unsupported" in low and "supported" not in low:
        return "not_supported"
    if "supported" in low and "unsupported" not in low:
        return "supported"
    return None


def score_not_supported_from_label(lab: Optional[str]) -> float:
    if lab == "not_supported":
        return 1.0
    if lab == "supported":
        return 0.0
    return 1.0  # conservative


def strict_sentence_supported(
    labels: List[Optional[str]],
    empty_claims_policy: str = "supported",
) -> Optional[bool]:
    """
    STRICT: supported iff every claim is supported.
    Any None/parse-fail counts as not_supported IF there are claims.

    If there are *no claims*:
      - supported: return True
      - not_supported: return False
      - skip: return None
    """
    if not labels:
        if empty_claims_policy == "supported":
            return True
        if empty_claims_policy == "not_supported":
            return False
        return None  # skip

    for lab in labels:
        if (lab or "").lower() != "supported":
            return False
    return True


# vLLM runtime / retries
def is_transient_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        m in msg
        for m in [
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
    )


def vllm_generate_with_retries(
    llm: LLM,
    prompts: List[str],
    sp: SamplingParams,
    max_tries: int = 3,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_tries + 1):
        try:
            outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
            return outs, None
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
                time.sleep(1.0 * (2 ** (attempt - 1)))
                continue
            return None, err_info

    return None, {
        "type": type(last_err).__name__ if last_err else "UnknownError",
        "message": str(last_err) if last_err else "",
        "traceback": traceback.format_exc(),
    }


class VLLMSession:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.55):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm: Optional[LLM] = None

    def __enter__(self) -> LLM:
        self.llm = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        return self.llm

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.llm is not None:
            del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# IO helpers
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _init_eff_fields(row: Dict[str, Any]) -> None:
    row["timing"] = {"extract_s": 0.0, "verify_s": 0.0, "total_s": 0.0}
    row["tokens"] = {
        "extract_prompt": 0,
        "extract_gen": 0,
        "verify_prompt": 0,
        "verify_gen": 0,
        "total_prompt": 0,
        "total_gen": 0,
    }
    row["flops"] = None


def _add_vllm_tokens(row: Dict[str, Any], out_obj: Any, stage: str) -> None:
    # stage in {"extract","verify"}
    prompt_tok = len(getattr(out_obj, "prompt_token_ids", []) or [])
    gen_tok = 0
    for o in getattr(out_obj, "outputs", None) or []:
        gen_tok += len(getattr(o, "token_ids", []) or [])
    row["tokens"][f"{stage}_prompt"] += int(prompt_tok)
    row["tokens"][f"{stage}_gen"] += int(gen_tok)


def _finalize_eff_row(
    row: Dict[str, Any], params_b: float, flops_per_param: float
) -> None:
    row["timing"]["total_s"] = float(
        row["timing"]["extract_s"] + row["timing"]["verify_s"]
    )

    row["tokens"]["total_prompt"] = int(
        row["tokens"]["extract_prompt"] + row["tokens"]["verify_prompt"]
    )
    row["tokens"]["total_gen"] = int(
        row["tokens"]["extract_gen"] + row["tokens"]["verify_gen"]
    )

    if not params_b or params_b <= 0:
        row["flops"] = None
        return

    P = params_b * 1e9
    k = float(flops_per_param)

    extract_tok = row["tokens"]["extract_prompt"] + row["tokens"]["extract_gen"]
    verify_tok = row["tokens"]["verify_prompt"] + row["tokens"]["verify_gen"]
    total_tok = extract_tok + verify_tok

    extract_gen_tok = row["tokens"]["extract_gen"]
    verify_gen_tok = row["tokens"]["verify_gen"]
    total_gen_tok = extract_gen_tok + verify_gen_tok

    def flops_for(tok: int) -> float:
        return k * P * float(tok)

    extract_flops = flops_for(extract_tok)
    verify_flops = flops_for(verify_tok)
    total_flops = flops_for(total_tok)

    extract_gen_flops = flops_for(extract_gen_tok)
    verify_gen_flops = flops_for(verify_gen_tok)
    total_gen_flops = flops_for(total_gen_tok)

    def tflops_per_s(flops: float, seconds: float):
        if seconds <= 0:
            return None
        return flops / seconds / 1e12

    row["flops"] = {
        "params_b": float(params_b),
        "k": float(k),
        "extract_tokens": int(extract_tok),
        "verify_tokens": int(verify_tok),
        "total_tokens": int(total_tok),
        "extract_gen_tokens": int(extract_gen_tok),
        "verify_gen_tokens": int(verify_gen_tok),
        "total_gen_tokens": int(total_gen_tok),
        "extract_flops": float(extract_flops),
        "verify_flops": float(verify_flops),
        "total_flops": float(total_flops),
        "extract_gen_flops": float(extract_gen_flops),
        "verify_gen_flops": float(verify_gen_flops),
        "total_gen_flops": float(total_gen_flops),
        "extract_tflops_per_s": tflops_per_s(extract_flops, row["timing"]["extract_s"]),
        "verify_tflops_per_s": tflops_per_s(verify_flops, row["timing"]["verify_s"]),
        "total_tflops_per_s": tflops_per_s(total_flops, row["timing"]["total_s"]),
        # won't be inflated by prompt prefix caching
        "extract_gen_tflops_per_s": tflops_per_s(
            extract_gen_flops, row["timing"]["extract_s"]
        ),
        "verify_gen_tflops_per_s": tflops_per_s(
            verify_gen_flops, row["timing"]["verify_s"]
        ),
        "total_gen_tflops_per_s": tflops_per_s(
            total_gen_flops, row["timing"]["total_s"]
        ),
    }


def _agg_efficiency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n_rows": 0,
            "avg_claims_per_sentence": 0.0,
            "avg_extract_time_s_per_sentence": 0.0,
            "avg_verify_time_s_per_sentence": 0.0,
            "avg_total_time_s_per_sentence": 0.0,
        }

    sum_claims = sum(int(r.get("total_claims", 0) or 0) for r in rows)
    sum_extract_s = sum(
        float((r.get("timing") or {}).get("extract_s", 0.0) or 0.0) for r in rows
    )
    sum_verify_s = sum(
        float((r.get("timing") or {}).get("verify_s", 0.0) or 0.0) for r in rows
    )
    sum_total_s = sum(
        float((r.get("timing") or {}).get("total_s", 0.0) or 0.0) for r in rows
    )

    return {
        "n_rows": n,
        "avg_claims_per_sentence": sum_claims / n,
        "avg_extract_time_s_per_sentence": sum_extract_s / n,
        "avg_verify_time_s_per_sentence": sum_verify_s / n,
        "avg_total_time_s_per_sentence": sum_total_s / n,
        "sum_extract_time_s": sum_extract_s,
        "sum_verify_time_s": sum_verify_s,
        "sum_total_time_s": sum_total_s,
    }


def _agg_compute(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sum_extract_flops = 0.0
    sum_verify_flops = 0.0
    sum_total_flops = 0.0

    sum_extract_gen_flops = 0.0
    sum_verify_gen_flops = 0.0
    sum_total_gen_flops = 0.0

    sum_extract_s = 0.0
    sum_verify_s = 0.0
    sum_total_s = 0.0

    for r in rows:
        t = r.get("timing") or {}
        sum_extract_s += float(t.get("extract_s", 0.0) or 0.0)
        sum_verify_s += float(t.get("verify_s", 0.0) or 0.0)
        sum_total_s += float(t.get("total_s", 0.0) or 0.0)

        f = r.get("flops") or {}
        sum_extract_flops += float(f.get("extract_flops", 0.0) or 0.0)
        sum_verify_flops += float(f.get("verify_flops", 0.0) or 0.0)
        sum_total_flops += float(f.get("total_flops", 0.0) or 0.0)

        sum_extract_gen_flops += float(f.get("extract_gen_flops", 0.0) or 0.0)
        sum_verify_gen_flops += float(f.get("verify_gen_flops", 0.0) or 0.0)
        sum_total_gen_flops += float(f.get("total_gen_flops", 0.0) or 0.0)

    def tflops_per_s(flops: float, seconds: float):
        if seconds <= 0:
            return None
        return flops / seconds / 1e12

    return {
        "sum_extract_flops": sum_extract_flops,
        "sum_verify_flops": sum_verify_flops,
        "sum_total_flops": sum_total_flops,
        "extract_tflops_per_s_agg": tflops_per_s(sum_extract_flops, sum_extract_s),
        "verify_tflops_per_s_agg": tflops_per_s(sum_verify_flops, sum_verify_s),
        "total_tflops_per_s_agg": tflops_per_s(sum_total_flops, sum_total_s),
        # gen-only aggregation
        "sum_extract_gen_flops": sum_extract_gen_flops,
        "sum_verify_gen_flops": sum_verify_gen_flops,
        "sum_total_gen_flops": sum_total_gen_flops,
        "extract_gen_tflops_per_s_agg": tflops_per_s(
            sum_extract_gen_flops, sum_extract_s
        ),
        "verify_gen_tflops_per_s_agg": tflops_per_s(sum_verify_gen_flops, sum_verify_s),
        "total_gen_tflops_per_s_agg": tflops_per_s(sum_total_gen_flops, sum_total_s),
        "sum_extract_time_s": sum_extract_s,
        "sum_verify_time_s": sum_verify_s,
        "sum_total_time_s": sum_total_s,
    }
