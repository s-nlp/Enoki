import ast
import gc
import json
import math
import re
import textwrap
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)

# Models that use <think>…</think> reasoning blocks.
_THINKING_MODEL_PATTERNS = ("qwen3", "qwq", "deepseek-r1", "deepseek-r2")


def is_thinking_model(model: str) -> bool:
    """Return True if the model name indicates a thinking/reasoning model."""
    return any(p in (model or "").lower() for p in _THINKING_MODEL_PATTERNS)


# Small helpers
def safe_div(a, b):
    return a / b if b else 0.0


def clean(s: str) -> str:
    s = (s or "").strip()
    s = s.replace('\\"', '"').strip()
    s = s.strip(' "\u201c\u201d')
    return s


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def majority_normalized(strings: List[str]) -> str:
    counts: Dict[str, int] = {}
    for s in strings:
        k = normalize_ws(s)
        if not k:
            continue
        counts[k] = counts.get(k, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""


def strip_think(text: str) -> str:
    if not text:
        return text
    
    # First, try removing think blocks normally
    result = _THINK_BLOCK_RE.sub("", text)
    result = result.replace("<think>", "").replace("</think>", "")
    result = result.strip()
    
    # If result is empty but original text had content, the entire response
    # might have been inside <think> tags. Extract the content instead.
    if not result and text.strip():
        # Extract content from inside think tags
        match = re.search(r"<think>(.*?)</think>", text, re.I | re.S)
        if match:
            result = match.group(1).strip()
        else:
            # No think tags found, return original stripped
            result = text.strip()
    
    return result


# Claimify excerpt window (paper §3.1)
def build_excerpt(sentences: List[str], cur_idx: int, p: int, f: int) -> str:
    n = len(sentences)
    lo = max(0, cur_idx - p)
    hi = min(n, cur_idx + 1 + f)
    parts = [clean(sentences[i]) for i in range(lo, hi) if clean(sentences[i])]
    if lo > 0:
        parts = ["[...]"] + parts
    if hi < n:
        parts = parts + ["[...]"]
    return "\n".join(parts).strip()


# Evidence helpers
def flatten_evidence(x: Any) -> List[str]:
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
    topic: str, ref_text: Any, max_passages: int = 80, max_chars: int = 1200
) -> List[Dict[str, str]]:
    if not ref_text or not str(ref_text).strip():
        return []
    if isinstance(ref_text, list):
        ref_text = "\n".join([str(x) for x in ref_text])

    txt = str(ref_text).replace("<s>", " ").replace("</s>", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", txt) if b.strip()]
    paras: List[str] = []
    for b in blocks:
        ls = [ln.strip() for ln in b.split("\n") if ln.strip()]
        paras.extend(ls if len(ls) > 1 else [b.strip()])

    passages: List[Dict[str, str]] = []
    for p in paras:
        for chunk in textwrap.wrap(
            p, width=max_chars, break_long_words=False, replace_whitespace=False
        ):
            passages.append({"title": topic, "snippet": chunk, "link": ""})
            if len(passages) >= max_passages:
                return passages
    return passages


# BM25
class BM25Lite:
    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs_tokens
        self.k1 = k1
        self.b = b
        self.N = len(docs_tokens)
        self.avgdl = (
            safe_div(sum(len(d) for d in docs_tokens), self.N) if self.N else 0.0
        )

        df: Dict[str, int] = {}
        for d in docs_tokens:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.idf = {
            t: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

        self.tfs: List[Dict[str, int]] = []
        for d in docs_tokens:
            m: Dict[str, int] = {}
            for t in d:
                m[t] = m.get(t, 0) + 1
            self.tfs.append(m)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        if self.N == 0:
            return 0.0
        tf = self.tfs[doc_idx]
        dl = len(self.docs[doc_idx]) or 1
        score = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            score += idf * (f * (self.k1 + 1) / denom)
        return score

    def topk(self, query: str, k: int) -> List[int]:
        qt = tokenize(query)
        scored = [(self.score(qt, i), i) for i in range(self.N)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored[:k]]


def build_search_results_str(passages: List[Dict[str, str]]) -> str:
    out = []
    for i, p in enumerate(passages, start=1):
        out.append(
            f"Search result {i}\n"
            f"Title: {p.get('title','')}\n"
            f"Link: {p.get('link','')}\n"
            f"Content: {p.get('snippet','')}\n"
        )
    return "\n".join(out).strip()


# Parsing
_SEL_FINAL_RE = re.compile(r"Final submission\s*:\s*(.+)", re.I)
_SEL_SENT_RE = re.compile(r"Sentence with only verifiable information\s*:\s*(.+)", re.I)


def parse_selection_output(
    text: str, original_sentence: str
) -> Tuple[Optional[bool], Optional[str], Optional[str]]:
    if not text:
        return None, None, None
    m_final = _SEL_FINAL_RE.search(text)
    m_sent = _SEL_SENT_RE.search(text)
    if not m_final or not m_sent:
        return None, None, None

    decision = m_final.group(1).strip().strip('"')
    sent = m_sent.group(1).strip().strip('"')

    low_dec = decision.lower()
    contains = ("does not contain" not in low_dec) and (
        "doesn't contain" not in low_dec
    )
    if not contains:
        return False, None, decision

    low_sent = sent.lower().strip()
    if low_sent in {"none", "<none>"}:
        return False, None, decision
    if low_sent == "remains unchanged":
        return True, original_sentence, decision

    sent = clean(sent)
    return (True, sent, decision) if sent else (None, None, decision)


# Disambiguation
_DISAMB_RE = re.compile(
    r"Decontextualized\s*Sentence\s*:\s*(.+)", re.I
)  # accept space/no-space
_S_RESTATED_RE = re.compile(r"^\s*S_restated\s*=\s*(.+?)\s*$", re.I | re.M)


def _strip_wrapping_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = s.strip(' "\u201c\u201d\u2018\u2019')
    return s.strip()


def _maybe_unescape_python_string(s: str) -> str:
    s2 = (s or "").strip()
    if len(s2) >= 2 and s2[0] in {"'", '"'} and s2[-1] == s2[0]:
        try:
            return str(ast.literal_eval(s2))
        except Exception:
            return s
    return s


def parse_disambiguation_output(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    m = _DISAMB_RE.search(text)
    if m:
        tail = clean(m.group(1))
    else:
        m2 = _S_RESTATED_RE.search(text)
        if not m2:
            return None, None
        tail = m2.group(1).strip()
        tail = _maybe_unescape_python_string(_strip_wrapping_quotes(tail))
        tail = clean(tail)

    low = tail.lower()
    if "cannot be decontextualized" in low or "cannot be disambiguated" in low:
        return "cannot", None
    if not tail:
        return None, None
    return "ok", tail


# Decomposition
_TRUE_FALSE_TAIL_RE = re.compile(r"\s*-\s*true\s+or\s+false\s*\?\s*$", re.I)
_QUOTED_ITEM_RE = re.compile(r'"([^"]+)"')

_DECOMP_HDR_CTX = re.compile(
    r"Propositions\s+with\s+Essential\s+Context\s*/\s*Clarifications\s*:\s*", re.I
)
_DECOMP_HDR_PLAIN = re.compile(
    r"Specific,\s*Verifiable,\s*and\s*Decontextualized\s*Propositions\s*:\s*", re.I
)


def _extract_balanced_bracket_list(s: str, start_idx: int) -> Optional[str]:
    if start_idx < 0 or start_idx >= len(s) or s[start_idx] != "[":
        return None
    depth = 0
    for j in range(start_idx, len(s)):
        ch = s[j]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start_idx : j + 1]
    return None


def _find_list_after_header(text: str, header_regex: re.Pattern) -> Optional[str]:
    m = header_regex.search(text or "")
    if not m:
        return None
    tail = text[m.end() :]
    idx = tail.find("[")
    if idx < 0:
        return None
    return _extract_balanced_bracket_list(tail, idx)


def _last_list_in_text(text: str) -> Optional[str]:
    if not text:
        return None
    last = text.rfind("[")
    if last < 0:
        return None
    return _extract_balanced_bracket_list(text, last)


def _parse_non_json_bracket_items(blk: str) -> List[str]:
    if not blk:
        return []
    inner = blk.strip()
    if inner.startswith("["):
        inner = inner[1:]
    if inner.endswith("]"):
        inner = inner[:-1]
    inner = inner.strip()
    if not inner:
        return []

    if "\n" in inner:
        items = []
        for ln in inner.splitlines():
            t = ln.strip()
            if not t:
                continue
            t = t.lstrip("-•*").strip()
            t = t.rstrip(",").strip()
            if t:
                items.append(t)
        return items

    parts = [p.strip() for p in inner.split(",")]
    return [p for p in parts if p]


def parse_decomposition_output(
    text: str, max_claims: int
) -> Tuple[Optional[bool], List[str], Optional[str]]:
    if not text:
        return None, [], None

    blk = (
        _find_list_after_header(text, _DECOMP_HDR_CTX)
        or _find_list_after_header(text, _DECOMP_HDR_PLAIN)
        or _last_list_in_text(text)
    )
    if not blk:
        return None, [], None

    items: List[str] = []
    try:
        arr = json.loads(blk)
        if isinstance(arr, list):
            items = [clean(str(x)) for x in arr if clean(str(x))]
    except Exception:
        pass

    if not items:
        items = [
            clean(m.group(1))
            for m in _QUOTED_ITEM_RE.finditer(blk)
            if clean(m.group(1))
        ]

    if not items:
        items = [clean(x) for x in _parse_non_json_bracket_items(blk) if clean(x)]

    cleaned: List[str] = []
    seen: set = set()
    for it in items:
        it2 = _TRUE_FALSE_TAIL_RE.sub("", it).strip()
        it2 = clean(it2)
        k = normalize_ws(it2).lower()
        if not it2 or k in seen:
            continue
        seen.add(k)
        cleaned.append(it2)
        if len(cleaned) >= max_claims:
            break

    return True, cleaned, blk


# Selection anti-meta filtering
_META_PAT = re.compile(
    r"\b(excerpt|the excerpt|this excerpt|the sentence|this sentence|question|the question|mentioned in the excerpt|"
    r"according to the excerpt|in the excerpt|in this excerpt|provided excerpt|provided context)\b",
    re.I,
)


def is_meta_rewrite(s: str) -> bool:
    s2 = normalize_ws(s).lower()
    if not s2:
        return True
    if _META_PAT.search(s2):
        return True
    if (
        s2.startswith("step ")
        or "reflect on criteria" in s2
        or "objective description" in s2
    ):
        return True
    return False


# Verification prompt + parsing
VERIFY_SYSTEM = "You are a careful fact-checking assistant."
_FINAL_ANS_RE = re.compile(r"Final\s*Answer\s*:\s*(Supported|Not\s*Supported)", re.I)


def build_verify_user(claim: str, passages: List[Dict[str, str]]) -> str:
    """Prompt for verification stage from SAFE paper (5.3 Decontextualization -> 3. Determine Veracity)"""
    sr = build_search_results_str(passages)
    return (
        "Instructions:\n"
        "1. You have been given a STATEMENT and some KNOWLEDGE points.\n"
        "2. Determine whether the STATEMENT is supported by the KNOWLEDGE. "
        "The STATEMENT does not need to be explicitly stated by the KNOWLEDGE, "
        "but should be strongly implied by it.\n"
        "3. Before showing your answer, think step-by-step and show your specific reasoning. "
        "As part of your reasoning, summarize the main points of the KNOWLEDGE.\n"
        "4. If the STATEMENT is supported by the KNOWLEDGE, show the supporting evidence.\n"
        "5. After stating your reasoning, restate the STATEMENT and then determine your final answer.\n"
        "6. If any element of the statement is not supported by the knowledge, the statement is not supported.\n"
        '7. Your final answer should be either "Supported" or "Not Supported". '
        "Wrap your final answer in square brackets.\n\n"
        f"KNOWLEDGE:\n{sr}\n\n"
        f"STATEMENT:\n{claim}\n"
    )


_FINAL_BRACKET_RE = re.compile(r"\[(supported|not\s*supported)\]", re.I)
_LASTLINE_VERDICT_RE = re.compile(r"(?mi)^\s*(supported|not\s*supported)\s*$")


def parse_verdict(text: str) -> Optional[str]:
    """Parse verification verdict from LLM output. Returns 'supported', 'not_supported', or None."""
    if not text:
        return None
    text = strip_think(text)

    m = None
    for mm in _FINAL_BRACKET_RE.finditer(text):
        m = mm
    if m:
        lab = m.group(1).lower().replace(" ", "")
        return "supported" if lab == "supported" else "not_supported"

    m2 = None
    for mm in _FINAL_ANS_RE.finditer(text):
        m2 = mm
    if m2:
        lab = m2.group(1).lower().replace(" ", "")
        return "supported" if lab == "supported" else "not_supported"
    
    # Accept a clean last-line verdict (common when models omit brackets)
    m3 = None
    for mm in _LASTLINE_VERDICT_RE.finditer(text):
        m3 = mm
    if m3:
        lab = m3.group(1).lower().replace(" ", "")
        return "supported" if lab == "supported" else "not_supported"

    low = text.lower()
    if "not supported" in low or "not_supported" in low or "unsupported" in low:
        return "not_supported"
    # If the model didn't emit an explicit verdict, treat as unknown.
    # if re.search(r"\bsupported\b", low):
        # return "supported"
    return None


def strict_sentence_supported(claim_labels: List[Optional[str]]) -> Optional[bool]:
    if claim_labels is None or len(claim_labels) == 0:
        return None
    if any(lab not in {"supported", "not_supported"} for lab in claim_labels):
        return None
    return all(lab == "supported" for lab in claim_labels)


def sentence_risk_strict(labels: List[Optional[str]]) -> Optional[float]:
    if labels is None or len(labels) == 0:
        return None
    if any(lab != "supported" for lab in labels):
        return 1.0
    return 0.0


# Time to add usage tracking
@dataclass
class Usage:
    prompt_tokens: int = 0
    gen_tokens: int = 0


@dataclass
class BatchResult:
    texts: List[List[str]]
    usage: Optional[List[Usage]] = None


# LLM backends
@dataclass
class GenParams:
    temperature: float
    max_tokens: int
    stop: Optional[List[str]] = None
    n: int = 1


class LLMBackend:
    def generate(
        self, batch_messages: List[List[Dict[str, str]]], gp: GenParams
    ) -> List[List[str]]:
        """Return: per input -> list of completions (len == gp.n ideally)."""
        raise NotImplementedError

    def generate_with_usage(
        self, batch_messages: List[List[Dict[str, str]]], gp: GenParams
    ) -> BatchResult:
        # fallback: just texts, no usage
        return BatchResult(texts=self.generate(batch_messages, gp), usage=None)


def is_transient_error_msg(msg: str) -> bool:
    msg = (msg or "").lower()
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


class VLLMBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        tokenizer_name: str,
        trust_remote_code: bool,
        gpu_memory_utilization: float,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model = model
        self.thinking = is_thinking_model(model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code
        )
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_name,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def close(self):
        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
                if self.thinking:
                    kwargs["enable_thinking"] = True
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        # fallback
        sys = ""
        user = ""
        for m in messages:
            if m["role"] == "system":
                sys += m["content"] + "\n"
            elif m["role"] == "user":
                user += m["content"] + "\n"
        return (sys + "\n" if sys else "") + user

    def _make_sp(self, gp) -> "SamplingParams":
        from vllm import SamplingParams
        if self.thinking:
            # Qwen3 docs recommend: temperature=0.6, top_p=0.95, top_k=20 for thinking mode.
            t = gp.temperature if gp.temperature == 0.0 else 0.6
            return SamplingParams(temperature=t, top_p=0.95, top_k=20,
                                  max_tokens=gp.max_tokens, stop=gp.stop, n=gp.n)
        return SamplingParams(temperature=gp.temperature, max_tokens=gp.max_tokens,
                              stop=gp.stop, n=gp.n)

    def generate(
        self, batch_messages: List[List[Dict[str, str]]], gp: GenParams
    ) -> List[List[str]]:
        prompts = [self._to_prompt(msgs) for msgs in batch_messages]
        sp = self._make_sp(gp)

        # light retry wrapper
        max_tries = 3
        base_sleep = 1.0
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                outs = self.llm.generate(prompts, sampling_params=sp, use_tqdm=False)
                res: List[List[str]] = []
                for o in outs:
                    texts = [strip_think(x.text) for x in (o.outputs or [])]
                    res.append(texts)
                return res
            except Exception as e:
                last_err = e
                if attempt < max_tries and is_transient_error_msg(str(e)):
                    time.sleep(base_sleep * (2 ** (attempt - 1)))
                    continue
                raise

        raise RuntimeError(f"vLLM generate failed: {last_err}")

    def generate_with_usage(
        self, batch_messages: List[List[Dict[str, str]]], gp: GenParams
    ) -> BatchResult:
        """Generate completions and return per-request token usage counts."""
        prompts = [self._to_prompt(msgs) for msgs in batch_messages]
        sp = self._make_sp(gp)

        max_tries = 3
        base_sleep = 1.0
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                outs = self.llm.generate(prompts, sampling_params=sp, use_tqdm=False)

                all_texts: List[List[str]] = []
                all_usage: List[Usage] = []

                for o in outs:
                    prompt_tok = len(getattr(o, "prompt_token_ids", []) or [])
                    texts = [strip_think(x.text) for x in (o.outputs or [])]
                    gen_tok = 0
                    for x in o.outputs or []:
                        gen_tok += len(getattr(x, "token_ids", []) or [])
                    all_texts.append(texts)
                    all_usage.append(
                        Usage(prompt_tokens=prompt_tok, gen_tokens=gen_tok)
                    )

                return BatchResult(texts=all_texts, usage=all_usage)

            except Exception as e:
                last_err = e
                if attempt < max_tries and is_transient_error_msg(str(e)):
                    time.sleep(base_sleep * (2 ** (attempt - 1)))
                    continue
                raise

        raise RuntimeError(f"vLLM generate failed: {last_err}")


class OpenRouterBackend(LLMBackend):
    """
    OpenRouter is OpenAI-compatible for chat completions.

    Env var recommended:
      OPENROUTER_API_KEY

    Optional headers:
      HTTP-Referer, X-Title (OpenRouter suggests these for attribution).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        timeout_s: float = 120.0,
        max_workers: int = 8,
    ):
        from concurrent.futures import ThreadPoolExecutor

        import httpx

        self.model = model
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **({"HTTP-Referer": http_referer} if http_referer else {}),
                **({"X-Title": x_title} if x_title else {}),
            },
        )
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
        try:
            self.pool.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    def _one(self, messages: List[Dict[str, str]], gp: GenParams) -> List[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": gp.temperature,
            "max_tokens": gp.max_tokens,
            "n": gp.n,
        }
        if gp.stop:
            payload["stop"] = gp.stop

        r = self.client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        outs: List[str] = []
        for c in choices:
            msg = (c.get("message") or {}).get("content") or ""
            outs.append(strip_think(msg))
        # Some providers might return < gp.n choices; keep what we got
        return outs

    def generate(
        self, batch_messages: List[List[Dict[str, str]]], gp: GenParams
    ) -> List[List[str]]:
        """
        IMPORTANT: Claimify selection/disambiguation rely on n>1 with min_successes>=2.
        Some OpenRouter providers ignore `n` and always return 1 choice.
        To make Claimify gating work, we top-up missing completions by issuing extra
        requests with n=1 until we have gp.n completions (or we hit a safety cap).
        """
        res: List[List[str]] = []

        # Top-up generator params: always request a single completion.
        gp1 = GenParams(
            temperature=gp.temperature,
            max_tokens=gp.max_tokens,
            stop=gp.stop,
            n=1,
        )

        for msgs in batch_messages:
            outs = self._one(msgs, gp)

            # If provider ignored n, fetch extra single completions.
            # Safety cap prevents infinite loops when provider returns 0 choices.
            topup_tries = 0
            max_topup_tries = max(3, (gp.n or 1) * 4)
            while (gp.n or 1) > len(outs) and topup_tries < max_topup_tries:
                more = self._one(msgs, gp1)
                if not more:
                    break
                outs.extend(more)
                topup_tries += 1

            # Keep exactly gp.n outputs (or fewer if provider is broken).
            want = gp.n or 1
            res.append(outs[:want])

        return res

    def _one_with_usage_raw(
        self, messages: List[Dict[str, str]], gp: GenParams
    ) -> Tuple[List[str], Usage]:
        """Single HTTP call. May return fewer than gp.n outputs if the provider ignores n."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": gp.temperature,
            "max_tokens": gp.max_tokens,
            "n": gp.n,
        }
        if gp.stop:
            payload["stop"] = gp.stop

        r = self.client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()

        choices = data.get("choices") or []

        # ADD: Check for empty choices and log
        if not choices:
            import warnings
            warnings.warn(f"OpenRouter returned 0 choices for model {self.model}")

        outs: List[str] = []
        for c in choices:
            msg = (c.get("message") or {}).get("content") or ""
            outs.append(strip_think(msg))

        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            gen_tokens=int(u.get("completion_tokens") or 0),
        )
        return outs, usage

    def _one_with_usage(
        self, messages: List[Dict[str, str]], gp: GenParams
    ) -> Tuple[List[str], Usage]:
        """
        If the provider ignores n and returns < gp.n outputs, top up with extra
        single-completion calls until we reach gp.n.  Token usage is accumulated
        across all calls.  Claimify gating (min_successes) requires gp.n outputs.
        """
        want = gp.n or 1
        max_attempts = 3
        for attempt in range(max_attempts):
            outs, usage = self._one_with_usage_raw(messages, gp)
            
            if len(outs) >= want:
                return outs[:want], usage
            
            if len(outs) == 0 and attempt < max_attempts - 1:
                # Retry if we got nothing
                import time
                time.sleep(1.0 * (attempt + 1))
                continue

        gp1 = GenParams(
            temperature=gp.temperature,
            max_tokens=gp.max_tokens,
            stop=gp.stop,
            n=1,
        )

        topup_tries = 0
        max_topup_tries = max(3, want * 4)
        while len(outs) < want and topup_tries < max_topup_tries:
            more, u2 = self._one_with_usage_raw(messages, gp1)
            usage.prompt_tokens += u2.prompt_tokens
            usage.gen_tokens += u2.gen_tokens
            if not more:
                break
            outs.extend(more)
            topup_tries += 1

        return outs[:want], usage

    def generate_with_usage(self, batch_messages, gp: GenParams) -> BatchResult:
        """Parallel generate with token usage tracking (thread pool over HTTP calls)."""
        futs = [
            self.pool.submit(self._one_with_usage, msgs, gp) for msgs in batch_messages
        ]
        texts: List[List[str]] = []
        usages: List[Usage] = []
        for fut in futs:
            out_texts, u = fut.result()
            texts.append(out_texts)
            usages.append(u)
        return BatchResult(texts=texts, usage=usages)


class OpenAIBackend(LLMBackend):
    """
    OpenAI Chat Completions backend.

    Uses Chat Completions (not Responses) because Claimify needs `n>1` for self-consistency gating.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 120.0,
        max_workers: int = 8,
        use_developer_role: bool = False,
    ):
        from concurrent.futures import ThreadPoolExecutor

        from openai import OpenAI

        self.model = model
        self.use_developer_role = use_developer_role
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    @staticmethod
    def _choice_text(choice: Any) -> str:
        """
        Robustly extract text from a Chat Completions choice.
        Some models/providers may return:
          - message.content = None (e.g., content filtered)
          - message.refusal populated instead
          - content as a list (multimodal) -> treat as empty here
        """
        try:
            msg = getattr(choice, "message", None)
            if msg is None:
                return ""

            content = getattr(msg, "content", None)
            refusal = getattr(msg, "refusal", None)

            txt = ""
            # content may be a string or a list (multimodal). We only handle string here.
            if isinstance(content, str):
                txt = content.strip()
            # if content is empty/non-string, try refusal (string)
            if (not txt) and isinstance(refusal, str):
                txt = refusal.strip()

            return txt
        except Exception:
            return ""

    def close(self):
        try:
            self.pool.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    def _map_roles(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not self.use_developer_role:
            return messages
        out = []
        for m in messages:
            if m.get("role") == "system":
                out.append({"role": "developer", "content": m.get("content", "")})
            else:
                out.append(m)
        return out
    
    def _create_with_retry_all_empty(self, kwargs: Dict[str, Any], tries: int = 3):
        """
        Call chat.completions.create. If choices exist but ALL extracted texts are empty,
        retry a couple times (helps with transient provider/model glitches returning empty content).
        """
        last = None
        for _ in range(max(1, tries)):
            resp = self.client.chat.completions.create(**kwargs)
            last = resp
            choices = getattr(resp, "choices", None) or []
            if not choices:
                return resp
            outs = [strip_think(self._choice_text(c)) for c in choices]
            # If at least one is non-empty, keep it
            if any(o.strip() for o in outs):
                return resp
        return last

    def _one_with_usage(
        self, messages: List[Dict[str, str]], gp: GenParams
    ) -> Tuple[List[str], Usage]:
        msgs = self._map_roles(messages)

        # Check if this is a GPT-5 model (has different parameter requirements)
        is_gpt5 = "gpt-5" in self.model.lower()
        # Check if model uses max_completion_tokens
        is_new_model = any(x in self.model.lower() for x in ["gpt-5", "o1", "o3"])
        
        # Prepare kwargs for API call
        kwargs = {
            "model": self.model,
            "messages": msgs,
        }
        
        # GPT-5 doesn't support temperature parameter - skip it entirely
        if not is_gpt5:
            kwargs["temperature"] = gp.temperature
            kwargs["n"] = gp.n
        # For GPT-5, we need to make multiple calls if n > 1 (handled below)
        
        # Token limit parameter
        if is_new_model:
            kwargs["max_completion_tokens"] = gp.max_tokens
        else:
            kwargs["max_tokens"] = gp.max_tokens
            
        # GPT-5 doesn't support stop parameter
        if gp.stop and not is_gpt5:
            kwargs["stop"] = gp.stop

        # For GPT-5 with n > 1, we need to make multiple API calls
        if is_gpt5 and gp.n > 1:
            all_outs: List[str] = []
            total_prompt = 0
            total_gen = 0
            for _ in range(gp.n):
                resp = self._create_with_retry_all_empty(kwargs, tries=3)
                for c in resp.choices or []:
                    txt = self._choice_text(c)
                    all_outs.append(strip_think(txt))
                u = getattr(resp, "usage", None)
                total_prompt += int(getattr(u, "prompt_tokens", 0) or 0)
                total_gen += int(getattr(u, "completion_tokens", 0) or 0)
            return all_outs, Usage(prompt_tokens=total_prompt, gen_tokens=total_gen)

        resp = self._create_with_retry_all_empty(kwargs, tries=3)

        outs: List[str] = []
        for c in resp.choices or []:
            txt = self._choice_text(c)
            outs.append(strip_think(txt))

        u = getattr(resp, "usage", None)
        usage = Usage(
            prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            gen_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        )
        return outs, usage

    def generate(
        self, batch_messages: List[List[Dict[str, str]]], gp: GenParams
    ) -> List[List[str]]:
        res: List[List[str]] = []
        for msgs in batch_messages:
            outs, _u = self._one_with_usage(msgs, gp)
            res.append(outs)
        return res

    def generate_with_usage(
        self, batch_messages: List[List[Dict[str, str]]], gp: GenParams
    ) -> BatchResult:
        futs = [
            self.pool.submit(self._one_with_usage, msgs, gp) for msgs in batch_messages
        ]
        texts: List[List[str]] = []
        usages: List[Usage] = []
        for fut in futs:
            outs, u = fut.result()
            texts.append(outs)
            usages.append(u)
        return BatchResult(texts=texts, usage=usages)


# Claimify stage runner
@dataclass
class StageCfg:
    name: str
    system: str
    user_tmpl: str
    p: int
    f: int
    n: int
    min_successes: int
    max_tokens: int
    max_retries: int = 2


def claimify_temperature_for(cfg: StageCfg) -> float:
    # Paper Appendix D: temperature = 0.2 when multiple completions are
    # requested, otherwise 0.
    return 0.2 if cfg.n and cfg.n > 1 else 0.0


def build_messages(system_msg: str, user_msg: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_msg and str(system_msg).strip():
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def run_stage_claimify(
    *,
    backend: LLMBackend,
    cfg: StageCfg,
    question: str,
    excerpt: str,
    sentence: str,
    parse_fn: Callable[[str], Any],
    is_parseable: Callable[[Any], bool],
    stop: List[str],
) -> Tuple[
    Optional[List[str]], Optional[List[Any]], Optional[Dict[str, Any]], float, Usage
]:
    """
    Returns:
      texts, parsed, err, wall_time_s, usage(prompt+gen)
    wall_time includes retries.
    """
    gp = GenParams(
        temperature=claimify_temperature_for(cfg),
        max_tokens=cfg.max_tokens,
        stop=stop,
        n=cfg.n,
    )
    user = cfg.user_tmpl.format(question=question, excerpt=excerpt, sentence=sentence)
    msgs = build_messages(cfg.system, user)

    t0_all = time.perf_counter()
    total_usage = Usage()

    last_err: Optional[Dict[str, Any]] = None
    for attempt in range(cfg.max_retries + 1):
        try:
            t0 = time.perf_counter()
            br = backend.generate_with_usage([msgs], gp)
            outs = br.texts[0]
            u = br.usage[0] if br.usage else Usage()
            total_usage.prompt_tokens += u.prompt_tokens
            total_usage.gen_tokens += u.gen_tokens
            _ = time.perf_counter() - t0
        except Exception as e:
            last_err = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "attempt": attempt,
            }
            continue

        # Hard check: Claimify gating assumes we actually got enough completions to possibly
        # satisfy min_successes. If a provider returns < min_successes choices - it's an API limitation.
        if len(outs) < cfg.min_successes:
            last_err = {
                "type": "too_few_completions",
                "stage": cfg.name,
                "attempt": attempt,
                "expected_n": cfg.n,
                "min_successes": cfg.min_successes,
                "got_n": len(outs),
                "texts": outs[:],
            }
            continue

        parsed = [parse_fn(t) for t in outs]
        parseable_mask = [bool(is_parseable(p)) for p in parsed]
        n_parseable = sum(parseable_mask)
        if n_parseable == 0:
            last_err = {
                "type": "format_invalid_all_completions",
                "attempt": attempt,
                "texts": outs[:],
                "parsed_preview": [str(p)[:800] for p in parsed],
                "n_parseable": 0,
                "n": len(parsed),
            }
            continue

        # If we have fewer parseable outputs than the stage needs to make a
        # decision, retry instead of prematurely treating the sentence as a
        # semantic stop (e.g. "no verifiable claims" or "cannot disambiguate").
        if n_parseable < cfg.min_successes:
            last_err = {
                "type": "too_few_parseable_completions",
                "stage": cfg.name,
                "attempt": attempt,
                "texts": outs[:],
                "parsed_preview": [str(p)[:800] for p in parsed],
                "n_parseable": n_parseable,
                "min_successes": cfg.min_successes,
                "n": len(parsed),
            }
            continue

        return outs, parsed, None, (time.perf_counter() - t0_all), total_usage

    return (
        None,
        None,
        last_err or {"type": "unknown_stage_failure"},
        (time.perf_counter() - t0_all),
        total_usage,
    )


def finalize_row_times_and_tokens(r: Dict[str, Any]):
    # "extract" = selection + disambiguation + decomposition
    r["timing"]["extract_s"] = (
        r["timing"]["selection_s"]
        + r["timing"]["disambiguation_s"]
        + r["timing"]["decomposition_s"]
    )
    r["timing"]["total_s"] = r["timing"]["extract_s"] + r["timing"]["verify_s"]

    r["tokens"]["total_prompt"] = (
        r["tokens"]["selection_prompt"]
        + r["tokens"]["disambiguation_prompt"]
        + r["tokens"]["decomposition_prompt"]
        + r["tokens"]["verify_prompt"]
    )
    r["tokens"]["total_gen"] = (
        r["tokens"]["selection_gen"]
        + r["tokens"]["disambiguation_gen"]
        + r["tokens"]["decomposition_gen"]
        + r["tokens"]["verify_gen"]
    )


def add_flops(r: Dict[str, Any], params_b: float, flops_per_param: float):
    """
    Estimate FLOPs per row.

    Formula:  FLOPs = flops_per_param * P * T
      P = params_b * 1e9         (total model parameters)
      T = prompt_tokens + gen_tokens  (counted by vLLM token ids, stage by stage)
      flops_per_param = 2  (standard: one multiply-add ≈ 2 FLOPs per param per token)
                       = 6  (if you include backward; use 2 for inference)

    Stages:
      extract = selection + disambiguation + decomposition
      verify  = verification (claim-by-claim LLM calls)
    """
    if not params_b:
        r["flops"] = None
        return

    P = params_b * 1e9  # total parameter count
    k = flops_per_param

    # Token buckets: count both prompt and generated tokens per stage
    extract_tok = (
        r["tokens"]["selection_prompt"]
        + r["tokens"]["selection_gen"]
        + r["tokens"]["disambiguation_prompt"]
        + r["tokens"]["disambiguation_gen"]
        + r["tokens"]["decomposition_prompt"]
        + r["tokens"]["decomposition_gen"]
    )
    verify_tok = r["tokens"]["verify_prompt"] + r["tokens"]["verify_gen"]
    total_tok = extract_tok + verify_tok

    def flops_for(tok: int) -> float:
        # Each token requires k FLOPs per parameter (2 = one multiply-add)
        return k * P * float(tok)

    extract_flops = flops_for(extract_tok)
    verify_flops = flops_for(verify_tok)
    total_flops = flops_for(total_tok)

    def tflops_per_s(flops: float, seconds: float) -> Optional[float]:
        if seconds <= 0:
            return None
        return flops / seconds / 1e12

    r["flops"] = {
        "params_b": params_b,
        "k": k,
        "extract_tokens": int(extract_tok),
        "verify_tokens": int(verify_tok),
        "total_tokens": int(total_tok),
        "extract_flops": extract_flops,
        "verify_flops": verify_flops,
        "total_flops": total_flops,
        "extract_tflops_per_s": tflops_per_s(extract_flops, r["timing"]["extract_s"]),
        "verify_tflops_per_s": tflops_per_s(verify_flops, r["timing"]["verify_s"]),
        "total_tflops_per_s": tflops_per_s(total_flops, r["timing"]["total_s"]),
    }


def _is_eval_row(r: Dict[str, Any]) -> bool:
    # "evaluable" sentence for latency/claims/compute aggregates
    return (r.get("gold_supported") is not None) and (r.get("fail_reason") is None)


def _sum_time(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(float((r.get("timing") or {}).get(key, 0.0) or 0.0) for r in rows)


def _sum_flops(rows: List[Dict[str, Any]], key: str) -> float:
    # key: "extract_flops" | "verify_flops" | "total_flops"
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


def roc_auc_manual(y_true: List[int], y_score: List[float]) -> float:
    """ROC-AUC via Mann-Whitney U rank statistic (handles ties). Positive class = 1."""
    n = len(y_true)
    if n == 0 or len(y_score) != n:
        return 0.0
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    pairs = sorted(
        [(score, label, idx) for idx, (label, score) in enumerate(zip(y_true, y_score))],
        key=lambda t: t[0],
    )
    ranks = [0.0] * n
    i = 0
    next_rank = 1
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (next_rank + (next_rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            _, _, orig_idx = pairs[k]
            ranks[orig_idx] = avg_rank
        next_rank += j - i
        i = j
    sum_ranks_pos = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    u_pos = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u_pos / (n_pos * n_neg))
