# Measured FLOPs/sentence — all baselines, all datasets

| Method | Model | Pipeline | LLM calls/sentence | FactCheck-Bench | ANAH | RAGTruth |
|---|---|---|---|---|---|---|
| FactOwl | Qwen3-8B | Single call into FactOwl's internal atomic-fact extractor + scorer. | opaque‡ | 5.84e13 | 3.43e13 | 7.13e13 |
| SAFE | Qwen3-8B | Atomic fact extraction (1/sentence) → per-fact revision (1/fact) → per-fact relevance filter (1/fact) → verification of relevant facts only (1/fact). | ≈8.3 | 1.47e14 | 7.17e13 | 1.165e14 |
| VeriScore | Qwen3-8B | Sentence-level claim extraction (1/sentence) → evidence-conditioned verification (1/claim). | ≈3.1 | 3.05e14 | 2.55e14 | 3.06e14 |
| Claimify | Qwen3-8B | Selection (verifiability gate, self-consistency n=3) → disambiguation (self-consistency n=3) → decomposition (1 call) → verification (1/claim); pipeline stops early if any gate fails. | ≈3.7§ | 2.63e14† | 2.21e14 | 1.95e14 |
| RefChecker | Qwen3-8B | Claim extraction (1/sentence) → per-claim entailment check (1/claim). | ≈3.9‡ | 2.44e13* | 3.67e12* | 1.47e13* |

Units: FLOPs per sentence (2·P·T, P = model params, T = prompt+gen tokens for that sentence).

"LLM calls/sentence" = avg. number of distinct prompts sent to the model per sentence, measured on FactCheck-Bench (277 gold rows), counting each pipeline stage invocation once even where a stage is skipped for some sentences (e.g. Claimify's disambiguation only fires for ~90% of sentences) — so these are dataset-specific empirical averages, not fixed constants; ANAH/RAGTruth differ slightly because avg. claims/facts per sentence differ per dataset.

‡ FactOwl and RefChecker call into third-party libraries that make their own internal LLM calls; this codebase only sees the aggregate token usage FactOwl/RefChecker report back, not the call count, so "opaque"/"‡" mean the true number of underlying completions isn't observable here. The RefChecker figure (≈3.9) is a lower bound from its *visible* extract/verify calls only — same caveat as the FLOPs column (see \* below).

§ Selection and disambiguation are each **one prompt with n=3 sampled completions** (self-consistency + majority vote), not three separate calls — vLLM shares the prompt prefill across the 3 samples, so only the decode (generation) cost triples, and the FLOPs column already accounts for this correctly (prompt tokens counted once, gen tokens summed over all 3 samples). So Claimify's real generation count is higher than its prompt count: ≈3.7 prompts/sentence but ≈5.5 sampled generations/sentence when the full pipeline runs (1×3 + 1×3 + 1×1 + ~1×1 per claim). This is why its FLOPs/sentence look comparable to methods with fewer, larger single calls (e.g. VeriScore's one verify call per claim carries ~18.5K tokens of retrieved evidence) — call *count* isn't what drives FLOPs, total tokens is.

\* RefChecker-Qwen3-8B token counts are the runner's own `visible_structured_lower_bound` estimate, not exact vLLM token ids like every other row — RefChecker hides its internal prompts, so this is a **lower bound** on real FLOPs, not a directly comparable measurement. Everything else in this table is exact.