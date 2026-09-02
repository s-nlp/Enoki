# Single-step vs. Enoki: latency/FLOPs, all 4 datasets (t=0.5, uncalibrated)

Qwen3.6-35B-A3B backbone for Single-step and Enoki-LLM (active params 3e9 used for FLOPs).

## RAGTruth (900 ex / 5910 sent)

| Method | Span F1 | P | R | Latency/sent | Latency/example | FLOPs/sent |
|---|---|---|---|---|---|---|
| Single-step (ZS) | 37.39 | 0.281 | 0.560 | ≈4.190s* | 27.415s | ≈7.31e11* |
| Enoki-LLM | 26.34 | 0.159 | 0.770 | 6.5112s | 42.757s | 6.347e12 |
| Enoki-Rule | 21.68 | 0.129 | 0.689 | 0.0516s | 0.339s | 0.000e+00 |
| Enoki-Encoder | 23.34 | 0.141 | 0.668 | 0.1351s | 0.887s | 2.265e10 |

## PsiloQA (1098 ex / 3741 sent)

| Method | Span F1 | P | R | Latency/sent | Latency/example | FLOPs/sent |
|---|---|---|---|---|---|---|
| Single-step (ZS) | 39.07 | 0.443 | 0.350 | ≈3.639s* | 12.397s | ≈9.22e11* |
| Enoki-LLM | 71.45 | 0.673 | 0.761 | 12.3014s | 41.912s | 6.481e12 |
| Enoki-Rule | 65.74 | 0.619 | 0.701 | 0.0434s | 0.148s | 0.000e+00 |
| Enoki-Encoder | 64.06 | 0.639 | 0.643 | 0.1609s | 0.548s | 2.594e10 |

## Mushroom (154 ex / 403 sent)

| Method | Span F1 | P | R | Latency/sent | Latency/example | FLOPs/sent |
|---|---|---|---|---|---|---|
| Single-step (ZS) | 8.92 | 0.128 | 0.069 | ≈4.103s* | 10.738s | ≈5.37e12* |
| Enoki-LLM | 51.99 | 0.519 | 0.521 | 10.0375s | 26.267s | 6.392e12 |
| Enoki-Rule | 47.29 | 0.472 | 0.474 | 0.2742s | 0.718s | 0.000e+00 |
| Enoki-Encoder | 45.15 | 0.510 | 0.405 | 0.4165s | 1.090s | 2.269e10 |

## HalluEntity (157 ex / 1237 sent, entity-level — AUROC/AUPRC instead of F1, threshold not involved)

| Method | AUROC | AUPRC | Latency/sent | Latency/example | FLOPs/sent |
|---|---|---|---|---|---|
| Single-step (ZS) | 74.04 | 37.51 | ≈3.181s* | 25.066s | ≈3.02e12* |
| Enoki-LLM | 79.94 | 56.20 | 14.9017s | 117.410s | 6.599e12 |
| Enoki-Rule | 76.55 | 47.45 | 0.5765s | 4.542s | 0.000e+00 |
| Enoki-Encoder | 70.44 | 44.21 | 0.5011s | 3.948s | 2.789e10 |

## Notes

\* Derived: Single-step is native per-example (one LLM call per whole response); Enoki numbers are
native per-sentence. Divided by sentences/example for that dataset (RAGTruth ÷6.567, PsiloQA ÷3.407,
Mushroom ÷2.617, HalluEntity ÷7.879) to make the two granularities comparable. State this conversion
explicitly wherever this table is used — don't silently mix per-example and per-sentence numbers.
