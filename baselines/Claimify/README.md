# Claimify (бейзлайн Claimify)

Извлечение утверждений в стиле Claimify с опциональной офлайн-верификацией для **FactBench** и **FELM**.

## Пайплайн
1. Этап Selection (rewrite или detector).
2. Этап Disambiguation.
3. Этап Decomposition для получения атомарных утверждений.
4. Опциональная верификация по офлайн-доказательствам.

**Источники доказательств (только для верификации)**
- FactBench: `auto_evidence`, `auto_evidence_url`, `human_evidence`.
- FELM: `ref_text`.

**Промпты**
Промпты читаются из `settings.py` в этой папке.

## Бэкенды
- `vllm` (по умолчанию)
- `openrouter`
- `openai`

## Выходные файлы
`--out_root` содержит `metrics.json` и `segments.jsonl`.

## Установка
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```

## Запуск
### FactBench (vLLM)
```bash
python run_claimify.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./out/claimify_factbench \
  --backend vllm \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --do_verify \
  --evidence_scope sentence \
  --evidence_mode bm25 \
  --bm25_query claim \
  --selection_mode rewrite \
  --no_claim_policy_all penalize
```

### FELM (vLLM)
```bash
python run_claimify.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/claimify_felm_wk \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --do_verify \
  --evidence_mode bm25 \
  --bm25_query claim \
  --selection_mode rewrite \
  --no_claim_policy_all penalize
```

### OpenRouter / OpenAI
Установите `OPENROUTER_API_KEY` или `OPENAI_API_KEY` и переключите `--backend` на `openrouter` или `openai`.
