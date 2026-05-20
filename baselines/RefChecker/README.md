# RefChecker

Бейзлайн RefChecker для **FactBench** и **FELM** с использованием офлайн-доказательств.

## Пайплайн
1. Запустить LLM-экстрактор RefChecker для каждого предложения.
2. Запустить чекер RefChecker на офлайн-референсе.
3. Строго агрегировать предсказания на уровне предложения: предложение считается `supported` только если каждое извлеченное утверждение размечено как `Entailment`.

Раннер записывает `metrics.json` и `segments.jsonl` в `--out_root`, включая задержку по каждому предложению: `extract_s`, `verify_s` и `total_s`. Метрики включают macro-F1 с положительным классом `not_supported` и `roc_auc_not_supported`, где score = доля claims с не-`Entailment` вердиктом.

Если передать `--model_params_b`, раннер пишет две compute-оценки:
- `visible_*`: нижняя граница по видимому тексту (`sentence/question/reference/claims/labels`) c более реалистичной сериализацией claims;
- основные поля без суффикса: `adjusted` proxy-оценка, если заданы `--compute_*_overhead_tokens`, иначе совпадают с `visible_*`.

Это по-прежнему **оценка**, потому что RefChecker не возвращает реальные provider token usage и внутренние prompt tokens.

`--no_claim_policy_all` больше не нужен как отдельная идея RefChecker: это был локальный флаг раннера для неопределенных предложений. Используйте более явный `--undefined_prediction_policy`:
- `penalize`: считать предложения без строгого предсказания `not_supported` (рекомендуется для вашей строгой логики sentence supported iff every claim is supported);
- `skip`: исключить такие предложения из метрик.

`--claim_format` позволяет запросить `triplet` или `subsentence`. Это особенно полезно для sentence-level задач. Если установленная версия `refchecker` не поддерживает `subsentence`, раннер автоматически откатится к дефолтному формату и зафиксирует это в `metrics.json` через `claim_format_requested`, `claim_format_effective` и `claim_format_note`.

## Установка
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Для LLM-экстракторов и чекеров настройте креды провайдера, который используется через LiteLLM, например `OPENAI_API_KEY` для моделей OpenAI. Для локальной модели используйте `--run_local_vllm`: раннер сам поднимет совместимый с OpenAI vLLM-эндпоинт, дождется готовности, запустит RefChecker и остановит сервер после завершения.

## Запуск
### FELM
```bash
python run_refchecker.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/refchecker_felm_wk \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format triplet \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize
```

### FELM sentence-level variant
Для более честного сравнения с sentence-level пайплайнами вроде Claimify/SAFE/VeriScore попробуйте:
```bash
python run_refchecker.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/refchecker_felm_wk_subsentence \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format subsentence \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize
```

Если хотите не только lower bound FLOPs, но и proxy-оценку с явным prompt-overhead:
```bash
python run_refchecker.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/refchecker_felm_wk_subsentence_proxy_compute \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format subsentence \
  --compute_extract_prompt_overhead_tokens 120 \
  --compute_verify_prompt_overhead_tokens 220 \
  --compute_verify_per_claim_overhead_tokens 12 \
  --undefined_prediction_policy penalize
```

### FactBench
```bash
python run_refchecker.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./out/refchecker_factbench \
  --extractor_name gpt-4o \
  --checker_type llm \
  --checker_name gpt-4o \
  --claim_format triplet \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize
```

### FactBench + локальная Llama 3.1
Раннер сам поднимет локальный vLLM-эндпоинт:
```bash
python run_refchecker.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./out/refchecker_factbench_llama31_local \
  --run_local_vllm \
  --local_vllm_model VityaVitalich/Llama3.1-8b-instruct \
  --checker_type llm \
  --claim_format triplet \
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --undefined_prediction_policy penalize \
  --model_params_b 8 \
  --flops_per_param 2
```

Или используйте готовый скрипт:
```bash
bash run_llama31_factbench_local.sh
```

Для короткого тестового запуска:
```bash
MAX_SAMPLES=10 bash run_llama31_factbench_local.sh
```

Можно вынести параметры в `.env`:
```bash
cp .env.example .env
set -a; source .env; set +a
bash run_llama31_factbench_local.sh
```

### FELM + локальная Llama 3.1
```bash
SUBSET=wk MAX_EXAMPLES=10 bash run_llama31_felm_local.sh
```

Полный запуск по нужному подмножеству:
```bash
SUBSET=wk bash run_llama31_felm_local.sh
```

Доступные локальные переменные: `MODEL`, `FELM_DIR`, `SUBSET`, `SPLIT`, `OUT_ROOT`, `BATCH_SIZE_EXTRACTOR`, `BATCH_SIZE_CHECKER`, `EXTRACTOR_MAX_NEW_TOKENS`, `MAX_REFERENCE_SEGMENT_LENGTH`, `UNDEFINED_PREDICTION_POLICY`, `MODEL_PARAMS_B`, `FLOPS_PER_PARAM`, `TOKENIZER_NAME`, `LOCAL_VLLM_PORT`, `LOCAL_VLLM_GPU_MEMORY_UTILIZATION`, `LOCAL_VLLM_TENSOR_PARALLEL_SIZE`, `LOCAL_VLLM_MAX_MODEL_LEN`, `LOCAL_VLLM_STARTUP_TIMEOUT_S`.

### Уже запущенный vLLM-эндпоинт
```bash
python run_refchecker.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./out/refchecker_felm_wk_local \
  --extractor_name openai/meta-llama/Meta-Llama-3-8B-Instruct \
  --checker_type llm \
  --checker_name openai/meta-llama/Meta-Llama-3-8B-Instruct \
  --claim_format triplet \
  --extractor_api_base http://127.0.0.1:5000/v1 \
  --checker_api_base http://127.0.0.1:5000/v1
```
