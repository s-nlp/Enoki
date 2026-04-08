# RefChecker

Бейзлайн RefChecker для **FactBench** и **FELM** с использованием офлайн-доказательств.

## Пайплайн
1. Запустить LLM-экстрактор RefChecker для каждого предложения.
2. Запустить чекер RefChecker на офлайн-референсе.
3. Строго агрегировать предсказания на уровне предложения: предложение считается `supported` только если каждое извлеченное утверждение размечено как `Entailment`.

Раннер записывает `metrics.json` и `segments.jsonl` в `--out_root`, включая задержку по каждому предложению: `extract_s`, `verify_s` и `total_s`.

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
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --no_claim_policy_all penalize
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
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --no_claim_policy_all penalize
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
  --batch_size_extractor 8 \
  --batch_size_checker 8 \
  --no_claim_policy_all penalize
```

Или используйте готовый скрипт:
```bash
bash run_llama31_factbench_local.sh
```

Для короткого тестового запуска:
```bash
MAX_SAMPLES=10 bash run_llama31_factbench_local.sh
```

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
  --extractor_api_base http://127.0.0.1:5000/v1 \
  --checker_api_base http://127.0.0.1:5000/v1
```
