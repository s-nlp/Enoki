# SAFE (SAFE-подобный бейзлайн)

SAFE-подобный пайплайн оценки фактичности для **FactBench** и **FELM**, использующий только офлайн-доказательства из датасета.

**Источники доказательств**
- FactBench: `auto_evidence`, `auto_evidence_url`, `human_evidence` по всему сэмплу.
- FELM: `ref_text` (для каждого примера).

## Пайплайн
1. Построить контекст доказательств.
2. Выполнить атомарное извлечение для каждого предложения.
3. Деконтекстуализировать каждый атом с использованием полного ответа.
4. Проверить релевантность: `[Foo]` против `[Not Foo]`.
5. Выполнить верификацию: `[Supported]` против `[Not Supported]`.
6. Агрегировать по предложениям: любой `not_supported` => предложение `not_supported`; иначе если есть хотя бы один `supported` => `supported`; иначе `ir`.

**Политика FAIL**
- `empty_segment`, `no_context`, `no_atoms_or_abstain`, `exception`.
- FAIL всегда считается **ошибкой** в метриках `all`.
- `ir` **не** считается FAIL; при этом он все равно считается ошибкой в метриках `all`.

## Выходные файлы
- FactBench: `out_root/factbench/metrics.json`, `segments_with_safe_like.jsonl`, `examples_with_safe_like.jsonl`.
- FELM: `out_root/felm/<subset>/<split>/metrics.json`, `segments_with_safe_like.jsonl`, `examples_with_safe_like.jsonl`.

## Установка
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```

## Запуск
### FactBench
```bash
python safe_run.py \
  --dataset factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --model_params_b 8 \
  --flops_per_param 2 \
  --out_root ./cachedir_safe_fact \
  --gpu_memory_utilization 0.5
```

### FELM
```bash
python safe_run.py \
  --dataset felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_root ./cachedir_safe_felm \
  --gpu_memory_utilization 0.5
```
