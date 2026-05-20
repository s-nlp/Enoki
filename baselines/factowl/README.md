# FactOwl (бейзлайн FactOwl)

Оценка фактичности на основе FactOwl для **FactBench** и **FELM** с использованием только офлайн-доказательств.

**Источники доказательств**
- FactBench: `auto_evidence`, `auto_evidence_url`, `human_evidence` по всему сэмплу.
- FELM: `ref_text` (для каждого примера).

## Заметки
- Этот раннер ожидает, что Python-пакет `factowl` установлен.
- Он патчит промпт атомарного экстрактора FactOwl во время выполнения (см. `--atomic_template` и `--atomic_set_examples`).

## Установка
Типовая настройка (адаптируйте под свое окружение):
```bash
pip install git+https://github.com/s-nlp/factowl.git
pip install jieba
python -m spacy download en_core_web_sm
```

## Запуск
### FactBench
```bash
python factowl_run.py factbench \
  --data ../data/factcheck-GPT-benchmark.jsonl \
  --out_root ./cachedir_factowl_factbench \
  --model Qwen/Qwen3-14B \
  --gpu_memory_utilization 0.5 \
  --verbose_patch \
  --atomic_set_examples
```

### FELM
```bash
python factowl_run.py felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --out_root ./cachedir_factowl_felm \
  --model Qwen/Qwen3-14B \
  --gpu_memory_utilization 0.5 \
  --verbose_patch
```

## Выходные файлы
- FactBench: `out_root/metrics.json`, `segments_with_factowl.jsonl`, `examples_with_factowl.jsonl`.
- FELM: `out_root/<subset>/<split>/metrics.json`, `segments_with_factowl.jsonl`, `examples_with_factowl.jsonl`.
