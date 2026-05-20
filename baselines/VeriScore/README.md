# VeriScore (VeriScore-подобный бейзлайн)

VeriScore-подобная оценка фактичности для **FactBench** и **FELM**, использующая только **офлайн**-доказательства.

**Источники доказательств**
- FactBench: `auto_evidence`, `auto_evidence_url`, `human_evidence` по всему сэмплу.
- FELM: `ref_text` (для каждого примера).

## Пайплайн
1. Загрузить сэмплы и предложения, пропустить `sentence_factuality_label == NA`.
2. Построить пассажи доказательств (только офлайн).
3. Извлечь утверждения.
4. Выполнить retrieval для каждого утверждения с помощью BM25 по пассажам (при необходимости fallback к первым `topk`).
5. Выполнить верификацию через бинарный промпт VeriScore.
6. Агрегировать по предложениям.

**Источники утверждений**
- `llm` (по умолчанию): извлекать утверждения с помощью шаблона.
- `sentence`: использовать само предложение как одно утверждение.
- `dataset`: использовать `sentences[*].claims`, если они есть (только FactBench).

**Метки верификации**
- `Supported` или `Unsupported` (парсятся в `supported` / `not_supported`).

## Ассеты
Передайте один из вариантов:
- `--veriscore_assets_dir` (автоматически подставляет стандартные пути), или
- явные `--extraction_template`, `--verification_instruction_binary`, `--fewshot_jsonl`.

Ожидаемые имена файлов ассетов внутри `--veriscore_assets_dir`:
- `prompt/non_qa_template.txt`
- `prompt/verification_instruction_binary.txt`
- `data/demos/few_shot_examples.jsonl`

## Выходные файлы
`--out_dir` содержит `metrics.json` и `segments_with_veriscore.jsonl`.

## Установка
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```

## Запуск
### FactBench
```bash
python veriscore_run.py factbench \
  --data_jsonl ../data/factcheck-GPT-benchmark.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_dir ./out/veriscore/qwen25_factbench \
  --veriscore_assets_dir ./assets/veriscore \
  --claims_source llm \
  --max_claims 4 \
  --empty_claims_policy supported
```

### FELM
```bash
python veriscore_run.py felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_dir ./out/veriscore/qwen25_felm \
  --veriscore_assets_dir ./assets/veriscore \
  --skip_no_context \
  --empty_claims_policy supported
```
