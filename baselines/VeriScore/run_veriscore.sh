export CUDA_VISIBLE_DEVICES=3
# FactBench
python veriscore_run.py factbench \
  --data_jsonl ../factcheck-GPT-benchmark.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_dir ./out/veriscore/qwen25_factbench \
  --veriscore_assets_dir ./assets/veriscore \
  --claims_source llm \
  --max_claims 4 \
  --empty_claims_policy supported

# FactBench Llama
python veriscore_run.py factbench \
  --data_jsonl ../factcheck-GPT-benchmark.jsonl \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --alpaca_template ./alpaca_template.txt \
  --out_dir ./out/veriscore/llama_factbench \
  --veriscore_assets_dir ./assets/veriscore


# FELM
python veriscore_run.py felm \
  --felm_dir /path/to/felm_on_disk \
  --subset writing_rec \
  --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_dir ./out/veriscore/qwen25_felm \
  --veriscore_assets_dir ./assets/veriscore \
  --skip_no_context \
  --empty_claims_policy supported
# FELM Llama
python veriscore_run.py felm \
  --felm_dir /path/to/felm_on_disk \
  --subset writing_rec \
  --split test \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --alpaca_template ./alpaca_template.txt \
  --out_dir ./out/veriscore/llama_felm \
  --veriscore_assets_dir ./assets/veriscore \
  --skip_no_context