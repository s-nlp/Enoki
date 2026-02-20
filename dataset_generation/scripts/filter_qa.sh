export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://99dgx-16:8128/v1/"

# python qa_filter.py --input data/wiki/en/questions_long.jsonl --out data/wiki/en/questions_long.filtered.jsonl --errors-out data/wiki/en/questions_long.filtered.errors.jsonl --min-overlap 2 --min-overlap-ratio 0.25

python qa_filter.py --no-rule-prefilter --input data/wiki/en/questions_long.jsonl \
--out data/wiki/en/questions_long.llm_filtered.jsonl \
--errors-out data/wiki/en/questions_long.llm_filtered.errors.jsonl \
--base-url "http://99dgx-16:8128/v1/" --api-key EMPTY --model openai/gpt-oss-120b --concurrency 12 \
--max-tokens 16000

    
# python qa_filter.py --no-rule-prefilter --input data/wiki/en/hallucinations/answers.jsonl --out data/wiki/en/hallucinations/answers.llm_filtered.jsonl --errors-out data/wiki/en/hallucinations/answers.llm_filtered.errors.jsonl --eval-target answer --base-url "http://99dgx-16:8128/v1/" --model openai/gpt-oss-120b --concurrency 12

# • Changes made:

#   - Rebuilt qa_filter.py to be LLM-based (AsyncOpenAI, default openai/gpt-oss-120b). Separate evaluation prompts for questions and for question+answer, few-shot examples included. Supports context-level or per-question inputs, concurrency, max-retries, optional regex prefilter. Keeps accepted rows, writes rejects with LLM reasons.
#   - Added dataset_stats.py to run NER with the same model, compute length/entity distributions for questions, contexts, and answers, and optionally plot histograms (matplotlib optional).

#   Usage (examples):

#   - LLM filter for questions (context-level from qa_gen_fast.py):
#     
#   - LLM filter for question+answer records (from hallucination_answerer.py):
#     python qa_filter.py --input data/wiki/en/answers.jsonl --out data/wiki/en/answers.llm_filtered.jsonl --errors-out data/wiki/en/answers.llm_filtered.errors.jsonl --eval-target answer --base-url http://localhost:8000/v1 --model openai/gpt-oss-120b
#   - Stats + NER:
#     python dataset_stats.py --input data/wiki/en/questions_long.llm_filtered.jsonl --out-json logs/stats.json --plot-prefix logs/stats --base-url http://localhost:8000/v1 --model openai/gpt-oss-120b --concurrency 8

#   Notes:

#   - Install deps if needed: pip install openai matplotlib orjson (orjson optional).
#   - --plot-prefix saves PNGs for length/entity-count histograms; skipped if matplotlib absent.
# #   - Adjust --max-text-chars in stats to cap context length sent to NER; adjust --temperature/--max-tokens as desired.