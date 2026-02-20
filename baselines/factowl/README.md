```bash
git clone git@github.com:s-nlp/factowl.git
mv factowl_FELM.py factowl/
mv felm_with_ref_text factowl/
cd factowl && cd factowl && pip install -e .
pip install jieba
python -m spacy download en_core_web_sm
cd ..
```

run:
```bash
python factowl_FELM.py \
--felm_dir ./felm_with_ref_text \
--subset wk \
--split test \
--model Qwen/Qwen3-8B \
--gpu_memory_utilization 0.5 \
--out_root ./factowl_felm_qwen3_8b
```

factowl_FELM.py должен быть в главное папке библиотеки factowl вместе с данными
