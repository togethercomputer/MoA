# Together Model

## Quickstart

### Preparation:

```bash
# install requirements
pip install -r requirements.txt
cd alpaca_eval
pip install -e .
cd FastChat
pip install -e ".[model_worker,llm_judge]"
cd ..

# setup api keys
export TOGETHER_API_KEY=<TOGETHER_API_KEY>
export OPENAI_API_KEY=<OPENAI_API_KEY>
```

### AlpacaEval

```
bash run_generation.sh
bash run_eval.sh
```

### MT-Bench

a minimal example:
```
bash run_eval_mt_bench.sh
```

### FLASK

a minimal example:
```
bash run_eval_flask.sh
```
