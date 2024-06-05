# Together Model

## Quickstart
We provide some scripts to quickly recreate some of the results in our paper.
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

## Interactive Demo
This interactive demo demonstrates a multi-turn conversation environment where inputs from users will be responded by a cohesive response aggregated from various reference models.

### Setup
Export your Together API key as an environment variable:
```bash
export TOGETHER_API_KEY={your_key}
```
Ensure you also have the `utils` module that includes the functions used in the script.


### Running the Demo

To run the interactive demo, navigate to the directory containing the script and execute it using Python:

```bash
python interactive_demo.py
```

The script will prompt you to input instructions interactively. Here's how to use it:

1. Start by entering your instruction at the "Input:" prompt.
2. The system will process your input using the predefined reference models.
3. It will generate a response based on the aggregated outputs from these models.
4. You can continue the conversation by inputting more instructions, with the system maintaining the context of the multi-turn interaction.

### Features

- **Multi-Model Aggregation:** Leverages multiple AI models to generate responses, improving the robustness and diversity of the output.
- **Multi-Turn Interaction:** Maintains conversation context over multiple turns, simulating a more natural dialogue flow.
- **Customizability:** Easily configurable to use different sets of models or adjust parameters like temperature and max tokens.

### Configuration

You can configure the demo by modifying the parameters in the `main` function call at the bottom of the script:
- `--aggregator`: The primary model used for final response generation.
- `--reference_models`: List of models used as references.
- `--temperature`: Controls the randomness of the response generation.
- `--max_tokens`: Maximum number of tokens in the response.
- `--rounds`: Number of rounds to process the input for refinement.
- `--num_proc`: Number of processes to run in parallel for faster execution.
- `--multi_turn`: Boolean to toggle multi-turn interaction capability.

Enjoy the interactive demo and explore the capabilities of Mixture of Agents!