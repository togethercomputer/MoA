# Mixture-of-Agents (MoA)

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/ArXiv-2406.04692-b31b1b.svg)](https://arxiv.org/abs/2406.04692)
[![Discord](https://img.shields.io/badge/Discord-Together%20AI-blue?logo=discord&logoColor=white)](https://discord.com/invite/9Rk6sSeWEG)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/togethercompute.svg?style=social&label=Follow%20%40togethercompute)](https://twitter.com/togethercompute)

<img alt="MoA architecture" src="./assets/moa.jpg">

<p align="center">
  <a href="#overview"><strong>Overview</strong></a> ·
  <a href="#quickstart-moa-in-50-loc"><strong>Quickstart</strong></a> ·
  <a href="#multi-layer-moa-example"><strong>Advanced example</strong></a> ·
  <a href="#interactive-cli-demo"><strong>Interactive CLI Demo</strong></a>
  ·
  <a href="#evaluation"><strong>Evaluation</strong></a>
  ·
  <a href="#results"><strong>Results</strong></a>
  .
  <a href="#credits"><strong>Credits</strong></a>
</p>

## Overview

Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results. By employing a layered architecture where each layer comprises several LLM agents, **MoA significantly outperforms GPT-4 Omni’s 57.5% on AlpacaEval 2.0 with a score of 65.1%**, using only open-source models!

## Quickstart: MoA in 50 LOC

To get to get started with using MoA in your own apps, see `moa.py`. In this simple example, we'll use 2 layers and 4 LLMs. You'll need to:

1. Install the Together Python library: `pip install together`
2. Get your [Together API Key](https://api.together.xyz/settings/api-keys) & export it: `export TOGETHER_API_KEY=`
3. Run the python file: `python moa.py`

<img alt="MoA explained" src="./assets/moa-explained.png">

## Multi-layer MoA Example

In the previous example, we went over how to implement MoA with 2 layers (4 LLMs answering and one LLM aggregating). However, one strength of MoA is being able to go through several layers to get an even better response. In this example, we'll go through how to run MoA with 3+ layers in `advanced-moa.py`.

```python
python advanced-moa.py
```

<img alt="MoA – 3 layer example" src="./assets/moa-3layer.png">

## Interactive CLI Demo

This interactive CLI demo showcases a simple multi-turn chatbot where the final response is aggregated from various reference models.

To run the interactive demo, follow these 3 steps:

1. Export Your API Key: `export TOGETHER_API_KEY={your_key}`
2. Install Requirements: `pip install -r requirements.txt`
3. Run the script: `python bot.py`

The CLI will prompt you to input instructions interactively:

1. Start by entering your instruction at the ">>>" prompt.
2. The system will process your input using the predefined reference models.
3. It will generate a response based on the aggregated outputs from these models.
4. You can continue the conversation by inputting more instructions, with the system maintaining the context of the multi-turn interaction.

### [Optional] Additional Configuration

The demo will ask you to specify certain options but if you want to do additional configuration, you can specify these parameters:

- `--aggregator`: The primary model used for final response generation.
- `--reference-models`: Models used as references.
- `--temperature`: Controls the randomness of the response generation.
- `--max-tokens`: Maximum number of tokens in the response.
- `--rounds`: Number of rounds to process the input for refinement. (num rounds == num of MoA layers - 1)
- `--num-proc`: Number of processes to run in parallel for faster execution.
- `--multi-turn`: Boolean to toggle multi-turn interaction capability.

Specify `--reference-models` multiple times to use multiple models as references. For example:

```bash
# Specify multiple reference models
python bot.py --reference-models "mistralai/Mixtral-8x22B-Instruct-v0.1" --reference-models "Qwen/Qwen2-72B-Instruct"
```

## Other OpenAI Compatible API endpoints

To use different OpenAI-compatible API endpoints, set the OPENAI_BASE_URL and OPENAI_API_KEY variable.

```
export TOGETHER_API_KEY=
export OPENAI_BASE_URL="https://your-api-provider.com/v1"
export OPENAI_API_KEY="your-api-key-here"
```

This way, any 3rd party API can be used, such as OpenRouter, Groq, local models, etc.

### Ollama

For example, to run the bot using Ollama:

1. Set up the environment:

```
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
```

2. Run the bot command:

```
python bot.py --model llama3 --reference-models llama3 --reference-models mistral
```

## Evaluation

We provide scripts to quickly reproduce some of the results presented in our paper
For convenience, we have included the code from [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval),
[MT-Bench](https://github.com/lm-sys/FastChat), and [FLASK](https://github.com/kaistAI/FLASK), with necessary modifications.
We extend our gratitude to these projects for creating the benchmarks.

### Preparation

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

### Run AlpacaEval 2

To run AlpacaEval 2, execute the following scripts:

```
bash run_eval_alpaca_eval.sh
```

### Run MT-Bench

For a minimal example of MT-Bench evaluation, run:

```
bash run_eval_mt_bench.sh
```

### Run FLASK

For a minimal example of FLASK evaluation, run:

```
bash run_eval_flask.sh
```

### Results

<div align="center">
  <img src="assets/alpaca_and_mtbench.png" alt="alpaca_mtbench" style="width: 100%; display: block; margin-left: auto; margin-right: auto;" />
  <br>
</div>

We achieved top positions on both the AlpacaEval 2.0 leaderboard and MT-Bench. Notably, on AlpacaEval 2.0, using solely open-source models, we achieved a margin of 7.6% absolute improvement from 57.5% (GPT-4 Omni) to 65.1% (MoA).

<div align="center">
  <img src="assets/flask.png" alt="flask" style="width: 50%; display: block; margin-left: auto; margin-right: auto;" />
  <br>
</div>

FLASK offers fine-grained evaluation of models across multiple dimensions. Our MoA method significantly outperforms the original Qwen1.5-110B-Chat on harmlessness, robustness, correctness, efficiency, factuality, commonsense, insightfulness, completeness. Additionally, MoA also outperforms GPT-4 Omni in terms of correctness, factuality, insightfulness, completeness, and metacognition.

Please feel free to contact us if you have difficulties in reproducing the results.

## Credits

Notably, this work was made possible by the collaborative spirit and contributions of active organizations in the AI field. We appreciate the efforts of Meta AI, Mistral AI, Microsoft, Alibaba Cloud, and DataBricks for developing the Llama 3, Mixtral, WizardLM 2, Qwen 1.5, and DBRX models. Additionally, we extend our gratitude to Tatsu Labs, LMSYS, and KAIST AI for developing the AlpacaEval, MT-Bench, and FLASK evaluation benchmarks.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{wang2024mixture,
  title={Mixture-of-Agents Enhances Large Language Model Capabilities},
  author={Wang, Junlin and Wang, Jue and Athiwaratkun, Ben and Zhang, Ce and Zou, James},
  journal={arXiv preprint arXiv:2406.04692},
  year={2024}
}
```
