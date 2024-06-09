# Mixture of Agents

## Overview

Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results. 
By employing a layered architecture where each layer comprises several LLM agents, MoA significantly outperforms GPT-4 Omni’s 57.5% on AlpacaEval 2.0 with a score of 65.1%, using only open-source models!

## Interactive Demo

The interactive demo showcases a simple multi-turn chatbot where the response is aggregated from various reference models.

### Setup

1. Export Your API Key:

   Ensure you have your Together API key and export it as an environment variable:

   ```bash
   export TOGETHER_API_KEY={your_key}
   ```

2. Install Requirements:
   
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

To run the interactive demo, execute the following script with Python:

```bash
python bot.py
```

The script will prompt you to input instructions interactively. Here's how to use it:

1. Start by entering your instruction at the ">>>" prompt.
2. The system will process your input using the predefined reference models.
3. It will generate a response based on the aggregated outputs from these models.
4. You can continue the conversation by inputting more instructions, with the system maintaining the context of the multi-turn interaction.
5. enter `exit` to exit the chatbot.

### Configuration

You can configure the demo by specifying the following parameters:

- `--aggregator`: The primary model used for final response generation.
- `--reference_models`: List of models used as references.
- `--temperature`: Controls the randomness of the response generation.
- `--max_tokens`: Maximum number of tokens in the response.
- `--rounds`: Number of rounds to process the input for refinement. (num rounds == num of MoA layers - 1)
- `--num_proc`: Number of processes to run in parallel for faster execution.
- `--multi_turn`: Boolean to toggle multi-turn interaction capability.

## Evaluation Benchmarks

We provide scripts to quickly recreate some of the results presented in our paper
For convinence, we have included the code from [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval), 
[MT-Bench](https://github.com/lm-sys/FastChat), and [FLASK](https://github.com/kaistAI/FLASK), with necessary modifications.
We extend our gratitude to these projects for creating the benchmarks.

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

### AlpacaEval 2

To run AlpacaEval 2, execute the following scripts:

```
bash run_generation.sh
bash run_eval.sh
```

### MT-Bench

For a minimal example of MT-Bench evaluation, run:

```
bash run_eval_mt_bench.sh
```

### FLASK

For a minimal example of FLASK evaluation, run:

```
bash run_eval_flask.sh
```

## License

All code in this repository was developed by Together Computer except where otherwise noted. Copyright (c) 2023, Together Computer. All rights reserved. The code is licensed under the Apache 2.0 license.

```
Copyright 2023 Together Computer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

This repository also contains code written by a number of other authors. Such contributions are marked and the relevant licensing is included where appropriate.

For full terms, see the LICENSE file. If you have any questions, comments, or concerns about licensing please [contact us](https://www.together.ai/contact).

## Citation

If you find this work helpful, please consider citing:

```
@article{wang2024mixture,
  title={Mixture-of-Agents Enhances Large Language Model Capabilities},
  author={Wang, Junlin and Wang, Jue and Athiwaratkun, Ben and Zhang, Ce and Zou, James},
  journal={arXiv preprint arXiv:2406.xxxxx},
  year={2024}
}
```