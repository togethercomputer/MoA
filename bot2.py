import json
import datasets
from functools import partial
from typing import List
from loguru import logger
import os
import argparse
from utils import (
    generate_together_stream,
    generate_with_references,
    DEBUG,
)
import typer
from rich import print

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console(color_system="standard")

welcome_message = """
# Welcome to the Together AI MoA (Mixture-of-Agents) interactive demo!

This demo uses the following LLMs as reference models:
- Qwen/Qwen2-72B-Instruct
- Qwen/Qwen1.5-72B-Chat
- mistralai/Mixtral-8x22B-Instruct-v0.1
- databricks/dbrx-instruct

The results are then used by our aggregate model, Qwen/Qwen2-72B-Instruct.

"""

default_reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]


def process_fn(
    item,
    temperature=0.7,
    max_tokens=2048,
):
    """
    Processes a single item (e.g., a conversational turn) using specified model parameters to generate a response.

    Args:
        item (dict): A dictionary containing details about the conversational turn. It should include:
                     - 'references': a list of reference responses that the model may use for context.
                     - 'model': the identifier of the model to use for generating the response.
                     - 'instruction': the user's input or prompt for which the response is to be generated.
        temperature (float): Controls the randomness and creativity of the generated response. A higher temperature
                             results in more varied outputs. Default is 0.7.
        max_tokens (int): The maximum number of tokens to generate. This restricts the length of the model's response.
                          Default is 2048.

    Returns:
        dict: A dictionary containing the 'output' key with the generated response as its value.
    """
    # task = progress.add_task(model, total=100)

    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if DEBUG:
        logger.info(
            f"model: {model}, instruction: {item['instruction']}, output: {output[:20]}"
        )

    return {"output": output}


def main(
    model: str = "Qwen/Qwen2-72B-Instruct",
    reference_models: list[str] = default_reference_models,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    multi_turn=True,
):
    md = Markdown(welcome_message)
    console.print(md)

    data = {
        "instruction": [[] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": [m for m in reference_models],
    }

    num_proc = len(reference_models)

    while True:

        try:
            instruction = Prompt.ask("\nMessage [cyan bold]Together MoA[/cyan bold] ")
        except EOFError:
            break

        if instruction == "exit" or instruction == "quit":
            print("Goodbye!")
            break
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append({"role": "user", "content": instruction})
                data["references"] = [""] * len(reference_models)
        else:
            data = {
                "instruction": [[{"role": "user", "content": instruction}]]
                * len(reference_models),
                "references": [""] * len(reference_models),
                "model": [m for m in reference_models],
            }

        eval_set = datasets.Dataset.from_dict(data)

        for i_round in range(rounds):
            eval_set = eval_set.map(
                partial(
                    process_fn,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                batched=False,
                num_proc=num_proc,
            )
            references = [item["output"] for item in eval_set]
            data["references"] = references
            eval_set = datasets.Dataset.from_dict(data)

        output = generate_with_references(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=data["instruction"][0],
            references=references,
            generate_fn=generate_together_stream,
        )

        all_output = ""
        for chunk in output:
            # print(chunk)
            out = chunk.choices[0].delta.content
            print(out, end="")
            all_output += out
        print()

        if DEBUG:
            logger.info(
                f"model: {model}, instruction: {data['instruction'][0]}, output: {all_output[:20]}"
            )
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append(
                    {"role": "assistant", "content": all_output}
                )


if __name__ == "__main__":
    typer.run(main)
