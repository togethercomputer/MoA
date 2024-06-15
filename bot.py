import datasets
from functools import partial
from loguru import logger
from utils import (
    generate_openai,
    generate_with_references,
    DEBUG,
)
import typer
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from datasets.utils.logging import disable_progress_bar
from time import sleep

disable_progress_bar()

console = Console()

welcome_message = """
# Welcome to the Together AI MoA (Mixture-of-Agents) interactive demo!

Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results. By employing a layered architecture where each layer comprises several LLM agents, MoA significantly outperforms GPT-4 Omni’s 57.5% on AlpacaEval 2.0 with a score of 65.1%, using only open-source models!

This demo uses the following LLMs as reference models, then passes the results to the aggregate model for the final response:
- llmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf
- microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf

"""

default_reference_models = [
    "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
    "microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
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

    print(f"\nFinished querying [bold]{model}.[/bold]")

    return {"output": output}


def main(
    model: str = "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
    reference_models: list[str] = default_reference_models,
    temperature: float = 0.7,
    max_tokens: int = 512,
    rounds: int = 1,
    multi_turn=True,
):
    """
    Runs a continuous conversation between user and MoA.

    Args:
    - model (str): The primary model identifier used for generating the final response. This model aggregates the outputs from the reference models to produce the final response.
    - reference_models (List[str]): A list of model identifiers that are used as references in the initial rounds of generation. These models provide diverse perspectives and are aggregated by the primary model.
    - temperature (float): A parameter controlling the randomness of the response generation. Higher values result in more varied outputs. The default value is 0.7.
    - max_tokens (int): The maximum number of tokens that can be generated in the response. This limits the length of the output from each model per turn. Default is 2048.
    - rounds (int): The number of processing rounds to refine the responses. In each round, the input is processed through the reference models, and their outputs are aggregated. Default is 1.
    - multi_turn (bool): Enables multi-turn interaction, allowing the conversation to build context over multiple exchanges. When True, the system maintains context and builds upon previous interactions. Default is True. When False, the system generates responses independently for each input.
    """
    md = Markdown(welcome_message)
    console.print(md)
    sleep(0.75)
    console.print(
        "\n[bold]To use this demo, answer the questions below to get started [cyan](press enter to use the defaults)[/cyan][/bold]:"
    )

    data = {
        "instruction": [[] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": [m for m in reference_models],
    }

    num_proc = len(reference_models)

    model = Prompt.ask(
        "\n1. What main model do you want to use?",
        default="lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
    )
    console.print(f"Selected {model}.", style="yellow italic")
    temperature = int(
        Prompt.ask(
            "2. What temperature do you want to use? [cyan bold](0.7) [/cyan bold]",
            default=0.7,
            show_default=True,
        )
    )
    console.print(f"Selected {temperature}.", style="yellow italic")
    max_tokens = int(
        Prompt.ask(
            "3. What max tokens do you want to use? [cyan bold](512) [/cyan bold]",
            default=512,
            show_default=True,
        )
    )
    console.print(f"Selected {max_tokens}.", style="yellow italic")

    while True:

        try:
            instruction = Prompt.ask(
                "\n[cyan bold]Prompt >>[/cyan bold] ",
                default="Top things to do in NYC",
                show_default=True,
            )
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

        with console.status("[bold green]Querying all the models...") as status:
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

        console.print(
            "[cyan bold]Aggregating results & querying the aggregate model...[/cyan bold]"
        )
        output = generate_with_references(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=data["instruction"][0],
            references=references,
            generate_fn=generate_openai,
        )

        print("\n")
        console.log(Markdown(f"## Final answer from {model}"))
        print(output)

        print()

        if DEBUG:
            logger.info(
                f"model: {model}, instruction: {data['instruction'][0]}, output: {output[:20]}"
            )
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append(
                    {"role": "assistant", "content": output}
                )


if __name__ == "__main__":
    typer.run(main)
