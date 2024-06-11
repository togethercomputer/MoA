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

    return {"output": output}


def main(
    model: str,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    num_proc: int = 6,
    multi_turn=True,
):
    """
    Runs a continuous conversation between user and MoA.

    Args:
        model (str): The primary model identifier used for generating the final response. This model aggregates
                     the outputs from the reference models to produce the final response.
        reference_models (List[str]): A list of model identifiers that are used as references in the initial
                                      rounds of generation. These models provide diverse perspectives and are
                                      aggregated by the primary model.
        temperature (float): A parameter controlling the randomness of the response generation. Higher values
                             result in more varied outputs. The default value is 0.7.
        max_tokens (int): The maximum number of tokens that can be generated in the response. This limits the
                          length of the output from each model per turn. Default is 2048.
        rounds (int): The number of processing rounds to refine the responses. In each round, the input is processed
                      through the reference models, and their outputs are aggregated. Default is 1.
        num_proc (int): The number of processes to run in parallel, improving the efficiency of the response
                        generation process. Typically set to the number of reference models. Default is 6.
        multi_turn (bool): Enables multi-turn interaction, allowing the conversation to build context over multiple
                           exchanges. When True, the system maintains context and builds upon previous interactions.
                           Default is True. When False, the system generates responses independently for each input.
    """
    print(
        "Welcome to MoA interactive demo! Please input instructions to generate responses..."
    )
    print(f"Reference models: {','.join(reference_models)}\nAggregate Model: {model}")

    data = {
        "instruction": [[] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": [m for m in reference_models],
    }

    while True:

        try:
            instruction = input("\n>>> ")
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
        # for i_round in range(rounds):
        #     eval_set = eval_set.map(
        #         partial(
        #             process_fn,
        #             temperature=temperature,
        #             max_tokens=max_tokens,
        #         ),
        #         batched=False,
        #         num_proc=num_proc,
        #     )
        #     references = [item["output"] for item in eval_set]
        #     data["references"] = references
        #     eval_set = datasets.Dataset.from_dict(data)

        total = 0
        for value in track(range(100), description="Processing..."):
            # Fake processing time
            time.sleep(0.01)
            total += 1
        print(f"Processed {total} things.")

        # output = generate_with_references(
        #     model=model,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     messages=data["instruction"][0],
        #     references=references,
        #     generate_fn=generate_together_stream,
        # )

        # all_output = ""
        # for chunk in output:
        #     # print(chunk)
        #     out = chunk.choices[0].delta.content
        #     print(out, end="")
        #     all_output += out
        # print()

        # if DEBUG:
        #     logger.info(
        #         f"model: {model}, instruction: {data['instruction'][0]}, output: {all_output[:20]}"
        #     )
        # if multi_turn:
        #     for i in range(len(reference_models)):
        #         data["instruction"][i].append(
        #             {"role": "assistant", "content": all_output}
        #         )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--aggregator",
        default="Qwen/Qwen2-72B-Instruct",
        type=str,
        help="the name of the aggregator model to use",
    )
    parser.add_argument(
        "--reference_models",
        type=str,
        default=",".join(
            [
                "Qwen/Qwen2-72B-Instruct",
                "Qwen/Qwen1.5-72B-Chat",
                "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "databricks/dbrx-instruct",
            ]
        ),
        help="reference models to use, separated by commas",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="the maximum number of tokens to generate",
    )
    parser.add_argument(
        "--round", default=1, type=int, help="the number of rounds to aggregate"
    )
    parser.add_argument(
        "--no-multi-turn",
        default=True,
        action="store_false",
        help="indicates whether to remeber context from previous turns or not",
    )
    parser.add_argument(
        "--temperature", default=0.7, type=float, help="temperature for the LM"
    )
    args = parser.parse_args()

    reference_models = args.reference_models.split(",")
    temperature = args.temperature
    max_tokens = args.max_tokens
    rounds = args.round
    multi_turn = args.no_multi_turn

    main(
        model=args.aggregator,
        reference_models=reference_models,
        temperature=temperature,
        max_tokens=max_tokens,
        rounds=rounds,
        num_proc=len(reference_models),
        multi_turn=multi_turn,
    )
