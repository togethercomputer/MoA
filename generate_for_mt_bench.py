"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

# import openai
import requests
from loguru import logger


"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures

import openai
import shortuuid
import tqdm

from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_palm,
)
from fastchat.llm_judge.gen_model_answer import reorg_answer_file
from fastchat.model.model_adapter import get_conversation_template

from typing import List
from loguru import logger
import openai

from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    DEBUG,
)


def get_answer(
    question: dict,
    model: str,
    reference_models: List[str],
    num_choices: int,
    max_tokens: int,
    answer_file: str,
    rounds: int,
    provider: str,
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []

    if provider == "together":
        generate_fn = generate_together
    elif provider == "openai":
        generate_fn = generate_openai
    else:
        assert False

    for i in range(num_choices):

        turns = []
        messages = []

        for j in range(len(question["turns"])):

            qs = question["turns"][j]

            messages.append({"role": "user", "content": qs})

            references = []

            if len(reference_models) > 0:

                prev_references = []

                for i_round in range(rounds):

                    if DEBUG:
                        logger.info(
                            f"Round {i_round+1}/{rounds} to collecting reference responses."
                        )

                    references = []

                    for reference_model in reference_models:

                        reference = generate_with_references(
                            model=reference_model,
                            messages=messages,
                            references=prev_references,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            generate_fn=generate_fn,
                        )

                        if reference is not None:

                            references.append(reference)

                    if i_round < rounds - 1:

                        prev_references = references

                        references = []

            output = generate_with_references(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                generate_fn=generate_fn,
                references=references,
            ).strip()

            messages.append(
                {
                    "role": "assistant",
                    "content": output,
                }
            )

            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--reference-models", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--provider", type=str, default="together")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    args = parser.parse_args()

    question_file = f"FastChat/fastchat/llm_judge/data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"outputs/{args.bench_name}/model_answer/{args.model}.jsonl"
    print(f"Output to {answer_file}")

    if args.reference_models is None:
        reference_models = []
    else:
        reference_models = args.reference_models.split(",")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                reference_models,
                args.num_choices,
                args.max_tokens,
                answer_file,
                args.rounds,
                args.provider,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
