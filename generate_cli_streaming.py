
import json
import datasets
from fire import Fire
from functools import partial
from typing import List
from loguru import logger
import os
from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    DEBUG,
)
DEBUG = 0

def process_fn(
    item, 
    temperature=0.7,
    max_tokens=2048,
):
    references = item.get('references', [])
    model = item['model']
    messages =  item['instruction']
    
    output = generate_with_references(
        model=model,
        messages=messages,
        references=references)
    if DEBUG:
        logger.info(f"model: {model}, instruction: {item['instruction']}, output: {output[:20]}")
    

    return {'output': output}


def main(
    model: str,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    num_proc: int = 16,
    multi_turn=False,
):

    # continously take in inputs, and generate outputs
    print("Please input instructions to generate responses from MoA")
    print(f"Reference models: {reference_models}\nAggregate Model: {model}")
    data = {"instruction": [[] for _ in range(len(reference_models))], "references": [""]*len(reference_models),"model": [m for m in reference_models]}
    while True:
        try:
            instruction = input("Input: ")
        except EOFError:
            break
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append({'role': 'user', 'content': instruction})
        else:
            data = {"instruction": [[{'role': 'user', 'content': instruction}]]*len(reference_models), "references": [""]*len(reference_models),"model": [m for m in reference_models]}
        eval_set = datasets.Dataset.from_dict(data)
        for i_round in range(rounds):
            eval_set = eval_set.map(
                partial(
                    process_fn, 
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                batched=False, num_proc=num_proc,
            )
            references = [item['output'] for item in eval_set]
            data["references"] = references
            eval_set = datasets.Dataset.from_dict(data)

        output = generate_with_references(
            model=model,
            messages=data["instruction"][0],
            references=references,
            streaming=True
        )
        all_output = ""
        print("Agent: ", end="")
        for chunk in output:
            # print(chunk)
            out = chunk.choices[0].delta.content
            print(out, end="")
            all_output += out
        print()
        if DEBUG:
            logger.info(f"model: {model}, instruction: {data['instruction'][0]}, output: {all_output[:20]}")
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append({'role': 'assistant', 'content': all_output})

if __name__ == '__main__':
    main_model = "Qwen/Qwen1.5-110B-Chat"
    reference_models=["microsoft/WizardLM-2-8x22B","Qwen/Qwen1.5-110B-Chat","Qwen/Qwen1.5-72B-Chat","meta-llama/Llama-3-70b-chat-hf","mistralai/Mixtral-8x22B-Instruct-v0.1","databricks/dbrx-instruct"]
    # reference_models=["meta-llama/Llama-3-70b-chat-hf","mistralai/Mixtral-8x22B-Instruct-v0.1"]
    temperature = 0.7
    max_tokens = 2048
    rounds = 1
    multi_turn = True
    main(model=main_model, reference_models=reference_models, temperature=temperature, max_tokens=max_tokens, rounds=rounds, num_proc=len(reference_models),multi_turn=multi_turn)