from concurrent.futures import ProcessPoolExecutor

import argparse

import openai

from time import sleep
import random
import json

import fcntl

from typing import List
import os
import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)



def main():
    API_KEYS = os.environ["OPENAI_API_KEYS"].split(",")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--fail-path", type=str, required=True)
    parser.add_argument("--requests-per-minute", type=int, default=60, help="Number of requests per minute per API key")
    parser.add_argument("--expected_response_seconds", type=float, default=5, help="Number of seconds to wait for a response")
    args = parser.parse_args()

    openai_concurrent = OpenAIChatCompletionConcurrent(api_keys=API_KEYS, requests_per_minute=args.requests_per_minute, expected_response_seconds=args.expected_response_seconds)
    openai_concurrent.create_many_file(input_path=args.input_path, output_path=args.output_path, fail_path=args.fail_path)


class OpenAIChatCompletionConcurrent:
    def __init__(self, api_keys: List[str], requests_per_minute: int = 60, expected_response_seconds: float = 5.0):
        self.api_keys = api_keys
        self.requests_per_minute = requests_per_minute
        self.expected_response_seconds = expected_response_seconds

        if len(api_keys) == 0:
            api_key = os.environ.get('OPENAI_API_KEY', None)
            assert api_key is not None
            self.api_keys = [api_key]

        self.num_api_keys = len(self.api_keys)

        requests_per_second = self.requests_per_minute / 60
        simultaneous_num_requests = requests_per_second * self.expected_response_seconds
        buffer = 2
        self.num_workers = int(simultaneous_num_requests * buffer)

        total_requests_per_second = requests_per_second * self.num_api_keys
        self.time_between_requests = 1 / total_requests_per_second


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def create(self, model: str, messages: List[dict], temperature: float, max_tokens: int):
        openai.api_key = random.choice(self.api_keys)
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        

    def create_many(self, requests: List[dict]):

        futures = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for item_index, item in enumerate(tqdm.tqdm(requests)):
                api_key = self.api_keys[item_index % self.num_api_keys]    
                future = executor.submit(call_and_return, api_key=api_key, item=item)
                futures.append(future)
                sleep(self.time_between_requests)

        responses = []
        fails = []
        for future in futures:
            response, success = future.result()
            if success:
                responses.append(response)
            else:
                fails.append(response)

        return responses, fails

        
    def create_many_file(self, input_path: str, output_path: str, fail_path: str):

        with open(input_path, "r") as input_file:
            requests = [json.loads(line) for line in input_file.readlines()]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for item_index, item in enumerate(requests):
                api_key = self.api_keys[item_index % self.num_api_keys]    
                executor.submit(call_and_write, api_key=api_key, item=item, output_path=output_path, fail_path=fail_path)
                sleep(self.time_between_requests)


def call_and_return(api_key: str, item: dict):
    try:
        response = completion_with_backoff(api_key, **item["request"])
        error = None
    except Exception as e:
        response = None
        error = repr(e)

    if error is None:
        output_item = {**item, "api_key": api_key, "response": response}
    else:
        output_item = {**item, "api_key": api_key, "error": error}

    
    return output_item, error is None


def call_and_write(api_key: str, item: dict, output_path: str, fail_path: str):
    try:
        response = completion_with_backoff(api_key, **item["request"])
        error = None
    except Exception as e:
        response = None
        error = repr(e)

    if error is None:
        output_item = {**item, "api_key": api_key, "response": response}
        output_line = json.dumps(output_item)
        with open(output_path, "a") as output_file:
            fcntl.flock(output_file, fcntl.LOCK_EX)
            output_file.write(output_line + "\n")
            fcntl.flock(output_file, fcntl.LOCK_UN)
    else:
        fail_item = {**item, "api_key": api_key, "error": error}
        fail_line = json.dumps(fail_item)
        with open(fail_path, "a") as fail_file:
            fcntl.flock(fail_file, fcntl.LOCK_EX)
            fail_file.write(fail_line + "\n")
            fcntl.flock(fail_file, fcntl.LOCK_UN)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
def completion_with_backoff(api_key, **kwargs):
    client = openai.OpenAI(
        api_key=api_key,
    )
    return client.chat.completions.create(**kwargs)


if __name__ == "__main__":
    main()

    # Test
    # openai_concurrent = OpenAIChatCompletionConcurrent(api_keys=API_KEYS, requests_per_minute=60, expected_response_seconds=5)

    # example_inputs = [
    #         {"request": {"model": "gpt-5", "messages": [{"role": "system", "content": "Hello, how are you?"}, {"role": "user", "content": "Hello"}], "temperature": 0.2, "max_tokens": 10}},
    #         {"request": {"model": "gpt-4", "messages": [{"role": "system", "content": "Hello, how are you?"}, {"role": "user", "content": "Hello"}], "temperature": 0.2, "max_tokens": 10}},
    # ]

    # print(openai_concurrent.create_many(example_inputs))
