"""Runs all unit tests for the decoders."""
import math
from types import SimpleNamespace

import pytest

from alpaca_eval.decoders.anthropic import anthropic_completions
from alpaca_eval.decoders.cohere import cohere_completions
from alpaca_eval.decoders.huggingface_api import huggingface_api_completions
from alpaca_eval.decoders.openai import openai_completions

MOCKED_COMPLETION = "Mocked completion text"


@pytest.fixture
def mock_openai_completion():
    # Create a mock Completion object
    completion_mock = dict()
    completion_mock["total_tokens"] = 3
    completion_mock["text"] = MOCKED_COMPLETION
    return completion_mock


def test_openai_completions(mocker, mock_openai_completion):
    # Patch the _openai_completion_helper function to return the mock completion object

    mocker.patch(
        "alpaca_eval.decoders.openai._openai_completion_helper",
        return_value=[mock_openai_completion],
    )
    # use num_procs=1 to avoid issues with pickling
    result = openai_completions(["Prompt 1", "Prompt 2"], "text-davinci-003", batch_size=1, num_procs=1)

    _run_all_asserts_completions(result)


def test_anthropic_completions(mocker):
    mock_response = dict(text=MOCKED_COMPLETION)

    mocker.patch(
        "alpaca_eval.decoders.anthropic._anthropic_completion_helper",
        return_value=mock_response,
    )
    result = anthropic_completions(
        ["<|im_start|>user\nPrompt 1\n<|im_end|>", "<|im_start|>user\nPrompt 2\n<|im_end|>"], num_procs=1
    )
    _run_all_asserts_completions(result)


def test_cohere_completions(mocker):
    mocker.patch(
        "alpaca_eval.decoders.cohere._cohere_completion_helper",
        return_value=["Mocked completion text", 42],
    )
    result = cohere_completions(["Prompt 1", "Prompt 2"], num_procs=1)
    _run_all_asserts_completions(result)


def test_huggingface_api_completions(mocker):
    mocker.patch(
        "alpaca_eval.decoders.huggingface_api.inference_helper",
        return_value="Mocked completion text",
    )
    result = huggingface_api_completions(
        ["Prompt 1", "Prompt 2"],
        model_name="timdettmers/guanaco-33b-merged",
        num_procs=1,
    )
    _run_all_asserts_completions(result)


def _run_all_asserts_completions(result):
    expected_completions = [MOCKED_COMPLETION, MOCKED_COMPLETION]
    assert result["completions"] == expected_completions

    for i in range(len(result["time_per_example"])):
        assert 0 < result["time_per_example"][i] < 1

    assert len(result["price_per_example"]) == 2
    if not math.isnan(result["price_per_example"][0]):
        assert result["price_per_example"][0] == result["price_per_example"][1]
        assert 0 <= result["price_per_example"][0] < 1e-2
    else:
        assert math.isnan(result["price_per_example"][1]) == math.isnan(result["price_per_example"][0])
