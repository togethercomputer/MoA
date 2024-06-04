import os
import subprocess

import pytest


@pytest.mark.slow
def test_openai_fn_evaluate_example():
    env = os.environ.copy()
    env["IS_ALPACA_EVAL_2"] = "True"
    result = subprocess.run(
        [
            "alpaca_eval",
            "--model_outputs",
            "example/outputs.json",
            "--max_instances",
            "2",
            "--annotators_config",
            "alpaca_eval_gpt4_fn",
            "--is_avoid_reannotations",
            "False",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    normalized_output = " ".join(result.stdout.split())
    expected_output = " ".join("0.00 0.00 2".split())

    assert expected_output in normalized_output
    assert "example" in normalized_output
    assert "length_controlled_winrate" in normalized_output
