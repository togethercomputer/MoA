"""Offline check that bot.py sends the right prompts on multi round runs.

This drives the real ``bot.main()``. It does not copy or re-implement the round
loop, so if the loop regresses the check fails, whatever the code looks like.
Issues #30 and #26.

What is replaced, and why:

* ``utils.requests.post``  captures the exact payload each proposer would
  receive and returns a canned answer, so no request leaves the machine.
* ``bot.generate_together_stream``  records the aggregator payload and returns
  a short fake stream. The aggregator transport is not what is under test.
* ``bot.Prompt.ask``  feeds scripted answers, ending with "exit" so the
  interactive loop finishes.
* ``datasets.Dataset.map``  runs in process. bot.py passes
  ``num_proc=len(reference_models)``, and spawned workers would not inherit the
  patched ``requests.post``, so leaving it alone could send real traffic.
* Outbound sockets raise, as a hard stop in case any of the above is bypassed.

Run:  .venv/bin/python verify_main.py
Exit code 0 means every scenario passed.
"""

import io
import socket
import sys
from contextlib import redirect_stdout
from unittest import mock

import datasets

MODELS = ["model-A", "model-B", "model-C", "model-D"]
INSTRUCTION = "Top things to do in NYC"
AGGREGATE_PROMPT_MARKER = "Responses from models:"


class NetworkUsed(Exception):
    pass


def _block_sockets():
    def deny(*args, **kwargs):
        raise NetworkUsed("a real network call was attempted")

    socket.socket.connect = deny
    socket.create_connection = deny
    socket.getaddrinfo = deny


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeChunk:
    def __init__(self, text):
        self.choices = [type("D", (), {"delta": type("C", (), {"content": text})()})()]


class Run:
    """One scripted run of bot.main(), with everything it sends recorded."""

    def __init__(self, failing_model=None):
        self.proposer_calls = []  # (model, messages)
        self.aggregator_calls = []  # messages
        self.failing_model = failing_model

    def round_of(self, model):
        return sum(1 for m, _ in self.proposer_calls if m == model) + 1

    def fake_post(self, url, json=None, headers=None, **kwargs):
        model = json["model"]
        self.proposer_calls.append((model, json["messages"]))
        if model == self.failing_model:
            # What the API sends when the request cannot be served, which makes
            # generate_together return None.
            return FakeResponse({"error": {"type": "invalid_request_error"}})
        text = f"<{model} answer #{self.round_of(model) - 1}: visit The Met.>"
        return FakeResponse({"choices": [{"message": {"content": text}}]})

    def fake_stream(self, model, messages, **kwargs):
        self.aggregator_calls.append(messages)
        return [FakeChunk("aggregated. "), FakeChunk("done.")]


def run_bot(rounds, turns=1, multi_turn=True, failing_model=None):
    """Import bot fresh, patch its edges, and call the real main()."""
    for name in ("bot", "utils"):
        sys.modules.pop(name, None)
    import utils

    import bot

    run = Run(failing_model=failing_model)
    answers = ["Qwen/Qwen2-72B-Instruct", "0.7", "512"]
    answers += [INSTRUCTION] * turns + ["exit"]
    script = iter(answers)

    real_map = datasets.Dataset.map

    def map_in_process(self, *args, **kwargs):
        kwargs["num_proc"] = None
        return real_map(self, *args, **kwargs)

    with mock.patch.object(utils.requests, "post", run.fake_post), mock.patch.object(
        bot, "generate_together_stream", run.fake_stream
    ), mock.patch.object(
        bot.Prompt, "ask", lambda *a, **k: next(script)
    ), mock.patch.object(
        datasets.Dataset, "map", map_in_process
    ):
        with redirect_stdout(io.StringIO()):
            bot.main(
                reference_models=MODELS,
                rounds=rounds,
                multi_turn=multi_turn,
            )
    return run


def references_in(messages):
    """The enumerated reference list a proposer actually received."""
    system = [m for m in messages if m["role"] == "system"]
    if not system:
        return None
    body = system[0]["content"]
    if AGGREGATE_PROMPT_MARKER not in body:
        return None
    listed = body.split(AGGREGATE_PROMPT_MARKER, 1)[1].strip().splitlines()
    return [line.split(". ", 1)[1] for line in listed if ". " in line]


def check(name, rounds, turns=1, multi_turn=True, failing_model=None):
    run = run_bot(
        rounds, turns=turns, multi_turn=multi_turn, failing_model=failing_model
    )
    calls_per_round = len(MODELS)
    expected_references = len(MODELS) - (1 if failing_model else 0)
    problems = []

    if len(run.proposer_calls) != calls_per_round * rounds * turns:
        problems.append(
            f"expected {calls_per_round * rounds * turns} proposer calls, "
            f"got {len(run.proposer_calls)}"
        )

    # Round 1 of each turn must carry no references at all.
    first = run.proposer_calls[0][1]
    if references_in(first) is not None:
        problems.append("round 1 proposers were given references")

    # Every later round must carry all previous answers, in full.
    if rounds >= 2:
        for model, messages in run.proposer_calls[calls_per_round : calls_per_round * rounds]:
            got = references_in(messages)
            if got is None:
                problems.append(f"{model}: no references in a later round")
                continue
            if len(got) != expected_references:
                problems.append(
                    f"{model}: saw {len(got)} references, expected "
                    f"{expected_references}"
                )
            if any(len(reference) <= 1 for reference in got):
                problems.append(
                    f"{model}: references split into characters, first few {got[:6]}"
                )
            if "None" in got:
                problems.append(
                    f"{model}: a failed model was passed on as the text 'None'"
                )
            user_turns = [m for m in messages if m["role"] == "user"]
            if not user_turns or user_turns[-1]["content"] != INSTRUCTION:
                problems.append(f"{model}: original prompt missing from the user turn")

    if problems:
        print(f"FAIL [{name}]")
        for problem in problems:
            print(f"       {problem}")
        return False
    print(f"PASS [{name}]")
    return True


def main():
    _block_sockets()
    results = [
        check("rounds=1 (default path stays untouched)", 1),
        check("rounds=2 (issue #30)", 2),
        check("rounds=3", 3),
        check("rounds=2 across two turns", 2, turns=2),
        check("rounds=2 with multi_turn off", 2, multi_turn=False),
        check("rounds=2 with one model failing", 2, failing_model=MODELS[1]),
    ]
    if all(results):
        print("\nAll scenarios passed.")
        return 0
    print(f"\n{results.count(False)} scenario(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
