import typer
from flask import Flask, request, jsonify, Response, stream_with_context
import json
from functools import partial
import datasets
from utils import generate_together_stream, generate_with_references, DEBUG
from loguru import logger
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

app = Flask(__name__)

default_model = None
default_reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
_temperature = 0.7
_max_tokens = 512
_rounds = 1

def process_fn(item, temperature=0.7, max_tokens=2048):
    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=_temperature,
        max_tokens=_max_tokens,
    )
    if DEBUG:
        logger.info(
            f"model: {model}, instruction: {item['instruction']}, output: {output[:20]}"
        )

    return {"output": output}

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    stream = data.get('stream', False)  # Check if the client requested streaming
    temperature = data.get('temperature', _temperature)
    max_tokens = data.get('max_tokens', _max_tokens)
    
    # Prepare data for processing
    data = {
        "instruction": [messages] * len(default_reference_models),
        "references": [""] * len(default_reference_models),
        "model": [m for m in default_reference_models],
    }

    eval_set = datasets.Dataset.from_dict(data)

    # Process with reference models
    eval_set = eval_set.map(
        partial(
            process_fn,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        batched=False,
        num_proc=len(default_reference_models),
    )
    references = [item["output"] for item in eval_set]

    # Generate final output
    output = generate_with_references(
        model=default_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        references=references,
        generate_fn=generate_together_stream,
    )

    # Collect output
    all_output = ""
    for chunk in output:
        out = chunk.choices[0].delta.content
        if out is not None:
            # print(out)
            all_output += out

    # Prepare response
    print (all_output)
    response = {
        "id": "chatcmpl-123",  # TODO
        "object": "chat.completion",
        "created": 1720384636,  # TODO
        "model": default_model,
        "usage": {
            "prompt_tokens": 42,  # TODO
            "completion_tokens": len(all_output.split()),  # Rough estimate
            "total_tokens": 42 + len(all_output.split()),  # Rough estimate
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": all_output,
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    if DEBUG:
      print(json.dumps(response, indent=2))

    def generate():
        if stream:
            # Simulate streaming by yielding chunks
            #chunks = [all_output[i:i+5] for i in range(0, len(all_output), 5)]  # Split into 5-character chunks
            chunks = [all_output]
            for chunk in chunks:
                chunk_response = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1720384636,
                    "model": default_model,
                    "choices": [
                        {
                            "delta": {
                                "content": chunk,
                            },
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_response)}\n\n"
            
            # Send the final chunk with finish_reason
            final_chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1720384636,
                "model": default_model,
                "choices": [
                    {
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            # Non-streaming response
            yield json.dumps(response)

    if stream:
        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        return jsonify(response)

def main(
    model: str = "Qwen/Qwen2-72B-Instruct",
    reference_models: list[str] = default_reference_models,
    temperature: float = 0.7,
    max_tokens: int = 512,
    rounds: int = 1,
    port: int = 5001,
):
    global default_model, default_reference_models, _temperature, _max_tokens, _rounds
    default_model = model
    default_reference_models = reference_models
    _temperature = temperature
    _max_tokens = max_tokens
    _rounds = rounds
    app.run(port=port)

if __name__ == "__main__":
    typer.run(main)
