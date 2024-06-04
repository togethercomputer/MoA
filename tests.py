
from utils import (
    generate_together,
    generate_openai,
    inject_references_to_messages,
    generate_with_references,
)


if __name__ == "__main__":

    #####
    messages = [{'role': 'user', 'content': 'hello!'}]
    output = generate_together(
        'meta-llama/Llama-3-8b-chat-hf', 
        messages,
        temperature=0,
    )
    assert output.strip() == "Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?"
    print('#1 pass')

    #####
    messages = [{'role': 'user', 'content': 'hello!'}]
    output = generate_openai(
        'gpt-3.5-turbo',
        messages,
        temperature=0,
    )
    assert output.strip() == "Hello! How can I assist you today?"
    print('#2 pass')

    #####
    messages = [{'role': 'user', 'content': 'hello!'}]
    messages = inject_references_to_messages(messages, ["Hello! How can I help you today?", "Hello! How can I assist you today?"])
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    output = generate_together(
        'meta-llama/Llama-3-8b-chat-hf', 
        messages,
        temperature=0,
    )
    assert output.strip() == "Hello! It seems like you're looking for assistance with something. I'm here to help! Could you please provide more context or clarify what's on your mind? I'll do my best to offer a helpful and accurate response."
    print('#3 pass')

    ####
    messages = [{'role': 'user', 'content': 'hello!'}]
    output = generate_with_references(
        'meta-llama/Llama-3-8b-chat-hf', 
        messages,
        references=["Hello! How can I help you today?", "Hello! How can I assist you today?"],
        temperature=0,
    )
    assert output.strip() == "Hello! It seems like you're looking for assistance with something. I'm here to help! Could you please provide more context or clarify what's on your mind? I'll do my best to offer a helpful and accurate response."
    print('#4 pass')