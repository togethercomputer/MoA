import asyncio
import logging
from langchain.schema import HumanMessage
from langchain_core.runnables import Runnable
from mesop_langgraph.graph import GraphState, get_graph
import mesop as me
import mesop.labs as mel

logging.basicConfig(level=logging.DEBUG)
graph: Runnable = get_graph()


@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/chat",
    title="Mixture Of Agents Chat",
)
def page():
    mel.chat(transform, title="Demo Chat", bot_user="MoA Bot")


def transform(input: str, history: list[mel.ChatMessage]):
    messages = [
        HumanMessage(content=input),
    ]
    state: GraphState = {"depth": 0, "messages": messages}
    response: GraphState = asyncio.run(graph.ainvoke(state, debug=True))
    for layer, intermediate_output in enumerate(response["intermediate_outputs"]):
        yield "=" * 5 + f"Layer[{layer}]" + "=" * 5 + "\n\n"
        for model, output in intermediate_output.items():
            yield f"{model}: {output}\n\n"
    yield "=" * 5 + "Final Response" + "=" * 5 + "\n\n"
    yield response["messages"][-1].content
