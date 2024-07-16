"""
Implementation of MoA with langgraph

Total layer: 3(2 for propoesers, 1 for aggregator)
The number of proposers in a layer: 3

The mermaid UML is here
```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([__start__]):::first
        __end__([__end__]):::last
        entry(entry)
        proposer1(proposer1)
        proposer2(proposer2)
        proposer3(proposer3)
        asp_node(asp_node)
        aggregator(aggregator)
        __start__ --> entry;
        aggregator --> __end__;
        entry --> proposer1;
        entry --> proposer2;
        entry --> proposer3;
        proposer1 --> asp_node;
        proposer2 --> asp_node;
        proposer3 --> asp_node;
        asp_node -.-> entry;
        asp_node -.-> proposer1;
        asp_node -.-> proposer2;
        asp_node -.-> proposer3;
        asp_node -.-> aggregator;
        asp_node -.-> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
"""

from typing import Annotated, Dict, List, TypedDict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages
from langchain_community.chat_models.ollama import ChatOllama

LAYER_WIDTH, GRAPH_DEPTH = 3, 2


def add_outputs(
    origin: List[Dict[str, AIMessage]], added: Dict[str, AIMessage]
) -> List[Dict[str, AIMessage]]:
    """
    Save outputs of each layer.
    """
    result = origin + [added]
    return result


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    intermediate_outputs: Annotated[List[Dict[str, AIMessage]], add_outputs]
    depth: int


async def aggregate_and_synthesize(state: GraphState):
    """
    Make an intermediate output
    Proposers and Aggregator use this prompt via state["messages"][-2:]
    """
    responses = state["messages"][-LAYER_WIDTH:]
    intermediate_outputs = {}
    human_query = state["messages"][0].content
    aggregate_and_synthesize_prompt = """
You have been provided with a set of responses from various open-source models to the latest user query. Your
task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the
information provided in these responses, recognizing that some of it may be biased or incorrect. Your response
should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply
to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of
accuracy and reliability.
Responses from models:
"""
    for idx, response in enumerate(responses):
        aggregate_and_synthesize_prompt += f"{idx+1}: {response.content}\n"
        intermediate_outputs[response.response_metadata["model"]] = response.content
    # Input Prompts for each agents
    messages = [
        SystemMessage(content=aggregate_and_synthesize_prompt),
        HumanMessage(content=human_query),
    ]

    return {"messages": messages, "intermediate_outputs": intermediate_outputs}


async def depth_checker(state: GraphState):
    """
    Determine whether the process ends.
    """
    if state["depth"] == GRAPH_DEPTH:
        return "aggregator"
    else:
        return "entry"


async def entry(state: GraphState):
    """
    Add graph depth
    """
    return {"depth": state["depth"] + 1}


async def proposer1(state: GraphState):
    """
    [Proposer1]
    - model: gemma:2b-instruct
    - temperature: 0.2
    """
    system_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOllama(model="gemma:2b-instruct", temperature=0.2, max_tokens=1024)
    model = system_template | llm

    input_ = state["messages"][-2:]
    response = model.invoke({"messages": input_})
    return {"messages": response}


async def proposer2(state: GraphState):
    """
    [Proposer2]
    - model: qwen2:1.5b-instruct-fp16
    - temperature: 0.2
    """
    system_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOllama(model="qwen2:1.5b-instruct-fp16", temperature=0.2, max_tokens=1024)
    model = system_template | llm
    input_ = state["messages"][-2:]
    response = model.invoke({"messages": input_})
    return {"messages": response}


async def proposer3(state: GraphState):
    """
    [Proposer3]
    - model: phi3:3.8b-instruct
    - temperature: 0.2
    """
    system_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOllama(model="phi3:3.8b-instruct", temperature=0.2, max_tokens=1024)
    model = system_template | llm
    input_ = state["messages"][-2:]
    response = model.invoke({"messages": input_})
    return {"messages": response}


async def aggregator(state: GraphState):
    """
    [Aggregator]
    - model: qwen2:7b-instruct
    - temperature: 0.2
    """
    system_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOllama(model="qwen2:7b-instruct", temperature=0.2, max_tokens=1024)
    model = system_template | llm
    input_ = state["messages"][-2:]
    response = model.invoke({"messages": input_})
    return {"messages": response}


def get_graph():
    """
    Make DAG.
    """
    graph_builder = StateGraph(GraphState)
    proposers = {
        "proposer1": proposer1,
        "proposer2": proposer2,
        "proposer3": proposer3,
    }
    # Add nodes
    graph_builder.add_node("entry", entry)
    for name, node in proposers.items():
        graph_builder.add_node(name, node)
    graph_builder.add_node("asp_node", aggregate_and_synthesize)
    graph_builder.add_node("aggregator", aggregator)
    # Add edges
    graph_builder.add_edge(START, "entry")
    for proposer in proposers.keys():
        graph_builder.add_edge("entry", proposer)
        graph_builder.add_edge(proposer, "asp_node")
    graph_builder.add_conditional_edges("asp_node", depth_checker)
    graph_builder.add_edge("aggregator", END)
    # Compile
    graph = graph_builder.compile()
    return graph
