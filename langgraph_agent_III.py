"""
20260112
ReAct agent
(reasoning and acting)
Objectives:
1. create tools in langgraph
2. create a ReAct graph
3. work with different types of Messages such as ToolMessage
"""

import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, SystemMessage #BaseMessage is the parent class for other messages like HumanMessage, AIMessage and etc.
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.graph.message import add_messages    #it's a reducer function. it makes us append everything to the status. without it, the status will only have the newest data, but no old data.
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

load_dotenv()

MODEL = "gpt-4.1-nano"
BASE_URL = "https://api.zhizengzeng.com/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"

# Annotated - provides additional context without affecting the type itself
email = Annotated[str,"This has to be a valid email format!"]
print(email.__metadata__)   #annotation added to the metadata

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def add(a: int, b:int):
    """This is an addition functionthat adds two numbers together"""    #⭐️if it's used for llm, this docstring is a must!! or the llm won't know what the tools does
    return a + b

@tool
def substract(a:int, b:int):
    """ Substraction function """
    return a - b

@tool
def multiply(a:int, b:int):
    """ Multiplication function """
    return a * b 

tools = [add, substract, multiply]   #⭐️if my goal is to do maths, then the add tool here is not optional, is a must. because if you let llm do maths, it's just guessing, or to say 99.99% confidence. not 100%.


model = ChatOpenAI(
    model = MODEL,
    api_key= os.getenv("ZZZ_API_KEY"),
    base_url=BASE_URL,
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You're my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}



def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)  #⭐️here by printing, we can intuitively check if llm has called our tool or not.
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 3 + 4")]}
print_stream(app.stream(inputs, stream_mode="values"))


















# def main() -> None:
#     api_key = load_api_key()
#     send_test_request(api_key)





