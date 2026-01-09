import os
from typing import List, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

MODEL = "gpt-4.1-nano"
CHAT_URL = "https://api.zhizengzeng.com/v1/chat/completions"

class AgentState(TypedDict):
    messages: List[HumanMessage]


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ZZZ_API_KEY")  #get api key from "exported environment variable" using cmd. if no, then from .env file
    if not api_key:
        raise ValueError("ZZZ_API_KEY not found.")
    return api_key


def send_test_request(api_key: str) -> None:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "tell me a color name."}],
    }
    response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    print(response.json())


def build_agent() -> StateGraph:
    llm = ChatOpenAI(model=MODEL)

    def process(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        print(f"\nAI: {response.content}")
        return state

    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph.compile()  #agent refers to the compiled graph


def main() -> None:
    api_key = load_api_key()
    send_test_request(api_key)

    agent = build_agent()
    user_input = input("Enter: ")
    while user_input != "exit": # keep talking as long as the user doesn't want to exit
        agent.invoke({"messages": [HumanMessage(content=user_input)]})
        user_input = input("Enter: ")


if __name__ == "__main__":
    main()
