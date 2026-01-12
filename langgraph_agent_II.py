"""
Objectvies
1. use different message types: HumanMessage, AIMessage
2. history maintenance

Clarification
in this code, we use openai to call models. but actually we can directly use chatopenai from langchain. zhizengzeng supports that.
"""

import os
import requests
from typing import TypedDict, List, Union
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4.1-nano"
BASE_URL = "https://api.zhizengzeng.com/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ZZZ_API_KEY")
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
    

def build_agent(api_key: str) -> StateGraph:
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )
    def to_openai_messages(messages: List[Union[HumanMessage, AIMessage]]) -> List[dict]:
        payload = []
        for message in messages:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            payload.append({"role": role, "content": message.content})
        return payload

    def to_openai_messages(messages: List[Union[HumanMessage, AIMessage]]) -> List[dict]:
        payload = []
        for message in messages:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            payload.append({"role": role, "content": message.content})
        return payload

    def process(state: AgentState) -> AgentState:
        """ This node will solve the request you input"""
        response = client.chat.completions.create(
            model=MODEL,
            messages=to_openai_messages(state["messages"]),
        )
        content = response.choices[0].message.content or ""
        state["messages"].append(AIMessage(content=content))
        
        return state

    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph.compile()


def log_conversation(
    conversation_history: List[Union[HumanMessage, AIMessage]],
    file_path: str = "logging.txt",
) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("Your Conversation Log:\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"AI: {message.content}\n\n")
        file.write("End of Conversation\n")
    
        

def main() -> None:
    api_key = load_api_key()
    send_test_request(api_key)
    
    agent = build_agent(api_key)
    conversation_history = []
    
    user_input = input("Enter: ")
    while user_input != 'exit':
        conversation_history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": conversation_history})
        conversation_history = result["messages"]
        print(f'conversation history: {conversation_history}')
        for msg in conversation_history[-2:]:  # print only the last user and AI messages
            role = "You" if isinstance(msg, HumanMessage) else "AI"
            print(f"{role}: {msg.content}")
        log_conversation(conversation_history)
        
        
        
        
        print('--------𝜗ৎ--------------𝜗ৎ---------------𝜗ৎ---------')
        user_input = input("Enter: ")
        
    
    
    
    
if __name__ == "__main__":
    main()



