"""
20260112
Agent name
    Drafter
    
Task
    We spend too much time drafting documents and this needs to be fixed!
    
Reminder
    This agent structure is different from ReAct. In this structure, tool node can directly point to END.

"""


import os 
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

MODEL = "gpt-4.1-nano"
BASE_URL = "https://api.zhizengzeng.com/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
# this is a global variable to store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Updates the document with the provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the proces
    
    Args:
        filename: Name for the text file.
    """
    
    global document_content
    
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
        
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'"
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]

model = ChatOpenAI(
    model = MODEL,
    api_key= os.getenv("ZZZ_API_KEY"),
    base_url=BASE_URL,
).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
        
        - If the user wants to update or modify content, use the 'update' tool with the complete updated content.content_blocks=
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after modifications.
        
        The current document content is :{document_content}                                  
        """)
    
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
        
    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"\n UWER: {user_input}")
        user_message = HumanMessage(content=user_input)
        
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    
    response = model.invoke(all_messages)
    
    print(f"\n 🤖AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" 🔧USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
        
    return {"messages": list(state["messages"]) + [user_message, response]}



def should_continue(state:AgentState) -> str:
    """Determine if we should continue or end the conversation"""
    
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # this looks for the most recent tool message...
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save   (if it's save, it means that the user doesn't want to do any other actions)
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end"    #goes to the end edge which leads to the endpoint

    return "continue"   #if it's "update", always continue


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return 
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🔧TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n -----𝜗ৎ----- DRAFTER LAUNCHED-----𝜗ৎ-----")
    
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n -----𝜗ৎ----- DRAFTER FINISHED -----𝜗ৎ-----")


if __name__ == "__main__":
    run_document_agent()





