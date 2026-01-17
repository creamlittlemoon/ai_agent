"""

20260116
update: add RAG in workflow
main goal: retrieve document using agent

"""


from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
# new packages below (rag related)
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        TokenTextSplitter
    )
from langchain_chroma import Chroma
from langchain_core.tools import tool


load_dotenv()

MODEL = "gpt-4.1-nano"
BASE_URL = "https://api.zhizengzeng.com/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
# this is a global variable to store document content
document_content = ""


model = ChatOpenAI(
    model = MODEL,
    api_key= os.getenv("ZZZ_API_KEY"),
    base_url=BASE_URL,
    temperature=0   #🚩new! minimize hallucination
).bind_tools(tools)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  #⚠️embedding model must be compatible with the LLM
)

pdf_path = "Stock_Market_Performance_2024.pdf"






