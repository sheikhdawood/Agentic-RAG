import os
from dotenv import load_dotenv

import streamlit as st

from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from supabase.client import Client, create_client

load_dotenv()  

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)
 
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # you can also use "mixtral-8x7b-32768", "gemma-7b-it", etc.
    temperature=0.5,
    api_key=os.environ.get("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful RAG chatbot. Use the `retrieve` tool to answer questions using the vector store."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


user_question = st.chat_input("How are you?")

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))

    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))

