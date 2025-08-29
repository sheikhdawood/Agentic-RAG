import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_community.vectorstores import SupabaseVectorStore

from supabase.client import Client, create_client
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
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
    temperature=0,
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

response = agent_executor.invoke({"input": "why is agentic rag better than naive rag?"})

print(response["output"])