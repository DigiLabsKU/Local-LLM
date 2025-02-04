from langchain_ollama import ChatOllama
from local_embeddings import LocalEmbeddings
from vectorstore import vectorstore_pipeline, load_existing_vectorstore
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

# CONFIG
llm_model_name = "llama3.2"
embeddings_model_name = "all-MiniLM-L6-v2"

# Load LLM and Embeddings Model
llm = ChatOllama(
    model=llm_model_name,
    temperature=0.1,
    max_tokens=1000,
)

embeddings_model = LocalEmbeddings(embeddings_model_name)

# Vectorstore
vectorstore = load_existing_vectorstore(embeddings_model_name, "faiss_index", use_gpu=False)

# Retriever
retriever = vectorstore.as_retriever()


# Tool
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Agent Memory
memory = MemorySaver()

# Agent
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
config = {"configurable" : {"thread_id" : "local-rag"}}

input_message = (
    "How many ECTS points is the course 'Communication and Cooperation in Organisations' worth?\n\n"
)

for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    print(event["messages"][-1])