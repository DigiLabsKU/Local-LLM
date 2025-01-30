from langchain_ollama import ChatOllama
from custom_retriever import CustomRetriever
from local_embeddings import LocalEmbeddings
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.tools import tool
from vectorstore import vectorstore_pipeline, load_existing_vectorstore
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

from PIL import Image


# Initialize components as before
llm_model_name = "llama3.2"
llm = ChatOllama(
    model=llm_model_name,
    temperature=0.1,
    max_tokens=1000,
    num_gpu=1,
)

embeddings_model_name = "all-MiniLM-L6-v2"
embeddings_model = LocalEmbeddings(embeddings_model_name)

# Create a new vectorstore instance
# vectorstore = vectorstore_pipeline(embeddings_model_name=embeddings_model_name,
#                                   llm_model="meta-llama/Llama-3.2-3B",
#                                   file_paths=["DL_Assignment_4.pdf"],
#                                   store_path="faiss_index",
#                                   enrich_method="keywords")
# Load existing vectorstore

vectorstore = load_existing_vectorstore(embeddings_model_name, "faiss_index", use_gpu=False)

metadata_info = {
    "page_id": "The page id of the document or chunk.",
    "source": "The source file (i.e., path) of the document.",
    "keywords": "Topics extracted from the query, listed as a list of words."
}

# Retriever

retriever = CustomRetriever(
    docs=[],
    vectorstore=vectorstore,
    llm_model=llm,
    metadata_info=metadata_info,
    k=5,
    verbose=True,
)

# Retrieval Tool
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Retrieve information related to query"""
    retrieved_documents = retriever._get_relevant_documents(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata['source']}\n" f"Content: {doc.page_content}")
        for doc in retrieved_documents
    )
    return serialized, retrieved_documents



# Building RAG using Lang Graph
graph_builder = StateGraph(MessagesState)

# Step 1: Generate AI Message that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond"""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Step 2: Execute Retrieval
tools = ToolNode([retrieve])

# Step 3: Generate Response using the Retrieved Content
def generate(state: MessagesState):
    """Generate Answer"""
    # Get generated tool messages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    
    tool_messages = recent_tool_messages[::-1]

    # Format into proper prompt: 
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answers concise."
        "\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


# Compiling into a single Graph
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"}, # If we don't use tool, then directly to response node else continue to the tool calling node
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# Display Graph Image
# graph_image_pth = "graph_image.png"
# with open(graph_image_pth, "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())

# img = Image.open(graph_image_pth)
# img.show()


input_msg = "Hello"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_msg}]},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()