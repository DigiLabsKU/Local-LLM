from vectorstore import load_existing_vector_store, vectorstore_pipeline
from local_embeddings import LocalEmbeddings
from langchain_community.vectorstores import FAISS
from custom_retriever import CustomRetriever
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain
from langchain_ollama import OllamaLLM
import os

# Loading LLM Model
llm_model_name = ""
llm = OllamaLLM(
    model_name=llm_model_name,
    temperature=0.3,
    max_tokens=1000,
    num_gpu=1,
)

# Embeddings
embeddings_model_name = "paraphrase-multilingual-mpnet-base-v2"
embeddings_model = LocalEmbeddings(embeddings_model_name)

# Vectorstore (Load existing if there is alread)
store_path = "vectorstore_index.faiss"
files = [""]
if not os.path.exists(store_path):
    vectorstore = vectorstore_pipeline(embeddings_model_name=embeddings_model_name, 
                                       llm_model=llm_model_name, 
                                       file_paths=files, 
                                       enrich_method="keywords", 
                                       store_path=store_path,
                                       use_gpu=False)
    print("Created new vectorstore")
else:
    vectorstore = load_existing_vector_store(embeddings_model_name, store_path, use_gpu=False)

# Metadata Info

metadata_info = {
    "title": "title",
    "source": "source_url",
    "keywords": "keywords",
}

# Custom Retriever
retriever = CustomRetriever.from_llm(
    docs=None,
    metadata_info=metadata_info,
    llm_model=llm,
    vectorstore=vectorstore,
    k=5,
    verbose=True,
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")
read_only_memory = ReadOnlySharedMemory(memory=memory)

def clear_mem():
    memory.clear()

# Create the RetrievalQA chain with memory
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    memory=read_only_memory)

# Tools
tools = None

# Agent Prompt
prompt = None

# Agent Creation Logic

llm_chain = LLMChain(llm=llm, prompt=prompt)
def create_agent():
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True, 
        return_intermediate_steps=False, 
        handle_parsing_errors=True
    )
    return agent_chain
