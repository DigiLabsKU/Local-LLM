from local_embeddings import LocalEmbeddings
from doc_parser import parse_pipeline, free_resources_doc_parser
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
from langchain_core.documents import Document

def create_vectorstore(embeddings_model_name: str, use_gpu: bool=False) -> FAISS:
    embeddings_model = LocalEmbeddings(embeddings_model_name)
    dim = embeddings_model.get_dimensions()

    # Create an in-memory faiss vectorstore
    index_flat = faiss.IndexFlatL2(dim)
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Creating in-memory faiss vectorstore with GPU support")
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
    # Langchain FAISS VectorStore
    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=gpu_index_flat if use_gpu else index_flat,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store

def add_documents(vector_store, documents: Document, save: bool=False) -> None:
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents, ids=uuids)
    if save:
        vector_store.save_local('faiss_index')


def load_existing_vectorstore(embeddings_model_name: str, store_path: str="faiss_index", use_gpu: bool=False) -> FAISS:
    embeddings_model = LocalEmbeddings(embeddings_model_name)
    faiss_store = FAISS.load_local(store_path, embeddings_model, allow_dangerous_deserialization=True)
    # Move to GPU if requested
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Transferring Loaded Index to GPU")
        res = faiss.StandardGpuRessources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, faiss_store.index)
        faiss_store.index = gpu_index_flat
    return faiss_store

def vectorstore_pipeline(embeddings_model_name: str, llm_model_name: str, file_paths: list[str], enrich_method:str, store_path: str, use_gpu: bool=False) -> FAISS:
    """
    Pipeline for creating a vector store using a local embeddings model, parsing documents, and adding them to the vector store.
    
    Args:
        embeddings_model_name : str
            The name of the local embeddings model to use.
        llm_model_name : str
            The name of the llm model to use for in the RAG.
        file_paths : list[str]
            The file paths of the PDFs to be parsed.
        enrich_method : str
            An optional string telling which method to use for enriching the chunks, i.e. "summarization" or "keywords". The latter is more cost-effective.
        store_path : str
            The path to save the faiss index after creation.
        use_gpu : bool, optional
            - If True, use GPU for vectorization.
            - If False, use CPU for vectorization.
            - Default: False
    
    Returns:
        A LangChain FAISS VectorStore loaded in the computers memory. 
    """
    
    documents = parse_pipeline(file_paths, llm_model_name, enrich_method='keywords')
    vector_store = create_vectorstore(embeddings_model_name, use_gpu)
    add_documents(vector_store, documents, save=True)
    
    # Free memory
    del documents
    free_resources_doc_parser()

    return vector_store