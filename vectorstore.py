from local_embeddings import LocalEmbeddings
from langchain_openai import OpenAIEmbeddings
from doc_parser import parse_pipeline, free_resources_doc_parser
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
from langchain_core.documents import Document
from typing_extensions import List, Dict, Literal
from collections import defaultdict

class CustomMultiVectorStore:
    def __init__(self, embeddings_model_name: str, languages: List[str]=[], use_gpu=False):
        self.vectorstores : Dict[str, FAISS] = {}
        self.embeddings_model_name : str = embeddings_model_name
        self.use_gpu : bool = use_gpu
        self.languages : List[str] = languages

    def create_vectorstores(self, documents: List[Document], languages: List[str]) -> None:
        """
        Creates a vector store for each language in the provided list of documents.

        Args:
            documents (List[Document]) : The documents to be added to the vector stores.
            languages (List[str]) : The languages of the documents.
        Returns:
            None
        """
        docs_by_lang = defaultdict(list)
        for doc in documents:
            lang = doc.metadata.get("language", "unknown")
            docs_by_lang[lang].append(doc)
        
        for lang in languages:
            if lang not in docs_by_lang:
                print(f"No documents found for language '{lang}' – skipping creation.")
                continue

            vector_store = create_vectorstore(self.embeddings_model_name, self.use_gpu)
            self.vectorstores[lang] = vector_store
            self.languages.append(lang)
            lang_docs = docs_by_lang[lang]
            store_name = f'vector_stores/{lang_docs[0].metadata["language"]}_vector_store'
            add_documents(vector_store, lang_docs, save=store_name)
            print(f"Created vector store for {lang}")
    
    def load_vectorstores(self, languages: List[str]) -> None:
        """
        Loads the specified vector stores from their respective save locations.

        Args:
            languages (List[str]) : The languages of the vector stores to be loaded. Assumes `vectorstores` are stored in a directory called vectorstores with each vectorstore being named `{lang}_vector_store`.
        Returns:
            None
        """
        for lang in languages:
            store_name = f'vector_stores/{lang}_vector_store'
            self.vectorstores[lang] = load_vectorstore(self.embeddings_model_name, store_path=store_name, use_gpu=self.use_gpu)
            print(f"Loaded vector store for {lang}")
    
    def delete_vectorstore(self, lang: str) -> None:
        if lang in self.vectorstores:
            del self.vectorstores[lang]
            print(f"Deleted vector store for language: {lang}")
        else:
            print(f"No vector store found for language: {lang}")
    
    def extend_vectorstore(self, documents: List[Document]) -> None:
        """
        Extends the current MultiVectorStore Instance with the given list of documents. Creates new vector store if a document with a new language is encountered. 

        Args: 
            List[Document] : The list of documents to add. 
        
        Returns: 
            None. 
        """

        # Sort the documents by language
        docs_by_lang = defaultdict(list)
        for doc in documents:
            lang = doc.metadata.get("language", "unknown")
            docs_by_lang[lang].append(doc)
        
        extended_langs = docs_by_lang.keys()
        
        # Check if new languages are present
        for lang in docs_by_lang.keys():
            if lang not in self.languages:
                print(f"The following language: {lang} was not already present, creating a new vector store for it.")

                # Add language to universal list
                self.languages.append(lang)

                # Create vector store
                vector_store = create_vectorstore(self.embeddings_model_name, self.use_gpu)
                self.vectorstores[lang] = vector_store

                # Add documents to the vector store
                lang_docs = docs_by_lang[lang]
                store_name = f'vector_stores/{lang_docs[0].metadata["language"]}_vector_store'
                add_documents(vector_store, lang_docs, save=store_name)

                # Remove the docs from dict
                del docs_by_lang[lang]
        
        # Extend already existing vector store(s)
        for lang in docs_by_lang.keys():
            add_documents(self.vectorstores[lang], docs_by_lang[lang], save=f"vectorstores/{lang}_vector_store", use_gpu=self.use_gpu)
        
        print(f"Extended the vector stores of the following langs {extended_langs}")

    def query_vectorstore(self, query: str, lang: str, k: int=4) -> None | List[Document]:
        """
        Queries the specified vector store for relevant documents.
        
        Args:
            query (str) : The user's query
            lang (str) : The language of the vector store
            k (int) : The number of relevant documents to return. Defaults to 4.

        Returns:
            (None | List[Document]) : A list of relevant documents (langchain documents) or None if no relevant documents found.
        """

        if lang in self.vectorstores:
            retriever = self.vectorstores[lang].as_retriever(search_type="similarity", search_kwargs={'k': k})
            results = retriever.invoke(query)
            return results
        else:
            print(f"No vector store found for language: {lang}")
            return None
    
    # Extra Utility Methods
    def num_vectorstores(self):
        return len(self.vectorstores)

def create_vectorstore(embeddings_model_name: str, use_gpu: bool=False) -> FAISS:
    if "text-embedding-3" in embeddings_model_name:
        embeddings_model = OpenAIEmbeddings(model=embeddings_model_name)
        dim = len(embeddings_model.embed_query("Hello"))
    else:
        embeddings_model = LocalEmbeddings(embeddings_model_name)
        dim = embeddings_model.get_dimensions()

    # Create an in-memory faiss vectorstore
    index_flat = faiss.IndexFlatL2(dim)
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Creating in-memory faiss vectorstore with GPU support")
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
    # Langchain FAISS vector store
    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=gpu_index_flat if use_gpu else index_flat,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store

def add_documents(vector_store, documents: List[Document], save: str="vectorstores/en_vector_store", use_gpu: bool=False) -> None:
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents, ids=uuids)
    
    if save:
        # Ensure the index is on CPU before saving
        if use_gpu:
            print("Moving the GPU index to CPU for serialization...")
            vector_store.index = faiss.index_gpu_to_cpu(vector_store.index)
        
        vector_store.save_local(save)
        
        # Move it back after saving
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Moving the index back to GPU...")
            res = faiss.StandardGpuResources()
            vector_store.index = faiss.index_cpu_to_gpu(res, 0, vector_store.index)


def load_vectorstore(embeddings_model_name: str, store_path: str="en_vector_store", use_gpu: bool=False) -> FAISS:
    if "text-embedding-3" in embeddings_model_name:
        embeddings_model = OpenAIEmbeddings(model=embeddings_model_name)
    else:
        embeddings_model = LocalEmbeddings(embeddings_model_name)
    faiss_store = FAISS.load_local(store_path, embeddings_model, allow_dangerous_deserialization=True)
    # Move to GPU if requested
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Transferring Loaded Index to GPU")
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, faiss_store.index)
        faiss_store.index = gpu_index_flat
    return faiss_store

def vectorstore_pipeline(embeddings_model_name: str, llm_model_name: str, file_paths: list[str], parsing_method: Literal["local", "llama_index"] = "local", use_gpu: bool=False) -> CustomMultiVectorStore:
    """
    Pipeline for creating a CustomMultiVectorStore instance containing a vector store for each language in the provided files. 
    
    Args:
        embeddings_model_name : str
            The name of the local embeddings model to use.
        llm_model_name : str
            The name of the llm model to use for the RAG.
        file_paths : List[str]
            The file paths of the PDFs to be parsed.
        parsing_method : Literal
            Which parsing method to use, either "local" or "llama_index". Defaults to "local". 
        use_gpu : bool, optional
            - If True, use GPU for vectorization else CPU for vectorization. 
            - Default: False
    
    Returns:
        CustomMultiVectorStore : A class containing multiple vector stores in different languages. 
    """
    
    documents, languages = parse_pipeline(file_paths, llm_model_name, parsing_method=parsing_method)
    vector_store = CustomMultiVectorStore(embeddings_model_name=embeddings_model_name, use_gpu=use_gpu)
    vector_store.create_vectorstores(documents, languages)
    
    # Free memory
    del documents
    if parsing_method == 'local':
        free_resources_doc_parser()

    return vector_store


def extend_multi_vector_store(multi_vector_store: CustomMultiVectorStore, llm_model_name: str, file_paths: List[str], parsing_method: Literal["local", "llama_index"] = "local") -> CustomMultiVectorStore:
    """
    Extend an existing CustomMultiVectorStore instance with new documents. 
    
    Args:
        multi_vector_store : CustomMultiVectorStore
            The multi_vector_store instance to extend
        llm_model_name : str
            The name of the llm model to use for the RAG.
        file_paths : List[str]
            The file paths of the PDFs to be parsed.
        parsing_method : Literal
            Which parsing method to use, either "local" or "llama_index". Defaults to "local". 
    
    Returns:
        CustomMultiVectorStore : The extended multi_vector_store instance.
    """

    documents, _ = parse_pipeline(file_paths, llm_model_name, parsing_method=parsing_method)
    multi_vector_store.extend_vectorstore(documents)
    
    # Free memory
    del documents
    if parsing_method == 'local':
        free_resources_doc_parser()
    
    return multi_vector_store