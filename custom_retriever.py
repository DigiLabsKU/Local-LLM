from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_ollama import OllamaLM
from langchain_community.vectorstores import FAISS

class CustomRetriever(BaseRetriever):

    docs: List[Document]
    metadata_info: Dict[str, Any]
    llm_model: OllamaLM
    vectorstore: FAISS
    k: int = 5

    @classmethod
    def from_llm(cls, docs: List[Document], metadata_info: Dict[str, Any], llm_model: OllamaLM, vectorstore: FAISS, k: int=5):
        """
        Initializes a new instance of CustomRetriever from the required arguments. 

        Args:
            docs (List[Document]): List of documents to be used for retrieval. 
            metadata_info (Dict[str, Any]): Metadata information associated with the documents.
            llm_model (OllamaLM): Ollama language model for generating prompts.
            vectorstore (FAISS): Vectorstore for storing and searching for documents.
            k (int): Number of documents to return from the vectorstore.

        Returns:
            CustomRetriever: An instance of CustomRetriever initialized with the provided arguments.
        """
        return cls(docs=docs, metadata_info=metadata_info, llm_model=llm_model, vectorstore=vectorstore, k=k)

    def construct_query(self, query: str, metadata_info: Dict[str, Any]) -> str:
        """
        Constructs a query based on the users input using an LLM Model.  
        
        Args:
            query (str): The user query. 
            metadata_info (Dict[str, Any]): Metadata information associated with the documents.
        
        Returns:
            str: A query that can be used to query the VectorStore for relevant documents.
        """
        pass

    def get_filters(self, query: str):
        """
        Uses an LLM model to retrieve the metadata information from the user query
        
        Args:
            query (str): The user's query for metadata information.
        
        Returns:
            Dict[str, str]: A dictionary of filters to be passed into the VectorStore similarity search. 
        """
        pass


    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents from the FAISS index using the given query based on the metadata information.
        Uses an LLM to filter extract filters from the given query based on the metadata information. 

        Args:
            query (str): The user's query.
        
        Returns: 
            List[Document]: A list containing up to k documents relevant to the user's query, filtered based on the metadata of the documents.
        """

        generated_query = self.construct_query(query)
        filters = self.get_filters(query)
        results = self.vectorstore.similarity_search_with_score(generated_query, k=self.k, filter=filters)
        relevant_docs = []
        for doc, score in results:
            if score > 0: # Ensuring somewhat similarity
                relevant_docs.append(doc)
        return relevant_docs