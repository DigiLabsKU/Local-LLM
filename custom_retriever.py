from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
import json

class CustomRetriever(BaseRetriever):

    docs: List[Document]
    metadata_info: Dict[str, Any]
    llm_model: OllamaLLM
    vectorstore: FAISS
    k: int = 5
    verbose: bool = False

    @classmethod
    def from_llm(cls, docs: List[Document], metadata_info: Dict[str, Any], llm_model: OllamaLLM, vectorstore: FAISS, k: int=5, verbose: bool=False):
        """
        Initializes a new instance of CustomRetriever from the required arguments. 

        Args:
            docs (List[Document]): List of documents to be used for retrieval. 
            metadata_info (Dict[str, Any]): Metadata information associated with the documents.
            llm_model (OllamaLM): Ollama language model for generating prompts.
            vectorstore (FAISS): Vectorstore for storing and searching for documents.
            k (int): Number of documents to return from the vectorstore.
            verbose (bool): Whether to print verbose output.

        Returns:
            CustomRetriever: An instance of CustomRetriever initialized with the provided arguments.
        """
        return cls(docs=docs, metadata_info=metadata_info, llm_model=llm_model, vectorstore=vectorstore, k=k, verbose=verbose)

    def construct_query_and_filters(self, query: str, metadata_info: Dict[str, str]) -> tuple[str, Dict[str, str]]:
        """
        Constructs a query and extracts filters using the LLM, with detailed metadata context.
        
        Args:
            query (str): The user's input query.
            metadata_info (Dict[str, Any]): Metadata fields and their descriptions with examples.
        
        Returns:
            tuple[str, Dict[str, str]]: The constructed query and extracted filters.
        """

        # Format the metadata_info into a readable prompt section
        metadata_details = "\n".join(
            [f"{field}: {info}" for field, info in metadata_info.items()]
        )
        
        # Create the prompt
        prompt = (
            f"Given the following query:\n\n{query}\n\n"
            f"and the metadata fields and their descriptions:\n\n{metadata_details}\n\n"
            f"Extract the following:\n"
            f"1. A refined query for retrieving relevant documents.\n"
            f"2. Any filters in the format of {{field: value}} that can help narrow down the search.\n\n"
            f"Respond with the refined query and filters in JSON format as:\n"
            f"{{'query': '<refined_query>', 'filters': {{'field1': 'value1', 'field2': 'value2'}}}}.\n"
        )
        
        # Use the LLM to generate the response
        response = self.llm_model.invoke(prompt)
        if self.verbose:
            print(f"Prompt sent to LLM:\n{prompt}")
            print(f"Response from LLM:\n{response}")
        
        # Parse the response into structured data
        try:
            result = json.loads(response)  # Assuming the LLM returns well-formed JSON
            refined_query = result.get("query", query)  # Default to the original query if missing
            filters = result.get("filters", {})
        except json.JSONDecodeError:
            refined_query = query
            filters = {}
        
        return refined_query, filters



    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents from the FAISS index using the given query based on the metadata information.
        Uses an LLM to filter extract filters from the given query based on the metadata information. 

        Args:
            query (str): The user's query.
        
        Returns: 
            List[Document]: A list containing up to k documents relevant to the user's query, filtered based on the metadata of the documents.
        """

        generated_query, filters = self.construct_query_and_filters(query)
        results = self.vectorstore.similarity_search_with_score(generated_query, k=self.k, filter=filters)
        relevant_docs = []
        for doc, score in results:
            if score > 0: # Ensuring somewhat similarity
                relevant_docs.append(doc)
        return relevant_docs