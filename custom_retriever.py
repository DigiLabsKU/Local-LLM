from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
import json
import re

class CustomRetriever(BaseRetriever):

    docs: List[Document]
    metadata_info: Dict[str, Any]
    llm_model: ChatOllama
    vectorstore: FAISS
    k: int = 5
    verbose: bool = False

    @classmethod
    def from_llm(cls, docs: List[Document], metadata_info: Dict[str, Any], llm_model: ChatOllama, vectorstore: FAISS, k: int=5, verbose: bool=False):
        """
        Initializes a new instance of CustomRetriever from the required arguments. 

        Args:
            docs (List[Document]): List of documents to be used for retrieval. 
            metadata_info (Dict[str, Any]): Metadata information associated with the documents.
            llm_model (ChatOllama): Ollama language model for generating prompts.
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
            f"You are an expert at extracting structured metadata filters from user queries.\n\n"
            f"Given this query:\n"
            f"Query: \"{query}\"\n\n"
            f"Extract only the following metadata fields based on user intent:\n"
            f"{json.dumps(metadata_info, indent=2)}\n\n"
            f"**Rules:**\n"
            f"- Only extract values for keys explicitly listed in the metadata fields.\n"
            f"- Do not infer new keys beyond those specified.\n"
            f"- If a field is not present in the query, do not include it in the output.\n"
            f"- Maintain the correct data type for each field (e.g., `page_id` should be an integer if applicable).\n\n"
            f"**Example Queries and Expected Filter Formats:**\n"
            f"- Query: 'Show me content from page 42'\n"
            f"- Expected filter: {{'page_id': 42}}\n\n"
            f"- Query: 'What does the article say about climate change?'\n"
            f"- Expected filter: {{'keywords': ['climate change']}}\n\n"
            f"Return a JSON object containing only the relevant metadata fields."
        )

        
        # Use the LLM to generate the response
        response = self.llm_model.invoke(prompt)
        if self.verbose:
            print(f"Prompt sent to LLM:\n{prompt}")
            print(f"Response from LLM:\n{response.content}")
        
        return response.content

    def process_llm_response(self, response):
        """
        Processes the LLM response and returns a structured data object containing the relevant metadata fields.
        
        Args:
            response (str): The LLM response containing JSON object containing the relevant metadata fields.
        
        Returns:
            Dict[str, Any]: A structured dictionary containing the filters used for comparison during simalirty search.
        """

        if(isinstance(response, str)):
            try:
                llm_response = json.loads(response)
            except json.JSONDecodeError:
                print("Error: Invalid JSON response from LLM")
                return {}
        
        # Only keep keys that in metadata_info and aren't None
        cleaned_response = {key: value for key, value in llm_response.items() if key in self.metadata_info}
        cleaned_response = {key: value for key, value in llm_response.items() if value is not None}
        
        # Ensure correct types
        if "page_id" in cleaned_response:
            try:
                cleaned_response["page_id"] = int(cleaned_response["page_id"])
            except (ValueError, TypeError):
                del cleaned_response["page_id"]  # Remove if conversion fails

        if "keywords" in cleaned_response:
            if isinstance(cleaned_response["keywords"], str):
                cleaned_response["keywords"] = [cleaned_response["keywords"]]  # Convert single string to list
            elif not isinstance(cleaned_response["keywords"], list):
                del cleaned_response["keywords"]  # Remove if invalid type

            cleaned_response["keywords"] = [keyword.lower() for keyword in cleaned_response["keywords"]]
        
        # Convert to vectorstore filter format
        filter_query = {key: {"$in": val} if isinstance(val, list) else val for key, val in cleaned_response.items()}
        print(filter_query)

        return cleaned_response, filter_query

    def filter_documents_based_on_keywords(documents, query_keywords):
        """Filters documents based on matching keywords."""
        filtered_docs = []
        for doc in documents:
            doc_keywords = [keyword.lower() for keyword in doc.get('keywords', [])]  # Ensure lowercase matching
            if any(query_keyword in doc_keyword for query_keyword in query_keywords for doc_keyword in doc_keywords):
                filtered_docs.append(doc)
        return filtered_docs

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents from the FAISS index using the given query based on the metadata information.
        Uses an LLM to filter extract filters from the given query based on the metadata information. 

        Args:
            query (str): The user's query.
        
        Returns: 
            List[Document]: A list containing up to k documents relevant to the user's query, filtered based on the metadata of the documents.
        """

        #llm_filters = self.construct_query_and_filters(query, self.metadata_info)
        #filter_dict = self.process_llm_response(llm_filters)
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        relevant_docs = []
        for doc, score in results:
            if score > 0: # Ensuring somewhat similarity
                relevant_docs.append(doc)
        return relevant_docs