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
            f"Follow these rules:\n"
            f"- Only return fields that exist in metadata_info.\n"
            f"- Do NOT infer values that are not explicitly stated in the query.\n"
            f"- Return the filters in **strict JSON format** (inside <<<JSON_START>>> and <<<JSON_END>>>).\n\n"
            f"Example Output:\n"
            f"<<<JSON_START>>>\n"
            f"{{'filters': {{'page_id': 42, 'source': 'sample.pdf'}}}}\n"
            f"<<<JSON_END>>>\n\n"
            f"Now, extract the metadata filters from the given query."
        )

        
        # Use the LLM to generate the response
        response = self.llm_model.invoke(prompt)
        if self.verbose:
            print(f"Prompt sent to LLM:\n{prompt}")
            print(f"Response from LLM:\n{response}")
        
        match = re.search(r'<<<JSON_START>>>(.*?)<<<JSON_END>>>', response, re.DOTALL)
        if match:
            json_data = match.group(1).strip()
            try:
                filters = json.loads(json_data)
            except json.JSONDecodeError:
                filters = {}
        else:
            filters = {}
        
        return filters



    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents from the FAISS index using the given query based on the metadata information.
        Uses an LLM to filter extract filters from the given query based on the metadata information. 

        Args:
            query (str): The user's query.
        
        Returns: 
            List[Document]: A list containing up to k documents relevant to the user's query, filtered based on the metadata of the documents.
        """

        filters = self.construct_query_and_filters(query, self.metadata_info)
        results = self.vectorstore.similarity_search_with_score(query, k=self.k, filter=filters)
        relevant_docs = []
        for doc, score in results:
            if score > 0: # Ensuring somewhat similarity
                print(doc)
                relevant_docs.append(doc)
        return relevant_docs