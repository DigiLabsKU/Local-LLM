# Local-LLM (In-progress)

A completely local RAG pipeline allowing the user to be able to chat with their documents, without any data being collected from third parties, such as LLM-API's or other libraries used. 

## Build Instructions (Updated along the way)

* Setup conda environment: `conda create -n local-llm python=3.11.5`
* Activate conda environment: `conda activate local-llm`
* Install `uv`: `pip install uv`
* Install PyTorch with CUDA ([PyTorch](https://pytorch.org/get-started/locally/)): 

    `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

* Install `marker-pdf`: `uv pip install marker-pdf` 

## Features

* Completely private (hopefully this implies that we won't need internet connection to run this program)
* Use a local LLM for chatting (Model yet to be determined)
* Use a local LLM for embeddings (Model yet to be determined)
* Refined metadata structure when processing documents (Processes the pdf documents and tags it with relevant metadata that can help provide the LLM with useful insights when answering user queries). 
    * This must be done automatically, one potential idea is to summarize the documents and analyze the structure. 
* Memory for a smoother conversation (Memory is only supposed to be kept when in session, removed completely afterwards). 
* Interactive UI for the users, for easier usage for users not familiar with CLI. 

## Areas of focus

* Processing the documents using some local method, which is not only efficient but also preserves the quality of the document i.e. content and structure  (something like a pdf -> markdown parser). 
* Automatically tagging the documents with relevant metadata that could help the model to familiarize itself with the contents of the documents. 
* Developing a user friendly UI, no need for anything fancy -> going for a simplistic approach emphasizing efficiency and ease of use. 

## Notes

* Local LLM's probably needs some efficient library, maybe something like llama-cpp allowing for optimized usage of local models?
* Faiss-GPU is most likely going to be used, since it supports GPU-accelerated Vector-database that are stored in-memory.
* Maybe LangChain? (If it doesn't slow down the program too much) Could be handy for creating a conversation chain + memory. 
