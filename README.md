# Local-LLM (In-progress)

A completely local RAG pipeline allowing the user to be able to chat with their documents, without any data being collected from third parties, such as LLM-API's or other libraries used. 

## Build Instructions (Updated along the way)

### Conda Environment 
* Setup conda environment: `conda create -n local-llm python=3.11.5`
* Activate conda environment: `conda activate local-llm`

### Necessary Libraries
* Install `uv`: `pip install uv`
* Install PyTorch with CUDA ([PyTorch](https://pytorch.org/get-started/locally/)), replace `cu118` with your CUDA version: 

    `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

* Install necessary libraries:
    `uv pip install tiktoken langgraph langchain_community marker-pdf langchain langchain_ollama transformers sentence_transformers langchain_openai streamlit llama-index langdetect`

* Alternatively, you could also use the `requirements.txt` and install via. `pip install -r requirements.t  xt`

### API Requirements
* If you intend to use Chat-GPT models such as 4o or 4o-mini, then make sure to set an environment variable `OPENAI_API_KEY=YOUR_KEY_GOES_HERE` with your API-key. 
* The same applies if you intend to use the parsing API Llama-Index, then set an environment variable `LLAMA_CLOUD_API_KEY=YOUR_KEY_GOES_HERE` with your API-key.

### Running the Application

* In terminal, navigate to the project-folder (where `main.py` is located) and run `streamlit run main.py`.

## Features (Updated along the way)
* Local open-source models such as LLama-3.1, LLama-3.2 as well as LLama-3.3.
* Use of OpenAI models, currently GPT-4o, GPT-4o-mini.
* Local embeddings models for multilingual capabilities, such as `all-MiniLM-L6-v2` and `multilingual-e5-large-instruct`. 
* Use of OpenAI embeddings models, currently `text-embedding-3-small` and `text-embedding-3-large`.
* Local parsing method using `marker-pdf`, levaraging GPU to parse PDF-files efficiently to markdown format incl. tables.
* Use of LLama-Parse, a solution by LLama-Index (see [LLama-Parse](https://docs.cloud.llamaindex.ai/llamaparse/getting_started)), uses LLM to parse PDF files into markdown.

## Currently Working On
* Support for multiple document formats such as text files (.txt), word documents (.docx), powerpoints (.pptx) and urls fx. links to articles/blogs etc.
* Extending existing vector store by adding more documents (instead of creating new from scratch).
* Adding sources in response to show which text passages were used for answering the question. 

## Planned Updates
* Support for own LLM models in GGUF format from fx. HuggingFace
* Support for own embeddings models from fx. HuggingFace
* Adding more OpenAI models.
* Refining UI by moving to some better framework (Streamlit is for prototyping). 
