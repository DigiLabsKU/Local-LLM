from markitdown import MarkItDown
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import tiktoken
from langchain_core.documents import Document
import os
import gc
from torch.cuda import empty_cache, is_available
from langdetect import detect
from model_configuration import load_json, save_json
import ntpath
from typing_extensions import Literal, List, Tuple, Callable
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer

# Helper functions
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def is_valid_chunk(text: str, min_length: int=30, text_threshold: float=0.15) -> bool:

    # Check if chunk is too short
    if len(text) < min_length:
        return False

    # Check if text content meets given threshold
    letter_count = sum(c.isalpha() for c in text)
    ratio = letter_count / len(text)

    if ratio < text_threshold:
        return False
    
    return True

def free_resources_doc_parser():

    if is_available(): # GPU
        empty_cache()

    gc.collect()

    print("Freed resources for Document Parser: Marker-Pdf")

def create_converter():
    
    converter = PdfConverter(
    artifact_dict=create_model_dict(),
    )

    return converter

# Parsing Functions
def parse_pdf(converter: PdfConverter, filename: str) -> tuple[str, list, dict]:
    rendered = converter(filename)
    text, _, _ = text_from_rendered(rendered)

    return text

def parse_document(file_path: str):
    """
    Parses various other document formats and converts to Markdown. 

    Args:
        file_path : str
            The file path of the document to be parsed
    Returns:
        str : The parsed Markdown text
    """
    markitdown = MarkItDown()
    result = markitdown.convert(file_path)
    return result.text_content


def parse_pdf_llama(file_path: str, format: str='markdown') -> List[Document]:
    parser = LlamaParse(result_type=format, api_key=os.environ.get('LLAMA_CLOUD_API_KEY'))
    file_extractor = {".pdf": parser, 
                      ".txt": parser, 
                      ".docx" : parser, 
                      ".pptx" : parser, 
                      ".HTML" : parser,
                      ".xlsx" : parser,
                      }  # Associate the parser with the following file extensions
    try:
        documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor).load_data()
    except Exception as e:
        print(f"Failed to load documents: {e}\n")
        
        
    return documents

def parse_url(urls: List[str]) -> List[Document]:
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    md = MarkdownifyTransformer(strip=["a"])
    converted_docs = md.transform_documents(docs)
    return converted_docs

def token_len_fn(model_name: str) -> Callable[[str], int]:
    tokenizer = tiktoken.get_encoding('cl100k_base') if "gpt" in model_name else AutoTokenizer.from_pretrained(model_name)
    if "gpt" in model_name:
        def tiktoken_len(text: str):
            tokens = tokenizer.encode(
                text,
                disallowed_special=()
            )
            return len(tokens)
        return tiktoken_len
    else: # A model from huggingface
        def token_len(text: str):
            tokens = tokenizer.encode(
                text,
                add_special_tokens=False
            )
            return len(tokens)
        return token_len

def chunk_text(text: str, token_fn: Callable[[str], int], chunk_size: int=1024, chunk_overlap: int=256) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_fn,
        separators= ['\n \n', '\n', ' ', ''],
    )
    splits = splitter.split_text(text)
    return splitter.create_documents(splits)

# Creating Pipeline
def parse_pipeline(model_name:str, files: List[str], urls: List[str]=[], parsing_method: Literal["local", "llama_index"] = "local") -> Tuple[List[Document], List[str]]:
    """
    Parses and chunks a list of PDF files using the provided converter and enrichment method.
    
    Args:
        model_name : str
            The name of the model to use for tokenization/conversation
        files : List[str]
            A list of file paths
        urls: List[str]
            An optional list of urls to parse 
        parsing_method : str
            A string specifying the method to use for parsing the PDFs, either "local" or "llama_index". Defaults to "local". 
        
    Returns
        List[Document]: 
            A list of chunks with additional metadata as Langchain Documents. 
        List[str]:
            A list of languages detected from the contents of the documents. 

    """
    token_fn = token_len_fn(model_name)
    languages = []

    if parsing_method == "local": 
        documents : List[Document] = []
        parse_attempts: int = 3

        for i in range(parse_attempts): 
            try:
                print("Parsing attempt: ", i) 
                combined = '\t'.join(files)
                if ".pdf" in combined: 
                    converter = create_converter()
                for fname in files:
                    file_extension = os.path.splitext(fname)[1].lower()
                    file_name = path_leaf(fname)
                    if file_extension == ".pdf":
                        text = parse_pdf(converter, fname)
                    else:
                        text = parse_document(fname)
                    chunks = chunk_text(text, token_fn, chunk_size=1024, chunk_overlap=256)
                    for doc in chunks:
                        if is_valid_chunk(doc.page_content):
                            doc.metadata['file_path'] = fname
                            doc.metadata['file_name'] = file_name
                            doc.metadata['language'] = detect(doc.page_content)
                            if doc.metadata['language'] not in languages: languages.append(doc.metadata['language'])
                        else:
                            continue
                    documents.extend(chunks)
            except Exception as e:
                print(f"Parsing locally failed due to: {e}")
                continue
            else:
                break

    else:
        documents : List[Document] = []
        for fname in files:
            try: 
                docs = parse_pdf_llama(fname)
            except Exception as e:
                # LlamaParse failed the first time, so trying once more.
                print(f"LlamaParse error raised: {e}")
                docs = parse_pdf_llama(fname)
            if docs: 
                text = ''.join(doc.text for doc in docs)
                chunks = chunk_text(text, token_fn, chunk_size=1024, chunk_overlap=256)
                for doc in chunks:
                    if is_valid_chunk(doc.page_content):
                        doc.metadata['language'] = detect(doc.page_content)
                        doc.metadata['file_name'] = fname
                        if doc.metadata['language'] not in languages and doc.metadata['language'] is not None: languages.append(doc.metadata['language'])
                    else:
                        continue
                documents.extend(chunks)
            # If LlamaParse failed -> fall back to local
            else:
                print("LlamaParse failed again to parse the documents falling back to local parsing method.\n")
                combined = '\t'.join(files)
                if ".pdf" in combined: 
                    converter = create_converter()

                for fname in files:
                    file_extension = os.path.splitext(fname)[1].lower()
                    file_name = path_leaf(fname)
                    if file_extension == ".pdf":
                        text = parse_pdf(converter, fname)
                    else:
                        text = parse_document(fname)
                    chunks = chunk_text(text, token_fn, chunk_size=1024, chunk_overlap=256)
                    for doc in chunks:
                        if is_valid_chunk(doc.page_content):
                            doc.metadata['file_path'] = fname
                            doc.metadata['file_name'] = file_name
                            doc.metadata['language'] = detect(doc.page_content)
                            if doc.metadata['language'] not in languages: languages.append(doc.metadata['language'])
                        else:
                            continue
                    documents.extend(chunks)

    # Parse urls if present
    if urls: 
        urls_documents = parse_url(urls)
        for doc in urls_documents:
            doc.metadata['file_name'] = "URL"
            doc.metadata['language'] = detect(doc.page_content)
            if doc.metadata['language'] not in languages and doc.metadata['language'] is not None: languages.append(doc.metadata['language'])
            documents.append(doc)
        documents.extend(urls_documents)

    # Saving the list of unique document languages
    config = load_json("config.json")
    config["languages"] = languages
    save_json("config.json", config)

    return documents, languages

if __name__ == "__main__":
    parse_pipeline(files=["data/ComIt_MA_2022.pdf"], 
                   model="gpt-4o",
                   parsing_method="local")