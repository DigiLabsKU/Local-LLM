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
import requests
import tempfile
import hashlib

CONVERTER = None
CONFIG_FILE = "config.json"

SUPPORTED_FORMATS = (".pdf", 
                     ".txt", 
                     ".docx", 
                     ".pptx", 
                     ".xlsx", 
                     ".HTML",)

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

def hash_document(doc: Document) -> str:
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()

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
        file_path (str) :
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
        documents = [] 
        
    return documents

def _parse_single_file(file_path: str, languages: List[str], token_fn: Callable[[str], int], parsing_method: Literal["local", "llama_index"] = "local")-> Tuple[List[Document], List[str]]:
        """Parse and chunk a single file"""
        global CONVERTER
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = path_leaf(file_path)

        # Try parsing
        text = None

        # Parsing using LlamaParse
        if parsing_method == "llama_index":
            try:
                parsed_docs = parse_pdf_llama(file_path)
                text = ''.join(doc.text for doc in parsed_docs)
            except Exception as e:
                print(f"[LlamaParse] failed for {file_path}: {e}")

        # Parsing locally
        if not text:
            try:
                if file_extension == ".pdf":
                    if CONVERTER is None:
                        print("[INFO] Converter was not already initialized. Loading marker-pdf...")
                        CONVERTER = create_converter()
                    text = parse_pdf(converter=CONVERTER, filename=file_path)
                else:
                    text = parse_document(file_path)
            except Exception as e:
                print(f"[Local Parsing] failed for {file_path}: {e}")
                return ([], [])

        chunks = chunk_text(text, token_fn, chunk_size=1024, chunk_overlap=256)

        # Validate and tag chunks
        valid_chunks : List[Document] = []
        for chunk in chunks:
            if is_valid_chunk(chunk.page_content): 
                try:
                    lang = detect(chunk.page_content)
                except Exception as e:
                    print(f"[Language Detection] failed for chunk: {chunk} in file {file_path}: {e}")
                    lang = "english" # Fall back to english for now
                chunk.metadata.update({
                    'file_path' : file_path,
                    'file_name' : file_name,
                    'language' : lang,
                })
                if lang not in languages and lang != "unkown":
                    languages.append(lang)
                valid_chunks.append(chunk)
    
        return valid_chunks, languages


def parse_url(urls: List[str], token_fn: Callable[[str], int], parsing_method: Literal["local", "llama-index"] = "local") -> Tuple[List[Document], List[str]]:
    """
    Parses a list of URLs in to markdown format. Uses marker-pdf or LlamaParse for URLs to direct files. 

    Args:
        urls (List[str]):
            A list of URLs to parse.
        token_fn (Callable[[str], int]):
            A tokenizer function to tokenize the text for chunking. 
        parsing_method (Literal):
            Parsing method to use in case the URL is not a webpage but instead file, e.g. PDF. 
            Default: "local"

    Returns:
        tuple (List[Document], List[str]):
            A list of langchain documents from the parsed URLs.
            A list of languages detected from the parsed URLs
    """

    documents : List[Document]  = []
    languages : List[str]       = []

    # Split url into web/pdf
    html_urls : List[str] = []
    file_urls : List[str] = []

    for url in urls:
        if url.lower().endswith(SUPPORTED_FORMATS):
            file_urls.append(url)
        else:
            html_urls.append(url)

    # Parse Webpage URLs
    for url in html_urls: 
        loader = AsyncHtmlLoader([url])
        docs = loader.load()
        md = MarkdownifyTransformer(strip=["a"])
        converted_docs = md.transform_documents(docs)
        # Chunk the docs
        for doc in converted_docs: 
            chunks = chunk_text(doc.page_content, token_fn=token_fn, chunk_size=1024, chunk_overlap=256)
            # Add metadata
            for chunk in chunks:
                # Filter out non-useful chunks, i.e. too short or no text content
                if is_valid_chunk(chunk.page_content):
                    try:
                        lang = detect(chunk.page_content)
                    except Exception as e:
                        print(f"[Language Detection] failed on chunk {chunk}: {e}")
                        lang = "english" # Fall back to english for now
                    # Generate unique hash for document
                    content_hash = hash_document(chunk)
                    chunk.metadata.update({
                        'file_name' : url,
                        'file_path' : url,
                        'language'  : lang,
                        'content_hash' : content_hash,
                    })
                    if lang not in languages and lang != "unkown":
                        languages.append(lang)
                    documents.append(chunk)
    
    # Parse File URLs
    for fp in file_urls:
        try:
            file_extension = os.path.splitext(fp)[1].lower()
            response = requests.get(fp)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            parsed_chunks, langs = _parse_single_file(tmp_path, languages=languages, token_fn=token_fn, parsing_method=parsing_method)
            languages.extend(langs)
            for chunk in parsed_chunks:
                chunk.metadata.update({
                    'file_path' : fp,
                })
                documents.append(chunk)

        except Exception as e:
            print(f"[URL (File) Parsing] failed for {fp}: {e}")
            continue
    
    return documents, list(set(languages))

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
        model_name (str) : 
            The name of the model to use for tokenization/conversation
        files (List[str]) : 
            A list of file paths
        urls (List[str]) : 
            An optional list of urls to parse 
        parsing_method (str) : 
            A string specifying the method to use for parsing the PDFs, either "local" or "llama_index". Defaults to "local". 
        
    Returns:
        tuple (List[Document], List[str]):
            - A list of chunks from the parsed files in Langchain Documents format. 
            - A list of languages detected from the provided files. 
    """
    global CONVERTER

    token_fn : Callable[[str], int] = token_len_fn(model_name)
    documents : List[Document]  = []
    languages : List[str]       = []

    # Check if we need to load marker-pdf
    if parsing_method == "local" and any(fp.lower().endswith(".pdf") for fp in files):
        CONVERTER = create_converter()
    
    # Parse files
    for fp in files:
        print(f"Parsing file: {fp}\n")
        f_docs, file_langs = _parse_single_file(fp, languages=languages, token_fn=token_fn, parsing_method=parsing_method)
        documents.extend(f_docs)
        languages = file_langs
    
    # Parse urls
    if urls: 
        docs, url_langs = parse_url(urls, token_fn=token_fn, parsing_method=parsing_method)
        documents.extend(docs)
        languages.extend(url_langs)
    
    # Add unique has for each document as metadata (Used for preventing duplicates)
    for doc in documents:
        doc.metadata['content_hash'] = hash_document(doc=doc)
    
    # Erasing duplicate languages if any
    languages = list(set(languages))
    
    # Saving detected languages
    config = load_json(CONFIG_FILE)
    config["languages"] = languages
    save_json(CONFIG_FILE, config)

    return documents, languages

if __name__ == "__main__":
    parse_pipeline(files=["data/ComIt_MA_2022.pdf"], 
                   model_name="gpt-4o",
                   parsing_method="local")
