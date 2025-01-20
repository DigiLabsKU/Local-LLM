from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from keybert import KeyBERT
from langchain_core.documents import Document
from summarizer import Summarizer
import gc
from torch.cuda import empty_cache, is_available

def free_resources_doc_parser():
    global kw_model, bert_model

    del kw_model, bert_model

    if is_available():
        empty_cache()

    gc.collect()

    print("Freed resources for Document Parser: Marker-Pdf, BERT Summarizer and KeyBERT")

model_name = "gpt-4" #"meta-llama/Llama-2-7b-hf"

def create_converter():
    
    converter = PdfConverter(
    artifact_dict=create_model_dict(),
    )

    return converter

def clean_metadata(metadata) -> list:
    cleaned_data = []
    toc = metadata.get('table_of_contents', [])
    for i, item in enumerate(toc):
        title = item.get('title')
        page_id = item.get('page_id')
        parent = None
        if i > 0 and item.get('level', 0) > toc[i-1].get('level', 0):
            parent = toc[i-1].get('title')
        if title and title.strip():
            cleaned_data.append({
                'title': title.strip(),
                'page_id': page_id,
                'parent': parent,
                'level': item.get('level', 0)
            })
    return cleaned_data

def parse_pdf(converter: PdfConverter, filename: str, **kwargs) -> tuple[str, list, dict]:
    
    rendered = converter(filename)
    text, _, images = text_from_rendered(rendered)
    raw_metadata = rendered.metadata

    cleaned_metadata = clean_metadata(raw_metadata)
    cleaned_metadata.append({'source': filename})

    return text, cleaned_metadata, images


def token_len_fn(model_name: str):
    tokenizer = tiktoken.get_encoding('cl100k_base') if "gpt" in model_name else AutoTokenizer.from_pretrained(model_name)
    if "gpt" in model_name:
        def tiktoken_len(text):
            tokens = tokenizer.encode(
                text,
                disallowed_special=()
            )
            return len(tokens)
        return tiktoken_len
    else: # A model from huggingface
        def token_len(text):
            tokens = tokenizer.encode(
                text,
                add_special_tokens=False
            )
            return len(tokens)
        return token_len

token_fn = token_len_fn(model_name)

def chunk_text(text: str, token_fn, chunk_size=1024, chunk_overlap=256):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_fn,
        separators= ['\n \n', '\n', ' ', ''],
    )
    splits = splitter.split_text(text)
    return splitter.create_documents(splits)


# Keywords
kw_model = KeyBERT()
def extract_keywords(text, n_keywords=5):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return [kw[0] for kw in keywords[:5]]

# Summarization using BERT
bert_model = Summarizer()
def summarize_text(text: str, min_len=20, max_len=200) -> str:
    return ''.join(bert_model(text, min_length=min_len, max_length=max_len))


def enrich_chunks(documents, metadata, enrich_method=None):
    for i, doc in enumerate(documents):
        closest_section = next(
            (item for item in reversed(metadata) if 'page_id' in item and item['page_id'] <= i),
            {'title': 'Unknown Section'}
        )
        doc.metadata['title'] = closest_section['title']
        doc.metadata['page_id'] = closest_section['page_id'] if 'page_id' in closest_section else 'no page_id available'
        doc.metadata['chunk_index'] = i
        doc.metadata['source'] = metadata[-1]['source']
        if enrich_method == "keywords":
            doc.metadata['keywords'] = extract_keywords(doc.page_content)
        elif enrich_method == "summarization":
            summary = summarize_text(doc.page_content, min_len=20, max_len=200)
            doc.metadata['summary'] = summary
    return documents

def parse_pipeline(files: list[str], model_name:str, enrich_method: str=None) -> list[Document]:
    """
    Parses and chunks a list of PDF files using the provided converter and enrichment method.
    
    Parameters
    -----------
    files : list[str]
        A list of file paths
    model_name : str
        The name of the model to use for tokenization/conversation
    enrich_method : str
        An optional string telling which method to use for enriching the chunks, i.e. "summarization" or "keywords". The latter is more cost-effective. 
    
    Returns
    -------
    list
        A list of chunks with additional metadata as langchain Documents. 

    """

    documents = []
    converter = create_converter()
    token_fn = token_len_fn(model_name)
    for fname in files:
        text, cleaned_metadata, _ = parse_pdf(converter, fname)
        chunked_text = chunk_text(text, token_fn, chunk_size=1024, chunk_overlap=256)
        enriched_chunks = enrich_chunks(chunked_text, cleaned_metadata, enrich_method=enrich_method)
        documents.extend(enriched_chunks)
    
    return documents

# # Test

# files = ['data/ComIt_MA_2022.pdf']
# documents = parse_pipeline(files, enrich_method="keywords")
# print(len(documents))
# print(documents[1].metadata['keywords'])

# input_ = None
# while input_!="quit":
#     input_ = input("Enter something: ")
#     if input_=="free":
#         free_resources_doc_parser()
