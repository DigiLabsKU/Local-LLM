from langchain_ollama import ChatOllama
from custom_retriever import CustomRetriever
from local_embeddings import LocalEmbeddings
from langgraph.graph import MessagesState, StateGraph, END
from langchain.schema import Document
from langchain_core.tools import tool
from vectorstore import vectorstore_pipeline, load_existing_vectorstore
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
import json
import operator
from typing_extensions import TypedDict
from typing import List, Annotated

from PIL import Image


# Initialize components as before
llm_model_name = "llama3.2"

llm = ChatOllama(
    model=llm_model_name,
    temperature=0.1,
    max_tokens=1000,
    num_gpu=1,
)

llm_json = ChatOllama(
    model=llm_model_name,
    temperature=0.1,
    max_tokens=1000,
    num_gpu=1,
    format="json",
)

embeddings_model_name = "all-MiniLM-L6-v2"
embeddings_model = LocalEmbeddings(embeddings_model_name)

# Create a new vectorstore instance
# vectorstore = vectorstore_pipeline(embeddings_model_name=embeddings_model_name,
#                                   llm_model="meta-llama/Llama-3.2-3B",
#                                   file_paths=["data/ComIt_MA_2022.pdf"],
#                                   store_path="faiss_index",
#                                   enrich_method="keywords")
# Load existing vectorstore

vectorstore = load_existing_vectorstore(embeddings_model_name, "faiss_index", use_gpu=False)

metadata_info = {
    "page_id": "The page id of the document or chunk.",
    "source": "The source file (i.e., path) of the document.",
    "keywords": "Topics extracted from the query, listed as a list of words."
}

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.1})

### Router ###

#Prompt
router_instructions = """  
You are an expert in determining whether a user query requires retrieval from a vectorstore or a direct response.  

The vectorstore contains user-uploaded documents, which may cover any topic. If the user's query requires information that could be found in these documents, route it to the vectorstore.  

For queries that are purely conversational, such as greetings ("Hi", "Hello") or casual questions ("How are you?"), and do not require knowledge from the documents, use direct-response.  

Questions asking for **definitions** should generally be retrieved from the vectorstore, unless the term is universally known and unambiguous (e.g., "What is an apple?" or "Define water"). If there is **any uncertainty** about whether the term might have a specific meaning within the uploaded documents, route the query to the vectorstore.  

Return a JSON object with a single key, `datasource`, set to either `"direct_response"` or `"vectorstore"` based on the user's query.  
"""


### Retrieval Grader ###

# Instructions
doc_grader_instructions = """ You are a grader assessing the relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
"""

# Grader Prompt
doc_grader_prompt = """
Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

Carefully and objectively assess whether the document contains at least some information that is relevant to the question. 

Return JSON with single key, `binary_score`, that is "yes" or "no" score to indicate whether the document contains at least some information relevant to the question.
"""

### RAG ###

# Prompt
rag_prompt = """
You are an assistant for question-answering tasks. You are built to have conversations with users regarding their uploaded documents. 

Here is the context to use to answer the question:

{context}

Think carefully about the above context.

Now, review the user question: 

{question}

Provide an answer to this question using only the context above. 

Keep the answer brief and concise. Don't repeat yourself.

Answer:""" 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

### Answer Grader ###

# Answer grader instructions
answer_grader_instructions = """

You are a teacher grading a quiz. 

You will be given a QUESTION, FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow: 

(1) Ensure the STUDENT ANSWER helps to answer the QUESTION. 

(2) Ensure the STUDENT ANSWER is grounded in the FACTS. 

Score: 

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give. 

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion is correct. 

Avoid simply stating the correct answer at the outset. 
"""

# Grader Prompt
answer_grader_prompt = """QUESTION: {question} \n\n FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}

Return JSON with two keys, 'binary_score' is 'yes' or 'no' to indicate whether the STUDENT ANSWER is grounded in the FACTS and helps answer the QUESTION. And a key, explanation, that contains an explanation of the score.. 
"""

### Graph ###

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propogate to, and modify in, each graph node. 
    """

    question: str # User question
    generation: str # LLM Generation
    direct_response : str # Binary decision whether to retrieve documents or answer using already given context. 
    no_relevant_docs: int # Binary state if we no relevant documents for the questions were found. 
    max_retries : int # Max number of retries for generating an answer
    answers : int # Number of answers generated
    loop_step : Annotated[int, operator.add]
    documents : List[str] # List of retrieved documents 

# Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore.

    Args: 
        state (dict) : The current graph state.

    Returns: 
        state (dict) : New key added to state, documents, that contains retrieved documents. 
    """

    print("----RETRIEVAL----")
    question = state["question"]

    docs = retriever.invoke(question)
    return {"documents" : docs}

def generate(state):
    """
    Generates answer using RAG on retrieved documents. 
    
    Args: 
        state (dict) : The current graph state.

    Returns: 
        state (dict) : New key added to state, answer, that contains LLM generated answer.
    """

    print("----GENERATE----")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG Generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation" : generation, "loop_step" : loop_step+1}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question. 
    If no document is relevant we inform the user that we couldn't find any relevant documents for the question. 

    Args: 
        state (dict) : The current graph state

    Returns: 
        state (dict) : Filtered out irrelevant documents and updated no_relevant_docs state. 
    """

    print("----CHECK DOCUMENT RELEVANCE TO THE QUESTION----")
    question = state["question"]
    documents = state["documents"]

    # Check each doc
    filtered_docs = []
    no_relevant_docs = 0
    for doc in documents: 
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=doc.page_content, question=question)
        result = llm_json.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("----GRADE: DOCUMENT RELEVANT----")
            filtered_docs.append(doc)
        # Document not relevant
        else:
            continue

    # Check if no relevant documents were found
    if len(filtered_docs) == 0: 
        no_relevant_docs = 1

    return {"documents" : filtered_docs, "no_relevant_docs" : no_relevant_docs}

def no_relevant_documents(state):
    """
    Informs the user that no relevant documents were found in the vectorstore. 

    Args:
        state (dict) : The current graph state

    Returns:
        state (dict) : New graph state, generation, telling the user that no documents we're relevant to the query in the vectorstore. 
    """

    print("----DECISION : NO RELEVANT DOCUMENTS WERE FOUND----")
    generation = {'content': "Sorry, could not find any information regarding your question in the provided document(s)"}
    return {'generation' : generation}

def direct_response(state):
    """
    Generates a LLM answer for the qiven question using the knowledge from the current conversation. 

    Args: 
        state (dict) : The current graph state

    Returns: 
        state (dict) : New graph state, answer, giving the LLM generated response to the question as well as resetting direct_response.
    """

    # Add memory later -> So we can either use the previous conversation knowledge to respond or respond directly to a purely conversational question such as "Hi", "How are you?" etc. 
    print("----DIRECT RESPONSE----")
    question = state["question"]
    rag_prompt_formatted = rag_prompt.format(context= "No Context : Respond to the user approriate to the question", question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    result = generation.content if generation.content != "" else "No response could be generated"
    return {"generation" : generation}
    
    
# Edges 

def route_question(state):
    """
    Route question to direct response or RAG

    Args: 
        state (dict) : The current graph state
        
    Returns:
        str : Next node to call, either direct_response or retrieve
    """

    print("----ROUTE QUESTION----")
    route_question = llm_json.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content=state["question"])])
    source = json.loads(route_question.content)['datasource']
    if source == "direct_response":
        print("----ROUTING TO DIRECT_RESPONSE----")
        return "direct_response"
    elif source == "vectorstore":
        print("----ROUTING TO VECTORSTORE----")
        return "vectorstore"

def decide_to_generate(state):
    """
    Decides whether to generate an answer or respond no relevant documents found

    Args:
        state (dict) : The current graph state

    Returns:
        str : Binary decision for next node to call, either 'no_relevant_documents' or 'generate'
    """

    print("----ASSESS GRADED DOCUMENTS----")
    question = state["question"]
    no_relevant_docs = state["no_relevant_docs"]
    if no_relevant_docs:
        print("----NO RELEVANT DOCS WERE FOUND : INFORMING USER----")
        return "no_relevant_documents"
    else:
        print("----DECISION : GENERATE----")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generated answer is grounded in the documents and answers the question.

    Args: 
        state (dict) : The current state graph

    Returns: 
        str : Decision for next node to call
    """

    print("---CHECK ANSWER----")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3) # Defaults to 3 if not provided

    # Formatting
    answer_grader_prompt_formatted = answer_grader_prompt.format(question=question, documents=format_docs(documents), generation=generation.content)
    result = llm_json.invoke([SystemMessage(content=answer_grader_instructions)] + [HumanMessage(content=answer_grader_prompt_formatted)])
    grade = json.loads(result.content)["binary_score"]

    # Check answer
    if grade == "yes":
        # Meaning good answer
        print("----GENERATED ANSWER IS GROUNDED IN DOCUMENTS AND ANSWERS THE QUESTION----")
        return "useful"
    elif state["loop_step"] <= max_retries: 
        print("----GENERATED ANSWER IS NOT GROUNDED IN DOCUMENTS OR DOES NOT ANSWER THE QUESTION----")
        return "not useful"
    else:
        print("----MAX RETRIES REACHED : STOPPING RAG----")
        return "max retries"

# Building Graph
workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("respond_directly", direct_response)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("no_relevant_documents", no_relevant_documents)
workflow.add_node("generate", generate)
workflow.add_edge("respond_directly", END)
workflow.add_edge("no_relevant_documents", END)

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "direct_response" : "respond_directly",
        "vectorstore" : "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "no_relevant_documents" : "no_relevant_documents",
        "generate" : "generate",
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not useful" : "generate",
        "useful" : END,
        "max retries" : END,
    },
)
# Compile 
graph = workflow.compile()

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())



while True:
    query = input("Query: ")
    if query.lower() == "exit":
        break;
    inputs = {"question" : query, "max_retries" : 3}
    # for event in graph.stream(inputs, stream_mode="values"):
    #     print(event)
    results = graph.invoke(inputs)

    print(results['generation'].content)