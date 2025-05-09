from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from local_embeddings import LocalEmbeddings
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from vectorstore import CustomMultiVectorStore
import json
import operator
from typing import List, Annotated, Dict, Any
from model_configuration import get_settings_from_config
import re
from langchain_core.documents import Document
from lingua import Language, LanguageDetectorBuilder

# Configuration
settings = get_settings_from_config()
llm_model_name = settings["llm_model_name"]
embeddings_model_name = settings["embeddings_model_name"]
languages = [Language.ENGLISH, 
             Language.GERMAN, 
             Language.SWEDISH, 
             Language.DANISH, 
             Language.NYNORSK,
             Language.GREEK,
             ]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

if "gpt" in llm_model_name:
    llm = ChatOpenAI(
        model=llm_model_name,
        temperature=0.1,
        max_tokens=1000,
    )
    llm_json = ChatOpenAI(
        model=llm_model_name,
        temperature=0.1,
        max_tokens=1000,
        model_kwargs={"response_format" : {"type" : "json_object"}})
    
else:
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0.1,
        max_tokens=1000,
    )
    llm_json = ChatOllama(
        model=llm_model_name,
        temperature=0.1,
        max_tokens=1000,
        format="json"
    )

# Load Embeddings Model
if "text-embedding-3" in embeddings_model_name:
    embeddings_model = OpenAIEmbeddings(model=embeddings_model_name)
else:
    embeddings_model = LocalEmbeddings(embeddings_model_name)


# Helper Functions

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def clean_response(response: Dict[str, Any]) -> Dict[str, Any]:
    match = re.search(r'```json(.*?)```', response.content, re.DOTALL)
    if match:
        cleaned_response = match.group(1).strip()
        response.content = cleaned_response
    return response


def extract_sources(documents: List[Document]) -> List[str]:
    seen_paths = set()
    sources = []

    for doc in documents:
        f_name = doc.metadata.get("file_name", "Unkown Source")
        f_path = doc.metadata.get("file_path", "")

        if f_path not in seen_paths:
            seen_paths.add(f_path)
        
            if f_path.startswith("http"):
                sources.append(f"[{f_name}]({f_path})")
            else:
                sources.append(f"`{f_name}`")
    
    return sources


def format_sources(sources: List[str]) -> str:
    if not sources:
        return ""
    
    sources_formatted = "\n\n---\n**📚 Sources: **\n" + "\n".join(f"- {s}" for s in sources)
    return sources_formatted

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
You are an assistant specializing in question-answering based on provided documents. Your answers should be well-structured, informative, and formatted using Markdown.

### Context:
{context}

### User Question:
**{question}**

#### Your Answer:
- Provide a clear and complete answer based **only** on the context above.
- Format your response in **Markdown** (use bold for key points, bullet points for lists, and code blocks where appropriate).
- Do not be overly brief; ensure completeness while remaining concise.
- Avoid repetition and unnecessary padding.

**Final Answer:** 
"""

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

Return JSON with two keys, 'binary_score' is 'yes' or 'no' to indicate whether the STUDENT ANSWER is grounded in the FACTS and helps answer the QUESTION. And a key, explanation, that contains an explanation of the score. 
"""


### Graph ###

class GraphState(MessagesState):
    """
    Graph state is a dictionary that contains information we want to propogate to, and modify in, each graph node. 
    """

    direct_response : str # Binary decision whether to retrieve documents or answer using already given context. 
    no_relevant_docs: int # Binary state if we no relevant documents for the questions were found. 
    max_retries : int # Max number of retries for generating an answer
    answers : int # Number of answers generated
    loop_step : Annotated[int, operator.add]
    documents : List[str] # List of retrieved documents 
    question : str # User query translated to question
    translations : Dict[str, Dict[str, str]]   # { lang: {"query": str, "question": str}, ... }
    doc_sets : Dict[str, List[Document]]       # { lang: [Document, Document, ...], ... }

# Nodes
def generate(state : GraphState):
    """
    Generates answer using RAG on retrieved documents. 
    
    Args: 
        sate (GraphState) : The current graph state.

    Returns: 
        sate (GraphState) : New key added to state, answer, that contains LLM generated answer.
    """

    print("----GENERATE----")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG Generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    # Append sources to generation
    sources = extract_sources(documents)
    generation.content += format_sources(sources)
    
    print(generation)
    return {"messages" : [generation], "loop_step" : loop_step+1}

def grade_documents(state : GraphState):
    """
    Determines whether the retrieved documents are relevant to the question. 
    If no document is relevant we inform the user that we couldn't find any relevant documents for the question. 

    Args: 
        sate (GraphState) : The current graph state

    Returns: 
        sate (GraphState) : Filtered out irrelevant documents and updated no_relevant_docs state. 
    """

    print("----CHECK DOCUMENT RELEVANCE FOR EACH LANGUAGE----")

    translations = state.get("translations", {})
    doc_sets = state.get("doc_sets", {})
    print(doc_sets)

    final_docs = []
    # For each language, we do doc grading with the language-specific question
    for lang, docs in doc_sets.items():
        question_for_lang = translations[lang]["question"]
        filtered_docs = []

        for doc in docs or []:
            doc_grader_prompt_formatted = doc_grader_prompt.format(
                document=doc.page_content,
                question=question_for_lang
            )
            result = llm_json.invoke([
                SystemMessage(content=doc_grader_instructions),
                HumanMessage(content=doc_grader_prompt_formatted)
            ])
            grade = json.loads(result.content).get("binary_score", "no").lower()
            if grade == "yes":
                filtered_docs.append(doc)

        final_docs.extend(filtered_docs)

    # If no doc is relevant, set no_relevant_docs
    if not final_docs:
        no_relevant_docs = 1
    else:
        no_relevant_docs = 0

    return {
        "documents": final_docs,
        "no_relevant_docs": no_relevant_docs
    }

def no_relevant_documents(state : GraphState):
    """
    Informs the user that no relevant documents were found in the vectorstore. 

    Args:
        sate (GraphState) : The current graph state

    Returns:
        sate (GraphState) : New graph state, generation, telling the user that no documents we're relevant to the query in the vectorstore. 
    """

    print("----DECISION : NO RELEVANT DOCUMENTS WERE FOUND----")
    no_relevant_docs_prompt = """ You performed a search in the vectorstore to retrieve relevant documents to the user query.
    But no relevant documents were found. Please inform the user that you cannot answer this question, since there was no information available.
    Keep your answer brief and concise. Use a maximum of two sentences. 
    """
    response = llm.invoke(no_relevant_docs_prompt)
    return {'messages' : [response]}

def direct_response(state : GraphState):
    """
    Generates a LLM answer for the qiven question using the knowledge from the current conversation. 

    Args: 
        sate (GraphState) : The current graph state

    Returns: 
        sate (GraphState) : New graph state, answer, giving the LLM generated response to the question as well as resetting direct_response.
    """

    print("----DIRECT RESPONSE----")
    # Prompt
    convo_prompt = """
    You are a helpful assistant built to have natural conversations with users. You should provide clear, complete, and well-structured responses formatted in **Markdown**.

    ### Conversation History:
    {context}

    ### User Query:
    **{question}**

    #### Your Response:
    - Think carefully about the context before responding.
    - Provide a full, well-structured response while remaining concise.
    - You are allowed to use emojis when appropriate and needed.

    **Final Answer:** 
    """

    conversation_history = "\n".join([
        f"{message.type.capitalize()}: {message.content}"
        for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ])

    convo_prompt_formatted = convo_prompt.format(context=conversation_history, question=state["messages"][-1].content)
    print(convo_prompt_formatted)
    response = llm.invoke(convo_prompt_formatted)
    print(response)
    return {"messages" : [response]}

def final_response(state: GraphState):
    """
    Returns the final answers of the RAG, translated into the language of the user query if needed. 

    Args:
        state (GraphState) : The current graph state containing the answer from RAG.
    Returns:
        sate (GraphState) : New graph state, answer, containing the final answer from RAG.
    """

    print("----FINAL RESPONSE----")
    final_answer = state["messages"][-1]
    answer_langs = detector.compute_language_confidence_values(final_answer.content.lower())
    target_langs = detector.compute_language_confidence_values(state["messages"][-2].content.lower())
    answer_lang, target_lang = answer_langs[0].language.name, target_langs[0].language.name
    print(answer_langs, target_langs, answer_lang, target_lang, sep='\n')
    
    if answer_lang!=target_lang:
        # Translate the answer into the language of the user query
        translation_prompt = """
        You are an expert at translating the given text into the desired language. 

        ### Input Text:
        **{input_text}**

        ### Target Language: **{target_language}**

        Keep you translation accurate and precise, while maintining all the context and information of the input text. Output nothing else other than the translation.  

        #### Your Translation:
        """
        translation_prompt_formatted = translation_prompt.format(
            input_text=final_answer.content,
            target_language=target_lang
        )
        translation_result = llm.invoke(translation_prompt_formatted)
        final_answer.content = translation_result.content
    
    return {"messages" : [final_answer]}

    
# Edges 
def route_question(state: MessagesState):
    """
    Route question to direct response or RAG based on the conversation history.

    Args: 
        sate (GraphState) : The current graph state containing conversation history.

    Returns:
        str : Next node to call, either "direct_response" or "vectorstore".
    """

    print("----ROUTE QUESTION----")

    # Get full conversation history
    conversation_history = "\n".join([
        f"{message.type.capitalize()}: {message.content}"
        for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ])

    ### ROUTER ###
    router_prompt = """  
    You are an expert in determining whether a user query requires retrieval from a vectorstore or a direct response.

    The vectorstore contains user-uploaded documents, which may cover any topic. If the user's query requires information that could be found in these documents, route it to the vectorstore.

    For queries that are purely conversational, such as greetings ("Hi", "Hello") or casual questions ("How are you?"), and do not require knowledge from the documents, use direct-response. This applies even if the query is in a language other than English (e.g., Danish, French). 

    Questions asking for **definitions** should generally be retrieved from the vectorstore, unless the term is universally known and unambiguous (e.g., "What is an apple?" or "Define water"). If there is **any uncertainty** about whether the term might have a specific meaning within the uploaded documents, route the query to the vectorstore.
    
    The conversation history is:
    -------------------------
    {conversation_history}
    -------------------------
    The last response query by the Human is the user's query, in the conversation history above. 
    Now, determine whether to route the user's query to the vectorstore or a direct response.

    Return a JSON object with a single key, `datasource`, set to either `"direct_response"` or `"vectorstore"` based on the user's query.
    """

    # Formatting prompt
    formatted_prompt = router_prompt.format(conversation_history=conversation_history)
    print(formatted_prompt)

    # Get routing decision
    route_decision = llm_json.invoke(formatted_prompt)
    source = json.loads(route_decision.content)['datasource']

    if source == "direct_response":
        print("----ROUTING TO DIRECT_RESPONSE----")
        return "direct_response"
    elif source == "vectorstore":
        print("----ROUTING TO VECTORSTORE----")
        return "vectorstore"

def decide_to_generate(state : GraphState):
    """
    Decides whether to generate an answer or respond no relevant documents found

    Args:
        sate (GraphState) : The current graph state

    Returns:
        str : Binary decision for next node to call, either 'no_relevant_documents' or 'generate'
    """

    print("----ASSESS GRADED DOCUMENTS----")
    no_relevant_docs = state["no_relevant_docs"]
    if no_relevant_docs:
        print("----NO RELEVANT DOCS WERE FOUND : INFORMING USER----")
        return "no_relevant_documents"
    else:
        print("----DECISION : GENERATE----")
        return "generate"

def grade_generation_v_documents_and_question(state: MessagesState):
    """
    Determines whether the generated answer is grounded in the documents and answers the question.

    Args: 
        sate (GraphState) : The current state graph

    Returns: 
        str : Decision for next node to call
    """

    print("---CHECK ANSWER----")
    question = state["question"]
    documents = state["documents"]
    generation = None
    for message in reversed(state["messages"]):
        if message.type == "system" or (message.type == "ai" and not message.tool_calls):
            generation = message.content
            break
    max_retries = state.get("max_retries", 3) # Defaults to 3 if not provided

    # Formatting
    answer_grader_prompt_formatted = answer_grader_prompt.format(question=question, documents=format_docs(documents), generation=generation)
    #print(answer_grader_prompt_formatted)
    result = llm_json.invoke([SystemMessage(content=answer_grader_instructions)] + [HumanMessage(content=answer_grader_prompt_formatted)])
    grade = json.loads(result.content)["binary_score"]

    # Check answer
    if grade == "yes":
        # Meaning good answer
        print("----GENERATED ANSWER IS GROUNDED IN DOCUMENTS AND ANSWERS THE QUESTION----")
        return "useful"
    elif state["loop_step"] <= max_retries: 
        print("----GENERATED ANSWER IS NOT GROUNDED IN DOCUMENTS OR DOES NOT ANSWER THE QUESTION----")
        # Remove the answer from state["messages"]
        state["messages"] = state["messages"][:-1]
        return "not useful"
    else:
        print("----MAX RETRIES REACHED : STOPPING RAG----")
        return "max retries"


def initialize_graph(multi_vector_store: CustomMultiVectorStore, memory):

    vectorstore = multi_vector_store

    def retrieve(state : GraphState):
        """
        Retrieve documents from vectorstore.

        Args: 
            sate (GraphState) : The current graph state.

        Returns: 
            sate (GraphState) : New key added to state, documents, that contains retrieved documents. 
        """

        print("----RETRIEVAL----")
        retriever_prompt = """You are an expert at crafting queries from conversational history in order to retrieve relevant documents from a vectorstore. 

        Here is the conversation so far:

        -----------------------
        {conversation_history}
        -----------------------

        Based on the user's context, generate a concise and relevant query to retrieve the most appropriate documents from the vectorstore. The query should be based on the user's most recent message and the overall context of the conversation, avoiding irrelevant details.

        1) Produce an "ephemeral_question_en" which is an English question summarizing that final user query.
        2) For each language in this list: {languages},
        create:
        - "query" : a refined short query in that language formulated from the conversation, which is suitable for retrieving documents from the vectorstore.
        - "question": a well-formed question in that language

        Return JSON like:
        {{
        "ephemeral_question_en": "...",
        "translations": {{
            "en": {{"query": "...", "question": "..."}},
            "de": {{"query": "...", "question": "..."}},
            ...
            }}
        }}
        Be concise. 
        """

        conversation_history = "\n".join([
            f"{message.type.capitalize()}: {message.content}"
            for message in state["messages"]
            if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
        ])

        retriever_prompt_formatted = retriever_prompt.format(conversation_history=conversation_history, languages=vectorstore.languages)
        #print(retriever_prompt_formatted)

        response = llm_json.invoke(retriever_prompt_formatted)
        print(response)
        try:
            result_dict = json.loads(response.content)
        except:
            # fallback if parsing fails
            result_dict = {
                "ephemeral_question_en": state["messages"][-1].content,
                "translations": {}
            }

        ephemeral_en = result_dict.get("ephemeral_question_en", "")
        translations = result_dict.get("translations", {})

        # Now retrieve from each language in the multi-vectorstore
        doc_sets = {}
        for lang in vectorstore.languages:
            if lang in translations:
                refined_query = translations[lang]["query"]
                docs = vectorstore.query_vectorstore(refined_query, lang, k=5)
                doc_sets[lang] = docs if docs else []
            else:
                doc_sets[lang] = []

        return {
            "question": ephemeral_en,     # We store the final question in English
            "translations": translations, # language -> {query, question}
            "doc_sets": doc_sets
        }

    # Building Graph
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("respond_directly", direct_response)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("no_relevant_documents", no_relevant_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("final_response", final_response)
    workflow.add_edge("respond_directly", "final_response")
    workflow.add_edge("no_relevant_documents", "final_response")
    workflow.add_edge("final_response", END)

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
            "useful" : "final_response",
            "max retries" : "final_response",
        },
    )

    # Memory 
    config = {"configurable" : {"thread_id" : "local-rag"}}

    # Compile 
    graph = workflow.compile(checkpointer=memory)

    return graph, config
