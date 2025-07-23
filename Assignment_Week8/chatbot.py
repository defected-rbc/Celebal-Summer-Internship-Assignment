import os
import streamlit as st
from typing import List, TypedDict, Tuple
import asyncio
import platform

# --- Fix for RuntimeError: There is no current event loop in thread on Windows ---
# This must be set very early to ensure it applies to all threads created by Streamlit
# that might interact with asyncio-dependent libraries like grpcio.
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure you have these libraries installed:
# pip install langchain langchain-google-genai chromadb langgraph streamlit

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, END

# --- 1. Define the RAG Document (Your provided dataset information) ---
BASE_RAG_DOCUMENT_CONTENT = """
## About Company
Dream Housing Finance company deals in all home loans. They have a presence across all urban, semi-urban, and rural areas. Customer-first applies for a home loan after that company validates the customer eligibility for a loan.

## Problem Statement:
The company wants to automate the loan eligibility process (real-time) based on customer detail provided while filling the online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others. To automate this process, they have given a problem to identify the customer segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

## Features of Training Dataset:
1.  Loan_ID: Unique Loan ID
2.  Gender: Male/Female
3.  Married: Applicant married (Y/N)
4.  Dependents: Number of dependents
5.  Education: Applicant Education (Graduate/Under Graduate)
6.  Self_Employed: Self-employed (Y/N)
7.  ApplicantIncome: Applicant income
8.  CoapplicantIncome: Coapplicant income
9.  LoanAmount: Loan amount in thousands
10. Loan_Amount_Term: Term of the loan in months
11. Credit_History: Credit history meets guidelines
12. Property_Area: Urban/Semi-Urban/Rural
13. Loan_Status: (Target) Loan approved (Y/N)

## Features of Testing Dataset:
1.  Loan_ID: Unique Loan ID
2.  Gender: Male/Female
3.  Married: Applicant married (Y/N)
4.  Dependents: Number of dependents
5.  Education: Applicant Education (Graduate/Under Graduate)
6.  Self_Employed: Self-employed (Y/N)
7.  ApplicantIncome: Applicant income
8.  CoapplicantIncome: Coapplicant income
9.  LoanAmount: Loan amount in thousands
10. Loan_Amount_Term: Term of the loan in months
11. Credit_History: Credit history meets guidelines
12. Property_Area: Urban/Semi-Urban/Rural

## Features of Sample_submission Dataset:
1.  Loan_ID: Unique Loan ID
2.  Loan_Status: (Target) Loan approved (Y/N)

## Evaluation Metric:
Your model performance will be evaluated on the basis of your prediction of loan status for the test data (test.csv), which contains similar data-points as train except for the loan status to be predicted. Your submission needs to be in the format as shown in the sample submission. We at our end, have the actual loan status for the test dataset, against which your predictions will be evaluated. We will use the Accuracy value to judge your response.
"""

# --- Load Training Dataset from CSV ---
TRAINING_DATA_FILE = "Training Dataset.csv"
training_data_content = ""
try:
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, TRAINING_DATA_FILE)

    with open(file_path, 'r', encoding='utf-8') as f:
        training_data_content = f.read()
    # Streamlit message for successful load
    # st.success(f"Successfully loaded {TRAINING_DATA_FILE}") # Moved below set_page_config
except FileNotFoundError:
    # Streamlit warning for file not found
    # st.warning(f"Warning: '{TRAINING_DATA_FILE}' not found in the same directory.") # Moved below set_page_config
    # st.info("Please ensure the CSV file is present for comprehensive answers regarding loan applications.") # Moved below set_page_config
    training_data_content = "No training dataset loaded. Specific loan application details will not be available."
except Exception as e:
    # Streamlit error for other exceptions
    # st.error(f"Error loading '{TRAINING_DATA_FILE}': {e}") # Moved below set_page_config
    training_data_content = "Error loading training dataset. Specific loan application details may be limited."


# Combine base RAG content with training data
RAG_DOCUMENT_CONTENT = BASE_RAG_DOCUMENT_CONTENT + "\n\n## Training Dataset Content:\n" + training_data_content


# --- Streamlit Application Interface ---
# st.set_page_config must be the first Streamlit command
st.set_page_config(page_title="Dream Housing Finance Chatbot", page_icon="🏠")

st.title("🏠 Dream Housing Finance Chatbot")
st.markdown("Ask me anything about the company, loan process, or dataset features.")

# Now display messages related to file loading
if training_data_content.startswith("No training dataset loaded"):
    st.warning(f"Warning: '{TRAINING_DATA_FILE}' not found in the same directory.")
    st.info("Please ensure the CSV file is present for comprehensive answers regarding loan applications.")
elif training_data_content.startswith("Error loading training dataset"):
    st.error(f"Error loading '{TRAINING_DATA_FILE}'. Specific loan application details may be limited.")
else:
    st.success(f"Successfully loaded {TRAINING_DATA_FILE}")


# --- Streamlit UI for API Key Input ---
# Use st.secrets for production, but allow input for local testing
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.warning("GOOGLE_API_KEY environment variable not set.")
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
    if not api_key:
        st.stop() # Stop execution if API key is not provided

# Set the API key for the session
os.environ["GOOGLE_API_KEY"] = api_key


# --- 2. LangChain Setup: Embeddings, Vector Store, and Retriever ---
@st.cache_resource
def setup_rag_components(rag_content):
    """
    Sets up LangChain components (embeddings, vector store, retriever) and caches them.
    Ensures an event loop is available for GoogleGenerativeAIEmbeddings.
    """
    # Explicitly get or create an event loop for this thread if one isn't running
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError: # No running event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = [Document(page_content=rag_content)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

retriever = setup_rag_components(RAG_DOCUMENT_CONTENT)

# --- 3. LLM Integration ---
@st.cache_resource
def setup_llm():
    """
    Sets up the LLM and caches it.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
    return llm

llm = setup_llm()

# --- 4. LangGraph for Conversational Flow ---

# Define the graph state
class GraphState(TypedDict):
    question: str
    chat_history: List[Tuple[str, str]]
    documents: List[Document]
    answer: str

# Define nodes for the graph
def retrieve(state: GraphState):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "chat_history": state["chat_history"]}

def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant for Dream Housing Finance company. Answer the user's question based *only* on the provided context. If you don't know the answer, state that you don't have enough information."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Context: {context}\n\nQuestion: {question}"),
        ]
    )

    context = "\n\n".join([doc.page_content for doc in documents])

    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })
    return {"answer": answer, "question": question, "documents": documents, "chat_history": chat_history}

# Build the LangGraph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph and cache it
@st.cache_resource
def compile_workflow():
    return workflow.compile()

app = compile_workflow()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial bot message
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your RAG Q&A Chatbot for Dream Housing Finance. I can answer questions about the company, its loan process, and the dataset features. What would you like to know?"})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Prepare chat history for LangGraph (converting to HumanMessage/AIMessage)
    formatted_chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            formatted_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_chat_history.append(AIMessage(content=msg["content"]))

    # Remove the last user message from formatted_chat_history as it's the current prompt
    # LangGraph expects the current question separately, and previous history
    if formatted_chat_history and formatted_chat_history[-1].content == prompt:
        formatted_chat_history.pop()


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Initial state for the graph run
                inputs = {"question": prompt, "chat_history": formatted_chat_history, "documents": [], "answer": ""}
                final_state = app.invoke(inputs)
                bot_response = final_state["answer"]
                st.write(bot_response)
            except Exception as e:
                error_message = f"An error occurred: {e}\nPlease ensure your Google API Key is correctly set and has access to Gemini models."
                st.error(error_message)
                bot_response = "I apologize, but I encountered an error. Please try again or check your API key."

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Optional: Clear chat history button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your RAG Q&A Chatbot for Dream Housing Finance. I can answer questions about the company, its loan process, and the dataset features. What would you like to know?"})
    st.rerun()
