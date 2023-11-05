import sys
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

st.title("Ask Me Anything")

os.environ["OPENAI_API_KEY"] = "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"

# Define the global 'documents' list
documents = []

# Define a flag to indicate if documents have been loaded and the model is initialized
initialized = False

# Define an initial chat history
chat_history = []

# Function to load documents
def load_documents():
    global documents
    for file in os.listdir("prospectus"):
        if file.endswith(".pdf"):
            pdf_path = "./prospectus/" + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = "./prospectus/" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = "./prospectus/" + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

# Function to initialize the model
def initialize_model():
    global pdf_qa, documents, initialized, chat_history
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
    vectordb.persist()

    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
        vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

    initialized = True

# User input
user_input = st.text_input("Ask a question:")
if st.button("Ask"):
    if user_input:
        if not initialized:
            load_documents()
            initialize_model()

        # Create an input dictionary with 'question' and 'chat_history'
        input_data = {"question": user_input, "chat_history": chat_history}
        result = pdf_qa(input_data)
        
        # Update chat history with the new question and answer
        chat_history.append((user_input, result["answer"]))

        st.write("Answer:", result["answer"])

