import sys
import os
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

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = "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"

# Load documents from the "prospectus" directory
documents = []
for file in os.listdir("prospectus"):
    if file.endswith(".pdf"):
        pdf_path = "./prospectus/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith(".docx") or file.endswith(".doc"):
        doc_path = "./prospectus/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        text_path = "./prospectus/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

# Create vector database using OpenAIEmbeddings and persist it to disk
vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()

# Create ConversationalRetrievalChain using ChatOpenAI LLM and vector database
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

# Define color codes for prompts and answers
yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

# Initialize conversation history
chat_history = []

# Greet the user and start the chat loop
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome!')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")

    # Handle exit commands
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()

    # Skip empty prompts
    if query == '':
        continue

    # Generate response using the ConversationalRetrievalChain and add it to conversation history
    result = pdf_qa(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))

# Save the trained model and components to a file
import joblib

model_data = {
    "vectordb": vectordb,
    "pdf_qa": pdf_qa,
    "chat_history": chat_history,
}

joblib.dump(model_data, "model_data.pkl")

print("Model and components have been saved.")