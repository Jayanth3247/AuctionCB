import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load .env variables
load_dotenv()
os.environ["CHROMA_TELEMETRY"] = "False"

# Streamlit UI
st.title("Auction.com Chatbot")
st.write("This chatbot is trained on foreclosure documents provided by Mike's Team.")

# Define paths
PDF_PATH = "Foreclosure_Full_Doc.pdf"
CHROMA_PATH = "./chroma_db"

# Function to build vectorstore if not exists
@st.cache_resource
def load_or_build_retriever():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if vector DB already exists
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        with st.spinner("Building vector database..."):
            loader = PyPDFLoader(PDF_PATH)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            docs = text_splitter.split_documents(data)

            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embedding_model,
                persist_directory=CHROMA_PATH
            )
            st.success("✅ Vectorstore built and saved")
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Load retriever
retriever = load_or_build_retriever()

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)

# Chat input and response
query = st.chat_input("Hello Auction.com Team! Go ahead and ask something")

if query:
    with st.spinner("Thinking..."):
        system_prompt = (
            "You are an assistant for answering questions posed by auction.com employees. Ensure that all answers are related to Auction.com data "
            "provided to you. If you don't know the answer, say that you don't know and need more data. Be detailed and thorough.\n\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})

        st.write("**User Asked:** " + query)
        st.write("**Chatbot Response:** " + response["answer"])
