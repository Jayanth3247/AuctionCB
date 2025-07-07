import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["CHROMA_TELEMETRY"] = "False"

st.title("Auction.com Chatbot")
st.write("This chatbot is trained on 2 foreclosure documents provided by Mike's Team.")

# Cache the retriever
@st.cache_resource
def load_retriever():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

retriever = load_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)

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
