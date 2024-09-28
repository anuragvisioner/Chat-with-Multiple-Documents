import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

gemini_api_key = ""
inference_api_key = ""
# Define functions before they are called
def get_pdf_text(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()  # Extract text from each page
            if text:  # Check if text exists on the page
                doc = Document(page_content=text)  # Convert text to Document object
                documents.append(doc)
    return documents  # Return a list of Document objects


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len)
    docs = text_splitter.split_documents(text)
    return docs


def get_vectorstore(text_chunks):
    embedding_model = HuggingFaceInferenceAPIEmbeddings(api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")
    vectorstore = FAISS.from_documents(text_chunks, embedding_model)
    return vectorstore

def get_conversation_chain(vectorstore):
    model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly AI Assistant that helps people with long documents. \n Note: ALWAYS START Your Answer Directly, without mentioning your name. Do Not add extra Spaces. Always write in third person.Always make your answer clear and complete. Answer only if a relevant question is asked, and if you know the answer, otherwise don't answer with unknown or vague information. Take care of the ethics. Answer the user's questions based on the relevant sentences: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above chat history, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response.get("answer")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Send a message", key="user_input", max_chars = 2000):
        if st.session_state.conversation is None:
            st.error("Please upload PDFs to start the conversation")
            return
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = process_chat(st.session_state.conversation, prompt, st.session_state.messages)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                pages = get_pdf_text(pdf_docs)

                # Get the text chunks
                chunks = get_text_chunks(pages)

                # Create vector store
                vectorstore = get_vectorstore(chunks)

                # Create conversation chain
                chain = get_conversation_chain(vectorstore)
                st.session_state.conversation = chain


if __name__ == '__main__':
    main()
