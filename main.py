import os
import time
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from openai_key import OPENAI_API_KEY

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

run_button = st.sidebar.button("Process URLs")

index_path = "./faiss_index"

llm = ChatOpenAI(
    temperature=0.2,
    max_tokens=500
)

main_placeholder = st.empty()

# ✅ LOAD EXISTING FAISS INDEX 
if "vector_store" not in st.session_state:
    if os.path.exists(index_path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        st.session_state["vector_store"] = vectorstore


if run_button:

    loader = UnstructuredURLLoader(urls=urls)

    main_placeholder.text("Loading data...")

    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    main_placeholder.text("Creating embeddings...")

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(index_path)

    st.session_state["vector_store"] = vectorstore

    main_placeholder.text("Processing complete!")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""
)

query = st.text_input("Ask a question about the articles")

if query and "vector_store" in st.session_state:

    vectorstore = st.session_state["vector_store"]

    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(query)

    st.header("Answer")
    st.write(result)

    docs = retriever.invoke(query)

    st.subheader("Sources")
    for doc in docs:
        st.write(doc.metadata["source"])
