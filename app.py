pip install PyPDF2

import streamlit as st
from PyPDF2 import PdfFileReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.set_page_config(page_title="LLM on IPC", page_icon=":books:", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
"""
<style>
body {
background-image: url('https://images.unsplash.com/photo-1502691876148-84137c723a8f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1051&q=80');
background-size: cover;
}
</style>
""",
unsafe_allow_html=True
)

st.title("LLM on IPC")
st.write("Welcome to the LLM on IPC query system. Please enter your query below and click 'Search' to get an answer from the legal text.")

query = st.text_input("Enter your query here:")

if query:
    # read data from the pdf file
    with open('/path/to/LegalAI.pdf', 'rb') as f:
        pdf_reader = PdfReader(f)
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    # split the text into smaller chunks for efficient retrieval
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # create embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    # index the texts using FAISS
    docsearch = FAISS.from_texts(texts, embeddings)

    # load the question answering chain from Langchain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # search for the answer to the query in the indexed texts
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)

    st.write("Your answer is:", answer)
