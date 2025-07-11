import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings 
#from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


llm= ChatGroq(model='llama-3.1-8b-instant',api_key='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')

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    question:{input}
    """
)

def create_vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        st.session_state.loader=PyPDFDirectoryLoader(r'07ragQ&A\research_papers')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
st.title('RAG Document Q&A with groq and lama3')

user_prompt=st.text_input("Enter your query from the research paper")

if st.button('Document Embeddings'):

    create_vector_embedding()
    st.write('Vector database is ready')


import time

if user_prompt:
    document_chain= create_stuff_documents_chain(llm,prompt)
    retriver=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriver,document_chain)

    start=time.process_time()
    response=retriever_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    with st.expander('Document similarity Search'):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

