import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

#llm=ChatNVIDIA(model='meta/llama3-70b-instruct',api_key="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")

llm = ChatNVIDIA(
  model="meta/llama-3.3-70b-instruct",
  api_key="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", 
)

def vector_embedding():

    if 'vectors' not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings(api_key='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')
        st.session_state.loader=PyPDFDirectoryLoader(r"16nvidiaNIM\research_paper")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title('Nvidia NIM Demo')

prompt=ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)

prompt1=st.text_input('Enter Your Question: ')

if st.button('Document'):
    vector_embedding()
    st.write("FAISS vector store DB is ready using nvidiaEmbedding")

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever= st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander('Document similarity search'):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------------------------------')

