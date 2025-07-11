from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate(
    [
        ('system','you are a helpful assistant. please respond to the question asked'),
        ('user','questions:{questions}')
    ]
)

st.title('Langchain demo with google with LLAMA2')
query=st.text_input('what question you have in mind')

llm = Ollama(model='gemma:2b')
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if query:
    st.write(chain.invoke({'questions':query}))
