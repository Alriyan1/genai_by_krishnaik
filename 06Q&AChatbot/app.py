import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# LANGSMITH_TRACING='true'
# LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
# LANGSMITH_API_KEY="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"
# LANGSMITH_PROJECT="pr-standard-croissant-57"
# GROQ_API_KEY="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"


prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"you are a helpful assistant. please response to the user queries"),
        ('user','question: {question}')
    ]
)

def gen_response(question,api_key,llm,temperature,max_tokens):
    llm=ChatGroq(model=llm,api_key=api_key,max_tokens=max_tokens,temperature=temperature)
    #model=ChatGroq(model='Gemma2-9b-It',groq_api_key='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')
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer


st.title('Settings')
api_key=st.sidebar.text_input('Enter your API Key: ',type='password')

llm=st.sidebar.selectbox('Select an Chat model',['llama-3.3-70b-versatile','gemma2-9b-it','deepseek-r1-distill-llama-70b'])

temperature = st.sidebar.slider('Temperature',min_value=0.0,max_value=1.0,value=0.7)
max_tokens= st.sidebar.slider('Max. Tokens',min_value=50,max_value=500,value=150)

st.write("Go ahead and ask any question")
user_input=st.text_input('You:')

if user_input and api_key:
    response= gen_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)

else:
    st.write('Please provide the user input')

