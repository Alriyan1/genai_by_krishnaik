from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes


model=ChatGroq(model='Gemma2-9b-It',groq_api_key='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')

generic_template='Translate the following into {language}:'

prompt=ChatPromptTemplate(
    [
        ('system',generic_template),
        ('user','{text}')
    ]
)

parser = StrOutputParser()

chain = prompt|model|parser

app = FastAPI(
    title='langchain server',
    version = '1.0',
    description='a simple API server using langchain runnable interfaces'
)

add_routes(
    app,
    chain,
    path='/chain'
)


if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)
