import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

st.set_page_config(page_icon='🦜',page_title='Langchain: Summarize Text from YT or Website')
st.title('🦜 Langchain: Summarize text from YT or website')
st.subheader('Summarize URL')

with st.sidebar:
    groq_api_key=st.text_input('Groq API key',value='',type='password')

generic_url = st.text_input('URL',label_visibility='collapsed')

if groq_api_key:
    llm=ChatGroq(api_key=groq_api_key,model="Llama3-8b-8192")

prompt_template="""
provide a summary of the following content in 300 words:
content:{text}
"""

prompt=PromptTemplate(template=prompt_template,input_variables=['text'])

if st.button('Summarize the content from YT or website'):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error('Please enter a valid URL. It can may be a YT video url or website url')

    else:
        try:
            with st.spinner('Waiting...'):
                if 'youtube.com' in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")

