from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFacePipeline

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
from server.oobabooga_llm import OobaboogaLLM
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from constants import *

import os

load_dotenv()

CHROMA_SETTINGS = {}  # Set your Chroma settings here

def load_tools(llm_model):
    file_path = TEST_FILE
    # Load text file
    with open(file_path, 'r') as file:
        text = file.read()

    # Use filename as title
    title = os.path.basename(file_path)
    docs = {title: text}
    embedding = HuggingFaceInstructEmbeddings(model_name=EMB_INSTRUCTOR_XL, model_kwargs={"device": "cuda"})
    # Create new vector store and index it
    vectordb = None
    documents = [Document(page_content=docs[title]) for title in docs]
    # Split by section, then split by token limmit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    text_splitter = TokenTextSplitter(chunk_size=512,chunk_overlap=10, encoding_name="cl100k_base")  # may be inexact
    texts = text_splitter.split_documents(texts)
    print(title)
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding)
    retriever=vectordb.as_retriever(search_kwargs={"k":4})
    print(retriever)


    def searchChroma(key_word):
        
        hf_llm = OobaboogaLLM()   
        qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",\
                                        retriever=vectordb.as_retriever(search_kwargs=dict(top_k_docs_for_context=20)),return_source_documents=False)        
        #qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.), chain_type="stuff",\
        #                                retriever=vectordb.as_retriever(search_kwargs=dict(top_k_docs_for_context=20)), return_source_documents=False)
        print(qa)
        res=qa.run(key_word)
        print(res)
        return res

    dict_tools = {
        'Chroma Search': searchChroma
    }
    return dict_tools

