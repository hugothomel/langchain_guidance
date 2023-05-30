from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
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

TEST_FILE = os.getenv("TEST_FILE")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

EMBEDDINGS_MAP = {
    **{name: HuggingFaceInstructEmbeddings for name in ["hkunlp/instructor-xl", "hkunlp/instructor-large"]},
    **{name: HuggingFaceEmbeddings for name in ["all-MiniLM-L6-v2", "sentence-t5-xxl"]}
}

CHROMA_SETTINGS = {}  # Set your Chroma settings here

def load_unstructured_document(document: str) -> list[Document]:
    with open(document, 'r') as file:
        text = file.read()
    title = os.path.basename(document)
    return [Document(page_content=text, metadata={"title": title})]

def split_documents(documents: list[Document], chunk_size: int = 100, chunk_overlap: int = 0) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def load_tools(llm_model):
    def ingest_file(file_path):
        # Load unstructured document
        documents = load_unstructured_document(file_path)

        # Split documents into chunks
        documents = split_documents(documents, chunk_size=100, chunk_overlap=20)

        # Determine the embedding model to use
        EmbeddingsModel = EMBEDDINGS_MAP.get(EMBEDDINGS_MODEL)
        if EmbeddingsModel is None:
            raise ValueError(f"Invalid embeddings model: {EMBEDDINGS_MODEL}")

        model_kwargs = {"device": "cuda:0"} if EmbeddingsModel == HuggingFaceInstructEmbeddings else {}
        embedding = EmbeddingsModel(model_name=EMBEDDINGS_MODEL, model_kwargs=model_kwargs)

        # Store embeddings from the chunked documents
        vectordb = Chroma.from_documents(documents=documents, embedding=embedding)

        retriever = vectordb.as_retriever(search_kwargs={"k":4})

        print(file_path)
        print(retriever)

        return retriever, file_path

    file_path = TEST_FILE
    retriever, title = ingest_file(file_path)

    def searchChroma(key_word):
        hf_llm = OobaboogaLLM()  # commented out
        qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

        print(qa)
        res = qa.run(key_word)
        print(res)
        return res

    dict_tools = {
        'Chroma Search': searchChroma,
        'File Ingestion': ingest_file,
    }

    return dict_tools
