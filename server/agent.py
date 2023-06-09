from dotenv import load_dotenv
import os
from chromadb.config import Settings
from colorama import Fore, Style
from server.tools import load_tools 
from langchain.chains import RetrievalQA
from server.tools import clean_text
from server.tools import ingest_file
from langchain.llms import LlamaCpp

load_dotenv()

TEST_FILE = os.getenv("TEST_FILE")

EMBEDDINGS_MODEL_NAME = os.getenv('EMBEDDINGS_MODEL')
PERSIST_DIRECTORY = os.getenv('MEMORIES_PATH')
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)


model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx =1000
target_source_chunks = os.environ.get('TARGET_SOURCE_CHUNKS')
n_gpu_layers = os.environ.get('N_GPU_LAYERS')
use_mlock = os.environ.get('USE_MLOCK')
n_batch = os.environ.get('N_BATCH') if os.environ.get('N_BATCH') != None else 512
callbacks = []
qa_prompt = ""
llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False,n_gpu_layers=n_gpu_layers, use_mlock=use_mlock,top_p=0.9, n_batch=n_batch)
valid_answers = ['Action', 'Final Answer']
valid_tools = ["Check Question", "Google Search"]
tools = load_tools()
class CustomAgentGuidance:
    def __init__(self, guidance, retriever, num_iter=3):
        self.guidance = guidance
        self.retriever = ingest_file(TEST_FILE)
        self.llm = llm

        self.num_iter = num_iter
        self.prompt_template = """
        {{#system~}}
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Instruction:
        Answer the following questions as best you can. You have access to the following tools:
        Google Search: A wrapper around Google Search. Useful for when you need to answer questions about current events. The input is the question to search relevant information.
        {{~/system}}

        {{#user~}}
        Question: {{question}}
        {{~/user}}

        {{#assistant~}}
        Thought: Let's first check our database.
        Action: Check Question
        Action Input: {{question}}
        {{~/assistant}}

        {{#user~}}
        Here are the relevant documents from our database:{{search question}}
        {{~/user}}address?

        {{#assistant~}}
        Observation: Based on the documents, I think I can reach a conclusion.
        {{#if (can_answer)}} 
        Thought: I believe I can answer the question based on the information contained in the returned documents.
        Final Answer: {{gen 'answer' temperature=0.7 max_tokens=500}}
        {{else}}
        Thought: I don't think I can answer the question based on the information contained in the returned documents.
        Final Answer: I'm sorry, but I don't have sufficient information to provide an answer to this question.
        {{/if}}

        {{~/assistant}}
        """

    def searchQA(self, t):    
        return self.checkQuestion(self.question)

    def checkQuestion(self, question: str):
        question = question.replace("Action Input: ", "")
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever, return_source_documents=True)
        answer_data = qa({"query": question})

        if 'result' not in answer_data:
            print(f"\033[1;31m{answer_data}\033[0m")
            return "Issue in retrieving the answer."

        context_documents = answer_data['source_documents']
        context = " ".join([clean_text(doc.page_content) for doc in context_documents])
        return context

    def can_answer(self, question: str):
        question = question.replace("Action Input: ", "")
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever, return_source_documents=True)
        answer_data = qa({"query": question})

        if 'result' not in answer_data:
            print(f"\033[1;31m{answer_data}\033[0m")
            return "Issue in retrieving the answer."

        answer = answer_data['result']
        context_documents = answer_data['source_documents']
        context = " ".join([clean_text(doc.page_content) for doc in context_documents])

        question_check_prompt = """###Instruction: You are an AI assistant who uses document information to answer questions. Given the following pieces of context, determine if there are any elements related to the question in the context. To assist me in this task, you have access to a vector database context that contains various documents related to different topics.Don't forget you MUST answer with 'yes' or 'no'
 
        Context:{context}
        Question: Do you think it would be possible to infer an answer to this question: ""{question}"" from the provided context? You MUST include'yes' or 'no' in your answer.
        ### Response:
        """.format(context=context, question=question)
        
        print(Fore.GREEN + Style.BRIGHT + question_check_prompt + Style.RESET_ALL)
        answerable = self.llm(question_check_prompt)
        print(Fore.RED + Style.BRIGHT + context + Style.RESET_ALL)
        print(Fore.RED + Style.BRIGHT + answerable + Style.RESET_ALL)
        if "yes" in answerable.lower():
            return True
        else:
            return False

    def __call__(self, question):
        self.question = question
        prompt = self.guidance(self.prompt_template)
        result = prompt(question=question, search=self.searchQA, can_answer=self.can_answer(question),valid_answers=valid_answers, valid_tools=valid_tools)
        return result
