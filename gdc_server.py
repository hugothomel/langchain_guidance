import gradio as gr
import guidance
import torch
from server.model import load_model_main
from server.tools import load_tools
from server.agent import CustomAgentGuidance
import os
from langchain.llms import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from constants import *
import nest_asyncio

app = Flask(__name__)
CORS(app)

nest_asyncio.apply()
MODEL_PATH = MODEL
#CHECKPOINT_PATH = ''
DEVICE = torch.device('cuda:0')


llama = None
dict_tools = None

@app.route('/load_model', methods=['POST'])
@cross_origin()
def load_model():
    print("Loading model....")
    global llama
    llama = guidance.llms.Transformers(MODEL_PATH, device_map="auto", load_in_8bit=True)
    guidance.llm = llama
    return 'Model loaded successfully'

@app.route('/load_tools', methods=['POST'])
@cross_origin()
def load_tools_route():
    global dict_tools
    if llama is None:
        return 'Model is not loaded. Load the model first', 400
    dict_tools = load_tools(llama)
    return 'Tools loaded successfully'


@app.route('/run_script', methods=['POST'])
@cross_origin()
def run_script():
    global dict_tools
    if dict_tools is None:
        return 'Tools are not loaded. Load the tools first', 400
    question = request.json.get('question', 'Who is the main character?')
    custom_agent = CustomAgentGuidance(guidance, dict_tools)
    final_answer = custom_agent(question)
    return str(final_answer), str(final_answer['fn'])


CHROMA_SETTINGS = {}  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)