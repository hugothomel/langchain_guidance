## Internal Code Documentation: Server Application 

### Table of Contents

* [**Introduction**](#introduction)
* [**Module Imports and Initializations**](#module-imports-and-initializations)
* [**Flask App Configuration**](#flask-app-configuration)
* [**Model Loading Route**](#model-loading-route)
* [**Tools Loading Route**](#tools-loading-route)
* [**Run Script Route**](#run-script-route)
* [**Reload Modules Route**](#reload-modules-route)
* [**Main Function**](#main-function)


### Introduction 

This Python code implements a Flask server application designed for running a custom agent using Guidance and a large language model (LLM). The server provides functionalities for loading the model, tools, and executing scripts based on user input.

### Module Imports and Initializations

The code begins by importing necessary modules:

| Module | Description |
|---|---|
| `importlib` | Provides the `reload` function for reloading modules. |
| `server.tools` | Contains custom tools for the agent. |
| `flask` | The web framework for building the server. |
| `flask_cors` | Enables cross-origin requests. |
| `constants` | Defines constants used throughout the code. |
| `nest_asyncio` | Allows asynchronous operations within Flask. |
| `guidance` | The library for creating and running custom agents. |
| `torch` |  The deep learning framework for handling the model. |
| `server.model` | Contains the model loading function. |
| `server.agent` | Contains the custom agent class. |
| `os` | Provides operating system functionalities. |
| `langchain.llms` |  Provides the `OpenAI` class for accessing OpenAI LLMs. |
| `colorama` | Used for colored console output. |

The code then initializes the Flask application and applies `nest_asyncio` for asynchronous operations. It also defines variables for the model path (`MODEL_PATH`) and the device (`DEVICE`).

Global variables `llama` and `dict_tools` are initialized as `None`. These variables will store the loaded LLM and tools respectively.

### Flask App Configuration

The code configures the Flask application by:

* Enabling CORS to allow cross-origin requests.
* Setting the `debug` flag to `False` to disable debug mode in production.

### Model Loading Route

The `/load_model` route handles the loading of the LLM model:

* It takes a POST request.
* Loads the LLM model from `MODEL_PATH` using `guidance.llms.Transformers` and sets the `device_map` to "auto" for automatic device allocation. 
* Sets `load_in_8bit` to `True` to optimize memory usage.
* Assigns the loaded LLM to the global variable `llama` and updates `guidance.llm` with the loaded model.
* Returns a success message.

### Tools Loading Route

The `/load_tools` route handles the loading of the tools:

* It takes a POST request.
* Checks if the LLM has been loaded. If not, it returns an error message.
* Reloads the `server.tools` module using `reload` to ensure the latest version is loaded.
* Loads the tools using `load_tools(llama)` from `server.tools` and assigns them to the global variable `dict_tools`.
* Returns a success message.

### Run Script Route

The `/run_script` route handles execution of scripts with the loaded tools and LLM:

* It takes a POST request.
* Checks if the tools have been loaded. If not, it returns an error message.
* Retrieves the user's question from the request.
* Creates a `CustomAgentGuidance` object with the loaded tools and guidance.
* Executes the agent with the user's question and retrieves the final answer.
* Returns the final answer and the function name used to generate the answer.

### Reload Modules Route

The `/reload_modules` route handles reloading of all relevant modules:

* It takes a POST request.
* Reloads the `server.tools`, `server.agent`, and `server.model` modules using `reload`.
* Re-imports functions and classes from the reloaded modules.
* Returns a success message.

### Main Function

The main function starts the Flask application:

* It runs the Flask app on host `0.0.0.0`, port `5001`, and with `debug` mode set to `False`. 

This allows the server to listen for requests on all available interfaces and run without the debug mode enabled.
