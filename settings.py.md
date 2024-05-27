## Code Documentation: BrainChulo Settings

### Table of Contents 

* [Introduction](#introduction)
* [Configuration Settings](#configuration-settings)
* [Function: `load_config()`](#function-load_config)

### Introduction

This code defines the `Settings` class, which manages configuration settings for the BrainChulo project. The configuration values are loaded from environment variables using the `dotenv` library. The code also includes logging setup using the `logging` module.

### Configuration Settings

The `Settings` class provides a convenient way to access and manage configuration parameters. Here's a breakdown of the available settings:

| Setting | Description | Default Value |
|---|---|---|
| `embeddings_model` | The model used for generating text embeddings. | `hkunlp/instructor-xl` |
| `chat_api_url` | The URL of the chat API endpoint. | `http://localhost:5000/api/v1/generate` |
| `memories_path` | The directory where short-term memories are stored. | `memories/` |
| `upload_path` | The directory where uploaded files are saved. | `uploads/` |
| `document_store_name` | The name of the document store used for storing documents. | `brainchulo_docs` |
| `conversation_store_name` | The name of the conversation store used for storing conversations. | `brainchulo_convos` |
| `default_objective` | The default objective for the agent. | `Be a CEO.` |
| `use_agent` | A flag indicating whether to use the agent with tools and REaCT prompt framework. | `False` |


**Environment Variables:**

* **`EMBEDDINGS_MODEL`**: Defines the embedding model to be used. 
* **`CHAT_API_URL`**: Specifies the URL of the chat API endpoint.
* **`MEMORIES_PATH`**: Sets the path for storing short-term memories.
* **`UPLOAD_PATH`**: Determines the directory for uploaded files.
* **`DOCUMENT_STORE_NAME`**: Specifies the name of the document store.
* **`CONVERSATION_STORE_NAME`**: Specifies the name of the conversation store.
* **`DEFAULT_OBJECTIVE`**: Defines the default objective for the agent.

**Logging:**

The code sets up basic logging with the following format:

```
%(asctime)s - %(levelname)s - %(message)s
```

The logging level is set to `INFO`.

### Function: `load_config()`

This function instantiates the `Settings` class and returns the resulting object. This object provides access to all the configuration settings defined in the class.

```python
def load_config():
    return Settings()
```

**Usage:**

To load the configuration settings, simply call the `load_config()` function.

```python
config = load_config()
```

You can then access the configuration values through the `config` object.

```python
print(config.embeddings_model)
print(config.chat_api_url)
```
