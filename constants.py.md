##  Code Documentation: Model Configuration 

**Table of Contents:**

*  [Model Configuration Variables](#model-configuration-variables)
    * [Embedding Models](#embedding-models)
    * [Language Models](#language-models)
    * [Other Variables](#other-variables)

### Model Configuration Variables 

This code defines variables for configuring different models and file paths used in a text generation system. 

#### Embedding Models 

| Variable Name | Description | Value |
|---|---|---|
| `EMB_OPENAI_ADA` | OpenAI's text embedding model `ada-002`. |  `text-embedding-ada-002` | 
| `EMB_INSTRUCTOR_XL` | The Hugging Face `instructor-xl` model for text embedding. | `hkunlp/instructor-xl` |

#### Language Models 

| Variable Name | Description | Value |
|---|---|---|
| `LLM_OPENAI_GPT35` | OpenAI's GPT-3.5 Turbo model. | `gpt-3.5-turbo` |
| `OOBA` |  A reference to the `oobabooga` text generation system. | `oobabooga` | 

#### Other Variables 

| Variable Name | Description | Value |
|---|---|---|
| `TEST_FILE` | Path to the `macbeth.txt` file, used for testing. | `/home/karajan/Documents/macbeth.txt` | 
| `MODEL` | Path to the `alpaca-native` model directory. | `'/home/karajan/labzone/textgen/text-generation-webui/models/alpaca-native'` | 
