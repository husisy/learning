# openai

1. link
   * [documentation](https://platform.openai.com/docs/introduction)
   * [openai/playground](https://platform.openai.com/playground)
   * [openai/documentation/prompt-design](https://platform.openai.com/docs/guides/completion/prompt-design)
   * model comparison tool [link](https://gpttools.com/comparisontool)
   * openai lab-interface [link](https://labs.openai.com/)
   * openai web crawl QA tutorial [link](https://platform.openai.com/docs/tutorials/web-qa-embeddings)
   * [github/openai-python](https://github.com/openai/openai-python)
2. install
   * `conda install -c conda-forge openai`
   * `pip install openai`
3. token
   * many tokens start with a white space: `" hello"`, `" bye"`
   * `1` token is approximately `4` characters or `0.75` words for English text
4. setting
   * temperature: `[0,1]`, `0` for identitcal or very similar
   * top-p: `[0,1]` the smallest possible set of words whose cumulative probability exceeds the probability `p` [link](https://community.openai.com/t/a-better-explanation-of-top-p/2426/2) [huggingface-blog](https://huggingface.co/blog/how-to-generate)
   * model: `text-davinci-003`
5. for most models, a single API request can only process up to `2048` tokens (roughly `1500` words)
6. vector database
   * `pinecone`
   * `weaviate`
   * `redis`
   * `qdrant`
   * `milvus`
   * `chroma`
7. suggestions from openai
   * show and tell
   * provide quality data
   * check your settings: if only one right answer `temperature=0, top_p=1`
8. Chat Markup Language (ChatML)
9. ChatGPT
   * `role`: `system`, `user`, `assistant`

minimum python environment in VPS

```bash
micromamba create -n test00
micromamba install -n test00 -c conda-forge cython matplotlib h5py pandas pillow protobuf scipy requests tqdm flask ipython openai python-dotenv tiktoken beautifulsoup4 pandas
```

`.env` file

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# list models
openai api models.list

# create completion
openai api completions.create -m ada -p "hello world"
# hello world. Their loyal following grows.

# create a chat completion
openai api chat_completions.create -m gpt-3.5-turbo -g user "hello world"
# Hello there! How can I assist you today?

openai api image.create -p "two dogs playing chess, cartoon" -n 1
```

## PROMPT example

`%EOL%` is the end of line, `%BOL%` is the beginning of line

1. `suggest one name for a horse`
2. `suggest one name for a black horse`
3. `suggest three names for a horse that is a superhero`
4. see code block below

```text
Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: Horse
Names:
```

classfication

```text
Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new Batman movie!
Sentiment:
```

```text
Classify the sentiment in these tweets:

1. "I can't stand homework"
2. "This sucks. I'm bored 😠"
3. "I can't wait for Halloween!!!"
4. "My cat is adorable ❤️❤️"
5. "I hate chocolate"

Tweet sentiment ratings:
```

generation

```text
Brainstorm some ideas combining VR and fitness:
```

conversation

1. tell the API the intent but we also tell it how to behave
2. give the API an identity

```text
The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.

Human: Hello, who are you?
AI: I am an AI created by OpenAI. How can I help you today?
Human:%EOL%

%BOL%Could you help me find a good restaurant near me?
AI: Absolutely! What type of cuisine are you looking for?
```

```text
Marv is a chatbot that reluctantly answers questions with sarcastic responses:

You: How many pounds are in a kilogram?
Marv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.
You: What does HTML stand for?
Marv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.
You: When did the first airplane fly?
Marv: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they’d come and take me away.
You: What is the meaning of life?
Marv: I’m not sure. I’ll ask my friend Google.
You: What time is it?
Marv: %EOL%

%BOL%It's always time to ask better questions.
```

## ChatGPT

1. `finish_reason`
   * `stop`: API returned complete model output
   * `length`: incomplete model output due to `max_tokens` parameter or token limit
   * `content_filter`: omitted content due to a flag from our content filters
   * `null`: API response still in progress on incomplete

```json
{
    "id": "chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve",
    "object": "chat.completion",
    "created": 1677649420,
    "model": "gpt-3.5-turbo",
    "usage": {
        "prompt_tokens": 56,
        "completion_tokens": 31,
        "total_tokens": 87
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "The 2020 World Series was played in Arlington, Texas at the Globe Life Field, which was the new home stadium for the Texas Rangers."
            },
            "finish_reason": "stop",
            "index": 0
        }
    ]
}
```

```text
system: You are a helpful assistant.
user: Who won the world series in 2020?
assistent: The Los Angeles Dodgers won the World Series in 2020.
user: Where was it played?
assistent: The 2020 World Series was held at Globe Life Field in Arlington, Texas, which is the home ballpark of the Texas Rangers.
```

## model

1. price (USD/token)
2. `5USD` in free credit for first 3 months
3. gpt4: broad general knowledge and domain expertise
   * `gpt-4-8K`: 8K context, prompt `0.03/1K`, completion `0.06/1K`
   * `gpt-4-32K`: 32K context, prompt `0.06/1K`, completion `0.12/1K`
4. `gpt-3.5-turbo` (chatgpt): for dialogue, the performance is on par with `text-davinci-003`, `0.002/1K`
5. InstructGPT
   * optimized to follow single-turn instructions
   * can be used to fine-tuning (the price for fine-tunned model is higher)
   * Ada: the fastest, embedding `0.0004/1K`
   * `text-davinci-003`: the most powerful `0.02/1K`
   * babage: `0.0005/1K`
   * curie: `0.0002/1K`
6. DALL-E: image model
   * `1024x1024`: `0.02/image`
   * `512x512`: `0.018/image`
   * `256x256`: `0.016/image`
7. Whisper: audio model
   * `0.006/minute`

## tiktoken

1. link
   * [github](https://github.com/openai/tiktoken)
2. install
   * `conda install -c conda-forge tiktoken`
   * `pip install tiktoken`

```Python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")
```

## BeautifulSoup

1. link
   * [code-repository](https://code.launchpad.net/beautifulsoup/)
   * [documentation](https://beautiful-soup-4.readthedocs.io/en/latest/)
2. install
   * `conda install -c conda-forge beautifulsoup4`
   * `pip install beautifulsoup4`

## python pdf package

1. link
   * [github/pdfminer.six](https://github.com/pdfminer/pdfminer.six) choose this one (no too much reason)
   * [github/pymupdf](https://github.com/pymupdf/PyMuPDF)
   * [github/pikepdf](https://github.com/pikepdf/pikepdf) 反对文字提取功能
   * [github/pypdf](https://github.com/py-pdf/pypdf)
2. install
   * `pip install "pdfminer.six"`
   * `conda install -c conda-forge "pdfminer.six"`

TODO pdfminzer [link](https://github.com/pdfminer/pdfminer.six)

TODO chatgpt-telegram-bot [link](https://github.com/zzh1996/chatgpt-telegram-bot)