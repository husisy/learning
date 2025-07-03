# openai

1. link
   * [documentation](https://platform.openai.com/docs/introduction)
   * [openai/playground](https://platform.openai.com/playground)
   * [openai/documentation/prompt-design](https://platform.openai.com/docs/guides/completion/prompt-design)
   * model comparison tool [link](https://gpttools.com/comparisontool)
   * openai lab-interface [link](https://labs.openai.com/)
   * openai web crawl QA tutorial [link](https://platform.openai.com/docs/tutorials/web-qa-embeddings)
   * [github/openai-python](https://github.com/openai/openai-python)
   * [github/openai-responses-starter-app](https://github.com/openai/openai-responses-starter-app)
   * [openai/model-list](https://platform.openai.com/docs/models)
   * [openai/model-spec](https://model-spec.openai.com/2025-02-12.html)
   * [openai-cookbook](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)
   * [github/rich](https://github.com/Textualize/rich) `from rich import print`
2. install
   * `conda install -c conda-forge openai openai-agents`
   * `pip install openai openai-agents "httpx[socks]" rich`
3. concept
   * Chain-of-Thought (CoT)
   * Chat Markup Language (ChatML)
   * vector database: `pinecone`, `weaviate`, `redis`, `qdrant`, `milvus`, `chroma`
   * message roles, instruction following
   * [link](https://model-spec.openai.com/2025-02-12.html#chain_of_command) authority level (from higher to lower): `platform`, `developer`, `user`, `guideline`, no authority (`assistant`, tool messaging)
   * prompt caching: generally remain active for 5 to 10 minutes of inactivity
   * few shot learning
   * retrieval-augmented generation (RAG)
   * context window
   * token: `1` token is approximately `4` characters or `0.75` words for English text
   * assistant API (beta)
4. setting
   * temperature: `[0,1]`, `0` for identitcal or very similar
   * top-p: `[0,1]` the smallest possible set of words whose cumulative probability exceeds the probability `p` [link](https://community.openai.com/t/a-better-explanation-of-top-p/2426/2) [huggingface-blog](https://huggingface.co/blog/how-to-generate)
   * check your settings: if only one right answer `temperature=0, top_p=1`
5. instruction versus message roles
   * `instruction=` has a higher priority than `input=`
   * `instruction=` only applies to the current response generation
6. how to choose: reasoning model (`o3`, `o4-mini`), GPT model (`gpt-4.1`)
   * Speed and cost: GPT models are faster and cheaper
   * Executing well defined tasks: GPT models handle explicitly defined tasks well
   * Accuracy and reliability: o-series models are reliable decision makers
   * Complex problem-solving: o-series models work through ambiguity and complexity

minimum python environment in VPS

```bash
micromamba create -n openai
micromamba install -y -n openai -c conda-forge "numpy=1.26" cython matplotlib h5py pillow scipy requests tqdm flask ipython openai python-dotenv tiktoken openai-agents
micromamba activate openai
pip install torch==2.6 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# transformers only support torch<2.6
pip install 'transformers[torch]'
pip install open-webui hf-xet "httpx[socks]"
```

| model | o4-mini | o3 | gpt-4.1 |
|-------|---------|----|---------|
| input (USD/million tokens) | 1.1 | 10 | 2 |
| output (USD/million tokens) | 4.4 | 40 | 8 |
| window | 200,000 | 200,000 | 1,047,576 |
| Max Output Tokens | 100,000 | 100,000 | 32,768 |
| Knowledge Cutoff | 20240601 | 20240601 | 20240601 |

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

## Azure/openai

1. link
   * [MS-doc/get-started](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart)

## Google/Gemini

1. link
2. instal `pip install --upgrade google-cloud-aiplatform`

## misc

github/litellm

github/open-interpreter

[github/lit-gpt](https://github.com/Lightning-AI/lit-gpt)

## github-copilot

1. link
   * [github-link](https://github.com/microsoft/Mastering-GitHub-Copilot-for-Paired-Programming) Mastering-GitHub-Copilot-for-Paired-Programming
   * [github-link](https://github.com/tensorchord/Awesome-LLMOps) awesome-llmops
   * [github-link](https://github.com/Giskard-AI/giskard) giskard, The testing framework for ML models, from tabular to LLMs

## ollama

1. link
   * [github/ollama](https://github.com/jmorganca/ollama)
   * [documentation/api](https://github.com/jmorganca/ollama/blob/main/docs/api.md)
   * [codellama](https://ollama.ai/blog/how-to-prompt-code-llama)

## litellm

1. link
   * [github/litellm](https://github.com/BerriAI/litellm)
   * [github/awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)
   * [github/awesome-code-LLM](https://github.com/huybery/Awesome-Code-LLM)
2. install
   * `pip install litellm`
3. `top_p`, `top_k`, `temperature`

other company

1. monitor
   * [langfuse](https://langfuse.com/)
   * [langchain](https://www.langchain.com/)
   * [helicone](https://www.helicone.ai/)
   * [promptlayer](https://promptlayer.com/)
   * [traceloop](https://www.traceloop.com/)
   * [slack](https://slack.dev/bolt-python/concepts)
2. Bedrock (amazon, Anthropic) [blog-link](https://www.amazon.science/news-and-features/amazon-bedrock-offers-access-to-multiple-generative-ai-models)
3. Cohere (Canadian) [official-website](https://cohere.com/)
4. code-related
   * [continue](https://continue.dev/)
   * [cody](https://sourcegraph.com/cody)

## llama-index

1. link
   * [github](https://github.com/run-llama/llama_index)
   * [documentation](https://docs.llamaindex.ai/en/stable/index.html)
   * [llamahub](https://llamahub.ai/)
2. install
   * `pip install llama-index`
   * `mamba install -c conda-forge llama-index`
3. concept
   * retrieval augmented generation (RAG)
   * stages: loading, indexing, storing, querying, evaluation
   * `Document`: a PDF, an API output, retrieve data from a database
   * `node`: a "chunk" of a `Document`
   * `Connector,Reader`
   * plan, orchestrate
4. use cases
   * QA: semantic search (top-k search)
5. alternatives
   * llama-index: excels in retrieval-centric applications but is less focused on multi-agent orchestration
   * CrewAI: robust multi-agent orchestration with modular designs suitable for complex workflows
   * smolagents: simplicity and minimalism for lightweight agentic AI
   * langchain: robust multi-agent orchestration with modular designs suitable for complex workflows
   * Microsoft AutoGen: strong in asynchronous, real-time multi-agent conversations
   * Microsoft Semantic Kernel: Enterprise data integration for enhanced LLM augmentation

```bash
mamba install -c conda-forge llama-index
pip install llama-index-tools-yahoo-finance
pip install llama-index-tools-tavily-research
pip install llama-index-utils-workflow
```
