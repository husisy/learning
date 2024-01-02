import os
import sys
import json
import logging
import dotenv

import llama_index

ENV_CONFIG = dotenv.dotenv_values('.env') #not sync in .git
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']

## print detailed information
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


## ollama/llama2
llm = llama_index.llms.Ollama(model="llama2")
response = llm.complete("Who is Richard Feynman?")
print(response)

messages = [
    llama_index.llms.ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    llama_index.llms.ChatMessage(role="user", content="What is your name"),
]
response = llm.chat(messages)


## openai
# https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html
# https://raw.githubusercontent.com/run-llama/llama_index/main/examples/paul_graham_essay/data/paul_graham_essay.txt
DATA_DIR = 'data/paul_graham/data'
PERSIST_DIR = 'data/paul_graham/storage'
if not os.path.exists(PERSIST_DIR):
    documents = llama_index.SimpleDirectoryReader(DATA_DIR).load_data()
    index = llama_index.VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = llama_index.StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = llama_index.load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

response.source_nodes[0].text

query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("What did the author do growing up?")
