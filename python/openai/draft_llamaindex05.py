# https://docs.llamaindex.ai/en/stable/understanding/loading/loading/
import os
import dotenv
import pydantic
import asyncio

import llama_index
import llama_index.core

ENV_CONFIG = dotenv.dotenv_values('.env')
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']

documents = llama_index.core.SimpleDirectoryReader("./data").load_data()
documents[0].metadata

vector_index = llama_index.core.VectorStoreIndex.from_documents(documents) #show_progress=True
x0 = vector_index.as_query_engine()
