import os
import dotenv
import asyncio

import llama_index
import llama_index.core
import llama_index.core.workflow
import llama_index.core.agent.workflow
import llama_index.llms.openai

ENV_CONFIG = dotenv.dotenv_values('.env') #not sync in .git
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']

def hf_llama_index_agent_run(agent, prompt:str|list[str], ctx=None):
    # as `asyncio.run(agent.run(prompt))` will fail
    isone = isinstance(prompt, str)
    if isone:
        prompt:list[str] = [prompt]
    assert all(isinstance(p, str) for p in prompt), "All prompts must be strings."
    async def hf_agent_run(prompt:list[str]):
        ret = []
        for p in prompt:
            ret.append(await agent.run(p, ctx=ctx))
        return ret
    ret = asyncio.run(hf_agent_run(prompt))
    if isone:
        ret = ret[0]
    return ret


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

agent = llama_index.core.agent.workflow.FunctionAgent(
    tools=[multiply],
    llm=llama_index.llms.openai.OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)

x0 = hf_llama_index_agent_run(agent, "What is 1234 * 4567?")
print(x0.response.content)
# asyncio.run(agent.run("What is 1234 * 4567?")) #fail


# context
ctx = llama_index.core.workflow.Context(agent)
prompt = ['My name is Logan', 'What is my name?']
x0,x1 = hf_llama_index_agent_run(agent, prompt, ctx=ctx)
print(x0, x1, sep='\n')
# asyncio.run(ctx.get('memory')).to_dict()['chat_store']['store']['chat_history']
ctx_dict = ctx.to_dict(serializer=llama_index.core.workflow.JsonSerializer()) #JsonPickleSerializer
restored_ctx = llama_index.core.workflow.Context.from_dict(agent, ctx_dict, serializer=llama_index.core.workflow.JsonSerializer())

## RAG
# wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -O data/paul_graham_essay.txt
documents = llama_index.core.SimpleDirectoryReader("data").load_data()
storage_dir = 'llama_index_storage'
if os.path.exists(storage_dir):
    storage_context = llama_index.core.StorageContext.from_defaults(persist_dir=storage_dir)
    index = llama_index.core.load_index_from_storage(storage_context)
else:
    index = llama_index.core.VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=storage_dir)
query_engine = index.as_query_engine()

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)

tmp0 = ("You are a helpful assistant that can perform calculations "
    "and search through documents to answer questions.")
agent = llama_index.core.agent.workflow.FunctionAgent(tools=[multiply, search_documents],
    llm=llama_index.llms.openai.OpenAI(model="gpt-4o-mini"), system_prompt=tmp0)

x0 = hf_llama_index_agent_run(agent, "What did the author do in college? Also, what's 7 * 8?")
