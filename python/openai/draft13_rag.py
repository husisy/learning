import os
import json
import base64
import httpx
import dotenv
import asyncio
import openai
import agents

hf_data = lambda *x: os.path.join('data', *x)

# dotenv.load_dotenv('.env') #OPENAI_API_KEY
OPENAI_API_KEY = dotenv.dotenv_values('.env')['OPENAI_API_KEY']
OPENAI_PROXY = dotenv.dotenv_values('.env').get('OPENAI_PROXY', None)
client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(proxy=OPENAI_PROXY))


## vector store, file search
x0 = client.vector_stores.list()
name_list = [x.name for x in x0.data]
# client.vector_stores.delete(vector_store_id=x0.data[0].id)

vector_store = client.vector_stores.create(name="test00")
with open(hf_data('deep_research_blog.pdf'), "rb") as fid:
    # filename is also available in fid
    file_obj = client.vector_stores.files.upload_and_poll(vector_store_id=vector_store.id, file=fid)
    # client.vector_stores.files.upload is async
# client.vector_stores.files.update(vector_store_id=vector_store.id, file_id=file_obj.id, attributes={"key": "value"})
file_obj_list = client.vector_stores.files.list(vector_store_id=vector_store.id)
prompt = "What is deep research by OpenAI?"
results = client.vector_stores.search(vector_store_id=vector_store.id, query=prompt, rewrite_query=True)
results.data[0].filename
results.search_query #rewritten query by rewrite_query=True
results.data[0].score #similarity scores
results.data[0].attributes
results.data[0].content[0].text #text related to the prompt


prompt = "What is deep research by OpenAI?"
response = client.responses.create(model="gpt-4o-mini", input=prompt, tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}])
# tools=[{"max_num_results": 2}] #10 default
# include=["file_search_call.results"]
response.output[0].type #file_search_call
response.output[1].type #message
response.output[1].content[0].annotations #file_id


## custom format
def custom_format_results(results):
    ret = '<sources>'
    for x0 in results.data:
        x1 = f"<result file_id='{x0.file_id}' file_name='{x0.filename}'>"
        ret += x1 + ''.join((f"<content>{y.text}</content>" for y in x0.content)) + "</result>"
    ret = ret + "</sources>"
    return ret

prompt = "What is deep research by OpenAI?"
results = client.vector_stores.search(vector_store_id=vector_store.id, query=prompt, rewrite_query=True)

prompt1 = [{"role": "developer", "content": "Produce a concise answer to the query based on the provided sources."},
        {"role": "user", "content": f"Sources: {custom_format_results(results)}\n\nQuery: '{prompt}'"}]
response = client.responses.create(model="gpt-4.1", input=prompt1)
