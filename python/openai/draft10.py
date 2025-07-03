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

## deepseek client
client_deepseek = openai.OpenAI(base_url=dotenv.dotenv_values('.env')['DEEPSEEK_BASE_URL'], api_key=dotenv.dotenv_values('.env')['DEEPSEEK_API_KEY'])
prompt = "Write a one-sentence bedtime story about a unicorn."
message_list = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
response = client_deepseek.chat.completions.create(model="deepseek-chat", messages=message_list, stream=False)
response.choices[0].message.content


prompt = "Write a one-sentence bedtime story about a unicorn."
response = client.responses.create(model="gpt-4.1", input=prompt)
response.output_text
x = response.output[0] #unsafe to assume that `.output[0].content[0].text` is available
x.id
x.role
x.type
x.content[0].type
x.content[0].text
x.content[0].annotations


## image
tmp0 = "https://upload.wikimedia.org/wikipedia/commons/3/3b/LeBron_James_Layup_%28Cleveland_vs_Brooklyn_2018%29.jpg"
response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "user", "content": "what teams are playing in this image?"},
        {"role": "user", "content": [{"type": "input_image", "image_url": tmp0}]}
    ]
)
response.output_text
# The teams playing in the image are the Cleveland Cavaliers and the Brooklyn Nets.


## web-search
prompt = "What was a positive news story from today?"
response = client.responses.create(model="gpt-4.1", tools=[{"type": "web_search_preview"}], input=prompt)
# tools=[{"type": "web_search_preview", "search_context_size": "low"}]
response.output_text
response.output[1].content[0].text
response.output[1].content[0].annotations


## streaming
prompt = [{"role": "user", "content": "Say 'double bubble bath' ten times fast."}]
stream = client.responses.create(model="gpt-4.1", input=prompt, stream=True)
z0 = list(stream)
for x in z0:
    print(x)
# response.output_text.delta
# response.content_part.added
# response.completed
# response.output_item.done
# response.content_part.done
# response.output_text.done


## reasoning (likely fail due to network error)
prompt = [{"role": "user", "content": "Write a bash script that takes a matrix represented as a string with format "
                                        "'[1,2],[3,4],[5,6]' and prints the transpose in the same format."}]
response = client.responses.create(model="o4-mini", reasoning={"effort": "medium"}, input=prompt) #max_output_tokens=300
response.output_text
response.usage.output_tokens_details.reasoning_tokens
response.status
# effort: low, medium (default), high


## agent
# pip install openai-agents
spanish_agent = agents.Agent(name="Spanish agent", instructions="You only speak Spanish.")
english_agent = agents.Agent(name="English agent", instructions="You only speak English")
prompt = "Handoff to the appropriate agent based on the language of the request."
triage_agent = agents.Agent(name="Triage agent", instructions=prompt, handoffs=[spanish_agent, english_agent])

async def hf0():
    result = await agents.Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
asyncio.run(hf0())


## instruction versus message roles
response = client.responses.create(model="gpt-4.1", instructions="Talk like a pirate.", input="Are semicolons optional in JavaScript?")

prompt = [{"role": "developer", "content": "Talk like a pirate."},
        {"role": "user", "content": "Are semicolons optional in JavaScript?"}]
response = client.responses.create(model="gpt-4.1", input=prompt)


## pdf (png) file
with open(hf_data('deep_research_blog.pdf'), "rb") as fid:
    data = fid.read()
base64_string = base64.b64encode(data).decode('utf-8')
prompt = [{"role": "user",
            "content": [{"type": "input_file", "filename": "note.pdf", "file_data": f"data:application/pdf;base64,{base64_string}"},
                        {"type": "input_text", "text": "according to the provided pdf file, generate a summary of the content."}],
        }] #generate the LaTeX source code to produce this pdf file?
response = client.responses.create(model="gpt-4.1", input=prompt)
x0 = response.output_text
