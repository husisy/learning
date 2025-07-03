import openai
import dotenv
import httpx
import json
import pydantic
import requests

OPENAI_API_KEY = dotenv.dotenv_values('.env')['OPENAI_API_KEY']
OPENAI_PROXY = dotenv.dotenv_values('.env').get('OPENAI_PROXY', None)
client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(proxy=OPENAI_PROXY))

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City and country e.g. Bogotá, Colombia"}},
        "required": ["location"],
        "additionalProperties": False
    }
}]
prompt = [{"role": "user", "content": "What is the weather like in Paris today?"}]
response = client.responses.create(model="gpt-4.1", input=prompt, tools=tools)
x0 = response.output[0]
x0.name #get_weather
json.load(x0.arguments) #{"location":"Paris, France"}
x0.type #str, 'function_call'




def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
                            "&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {"latitude": {"type": "number"}, "longitude": {"type": "number"}},
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
}]
prompt = [{"role": "user", "content": "What's the weather like in Paris today?"}]
response = client.responses.create(model="gpt-4.1", input=prompt, tools=tools,)
x0 = response.output[0]
hf0 = {'get_weather':get_weather}[x0.name]
result = hf0(**json.loads(x0.arguments))

prompt.append(x0)
prompt.append({"type": "function_call_output", "call_id": x0.call_id, "output": str(result)})
response2 = client.responses.create(model='gpt-4.1', input=prompt)
response2.output_text
