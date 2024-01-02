import os
import json
import dotenv
from tqdm import tqdm
import numpy as np

import litellm

ENV_CONFIG = dotenv.dotenv_values('.env') #not sync in .git
os.environ["OPENAI_API_KEY"] = ENV_CONFIG['OPENAI_API_KEY']

# gpt-3.5-turbo gpt-3.5-turbo-1106
response = litellm.completion(model="gpt-3.5-turbo-1106", messages=[{"content": "Hello, how are you?","role": "user"}])
response.choices[0].message.content
cost = litellm.completion_cost(completion_response=response) #response.model no need to pass manually
print(f"Cost for completion call with {response.model}: ${float(cost):.10f}")

response = litellm.completion(model="gpt-3.5-turbo-1106", messages=[{ "content": "Hello, how are you?","role": "user"}], stream=True)
chunk_list = [x for x in tqdm(response)]
assert chunk_list[-1].choices[0].finish_reason=='stop'
''.join(x.choices[0].delta.content for x in chunk_list[:-1])

response = litellm.embedding(model='text-embedding-ada-002', input=["good morning from litellm"])
np0 = np.array(response.data[0]['embedding']) #(np,float64,1536)


## llama2(default 7b) llama2:13b llama2:70b llama2-uncensored codellama
response = litellm.completion(model="ollama/llama2", messages=[{"content": "Hello, how are you?","role": "user"}], api_base="http://localhost:11434")
# ollama/llama2 stream=True to be support in litellm-1.15.3
# response = litellm.embedding(model='ollama/llama2', input=["good morning from litellm"]) #not support



# Example dummy function hard coded to return the same weather
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def demo_parallel_function_call():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    # tool_choice=auto is default
    response = litellm.completion(model="gpt-3.5-turbo-1106", messages=messages, tools=tools, tool_choice="auto")
    print("\nFirst LLM Response:\n", response)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {"get_current_weather": get_current_weather}
        messages.append(response_message)  # extend conversation with assistant's reply

        for tool_call in tool_calls:
            function_args = json.loads(tool_call.function.arguments)
            function_response = available_functions[tool_call.function.name](
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append({"tool_call_id":tool_call.id, "role": "tool", "name": tool_call.function.name, "content": function_response})
        second_response = litellm.completion(model="gpt-3.5-turbo-1106", messages=messages)
        print("\nSecond LLM response:\n", second_response.choices[0].message.content)
