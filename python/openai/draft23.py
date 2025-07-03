import json
import os
import dotenv
import httpx
import openai

OPENAI_API_KEY = dotenv.dotenv_values('.env')['OPENAI_API_KEY']

# client = openai.OpenAI(api_key=OPENAI_API_KEY)
client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(proxy='socks5://localhost:9800'))

generate_response = lambda m: client.responses.create(model="gpt-4.1", input=m, max_output_tokens=1024).output_text


def extract_markdown_block(response: str, block_type: str = "json") -> str:
    """Extract code block from response"""
    if not '```' in response:
        ret = response
    else:
        ret = response.split('```')[1].strip()
        if ret.startswith(block_type):
            ret = ret[len(block_type):].strip()
    return ret


def parse_action(response: str) -> dict:
    """Parse the LLM response into a structured action dictionary."""
    try:
        response_json = json.loads(extract_markdown_block(response, "action"))
        if ("tool_name" in response_json) and ("args" in response_json):
            return response_json
        else:
            return {"tool_name": "error", "args": {"message": "You must respond with a JSON tool invocation."}}
    except json.JSONDecodeError:
        return {"tool_name": "error", "args": {"message": "Invalid JSON response. You must respond with a JSON tool invocation."}}


def list_files() -> list[str]:
    """List files in the current directory."""
    return os.listdir(".")

def read_file(file_name: str) -> str:
    """Read a file's contents."""
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except Exception as e:
        return f"Error: {str(e)}"

# Define system instructions (Agent Rules)
agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools.

Available tools:

```json
{
    "list_files": {
        "description": "Lists all files in the current directory.",
        "parameters": {}
    },
    "read_file": {
        "description": "Reads the content of a file.",
        "parameters": {
            "file_name": {
                "type": "string",
                "description": "The name of the file to read."
            }
        }
    },
    "terminate": {
        "description": "Ends the agent loop and provides a summary of the task.",
        "parameters": {
            "message": {
                "type": "string",
                "description": "Summary message to return to the user."
            }
        }
    }
}
```

If a user asks about files, documents, or content, first list the files before reading them.

When you are done, terminate the conversation by using the "terminate" tool and I will provide the results to the user.

Important!!! Every response MUST have an action.
You must ALWAYS respond in this format:

<Stop and think step by step. Parameters map to args. Insert a rich description of your step by step thoughts here.>

```action
{
    "tool_name": "insert tool_name",
    "args": {...fill in any required arguments here...}
}
```"""
}]

# Initialize agent parameters
iterations = 0
max_iterations = 10

user_task = input("What would you like me to do? ")
# tell me what files are in this directory
# summarize contents in file README.md

memory = [{"role": "user", "content": user_task}]

# The Agent Loop
while iterations < max_iterations:
    # 1. Construct prompt: Combine agent rules with memory
    prompt = agent_rules + memory

    # 2. Generate response from LLM
    print("Agent thinking...")
    response = generate_response(prompt)
    print(f"Agent response: {response}")

    # 3. Parse response to determine action
    action = parse_action(response)
    result = "Action executed"

    if action["tool_name"] == "list_files":
        result = {"result": list_files()}
    elif action["tool_name"] == "read_file":
        result = {"result": read_file(action["args"]["file_name"])}
    elif action["tool_name"] == "error":
        result = {"error": action["args"]["message"]}
    elif action["tool_name"] == "terminate":
        print(action["args"]["message"])
        break
    else:
        result = {"error": "Unknown action: " + action["tool_name"]}

    print(f"Action result: {result}")

    # 5. Update memory with response and results
    memory.extend([
        {"role": "assistant", "content": response},
        {"role": "user", "content": json.dumps(result)}
    ])

    # 6. Check termination condition
    if action["tool_name"] == "terminate":
        break

    iterations += 1
