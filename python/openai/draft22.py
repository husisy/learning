import json
import dotenv
import litellm
# from litellm import completion

dotenv.load_dotenv('.env')
# OPENAI_API_KEY = dotenv.dotenv_values('.env')['OPENAI_API_KEY']

generate_response = lambda m: litellm.completion(model="openai/gpt-4", messages=m, max_tokens=1024).choices[0].message.content

messages = [
    {"role": "system", "content": "You are an expert software engineer that prefers functional programming."},
    {"role": "user", "content": "Write a function to swap the keys and values in a dictionary."}
]
response = generate_response(messages)


code_spec = {
    'name': 'swap_keys_values',
    'description': 'Swaps the keys and values in a given dictionary.',
    'params': {'d': 'A dictionary with unique values.'},
}
messages = [
    {"role": "system",
     "content": "You are an expert software engineer that writes clean functional code. You always document your functions."},
    {"role": "user", "content": f"Please implement: {json.dumps(code_spec)}"}
]
response = generate_response(messages)



def extract_code_block(response:str)->str:
    if not '```' in response:
        ret = response
    else:
        ret = response.split('```')[1].strip()
        if ret.startswith("python"):
            ret = ret[6:]
    return ret


def develop_custom_function(user_message:str):
    messages = [
        {"role": "system", "content": "You are a Python expert helping to develop a function."},
        {"role": "user", "content": f"Write a Python function that {user_message}. Output the function in a ```python code block```."},
    ]
    initial_function = extract_code_block(generate_response(messages))
    print("\n=== Initial Function ===")
    print(initial_function)
    messages.append({"role": "assistant", "content": "```python\n\n"+initial_function+"\n\n```"})

    # Second prompt - Add documentation
    messages.append({
        "role": "user",
        "content": "Add comprehensive documentation to this function, including description, parameters, "
                    "return value, examples, and edge cases. Output the function in a ```python code block```."
    })
    documented_function = extract_code_block(generate_response(messages))
    print("\n=== Documented Function ===")
    print(documented_function)

    # Add documentation response to conversation
    messages.append({"role": "assistant", "content": "```python\n\n"+documented_function+"\n\n```"})
    messages.append({
        "role": "user",
        "content": "Add unittest test cases for this function, including tests for basic functionality, "
                    "edge cases, error cases, and various input scenarios. Output the code in a ```python code block```."
    })
    # We will likely run into random problems here depending on if it outputs JUST the test cases or the
    # test cases AND the code. This is the type of issue we will learn to work through with agents in the course.
    test_cases = extract_code_block(generate_response(messages))
    print("\n=== Test Cases ===")
    print(test_cases)

    # Save final version
    with open('tbd00.py', 'w') as fid:
        fid.write(documented_function + '\n\n' + test_cases)

user_message = 'Swaps the keys and values in a given dictionary.'
user_message = 'A function that calculates the factorial of a number'
user_message = 'Calculate fibonacci sequence up to n'
develop_custom_function(user_message)
