import datetime

import litellm


### set <pre> <suf> to indicate the string before and after the part you want codellama to fill
prompt_prefix = 'def remove_non_ascii(s: str) -> str:\n    """ '
prompt_suffix = "\n    return result"


messages = [{"content": f"<PRE> {prompt_prefix} <SUF>{prompt_suffix} <MID>", "role": "user"}]
response = litellm.completion(model='ollama/codellama:7b-instruct', temperature=0, messages=messages, max_tokens=192)
response = litellm.completion(model='ollama/codellama:7b-code', temperature=0, messages=messages, max_tokens=192)
print(response.choices[0].message.content)

response = litellm.completion(model='ollama/codellama', messages=messages, max_tokens=500)

prefix = 'def compute_gcd(x, y):\n    '
suffix = '\n    return gcd'
messages = [{"content": f'<PRE> {prefix} <SUF>{suffix} <MID>', "role": "user"}]
response = litellm.completion(model='ollama/codellama:7b-code', temperature=0, messages=messages, max_tokens=192)
tmp0 = response.choices[0].message.content.rstrip()
if tmp0.endswith('<EOT>'):
    tmp0 = tmp0[:-5]
    print(tmp0)
    print('---')
    print(prefix + tmp0 + suffix)



## multiple users
name_role_list = [
    ('Bob', 'project manager'),
    ('Alice', 'software developer'),
    ('Charlie', 'software tester'),
]
_SYSTEM_PROMPT = ('This is a group conversation to setup a projection on library management software developing. '
                  'You are going to be {name}, a {role}. '
                  'Other members are {other_names}. In this conversation, you will discuss how to develop this software. '
                  'Please be concise and clear. '
                  'every message should be no more than 100 words. ')
name_to_sysmtem_prompt = dict()
for name,role in name_role_list:
    tmp0 = ', '.join([f'({x}, {y})' for x,y in name_role_list if x!=name])
    tmp1 = _SYSTEM_PROMPT.format(name=name, role=role, other_names=tmp0)
    name_to_sysmtem_prompt[name] = dict(role='system', content=tmp1)

message_list = []
for name,role in name_role_list:
    tmp0 = f'Hi, I am {name}, a {role} in this project'
    message_list.append(dict(role=name, content=tmp0))
    print(f'{name} ({role}):\n{tmp0}\n')
for _ in range(3):
    for name,role in name_role_list:
        role_message = [name_to_sysmtem_prompt[name]]
        for x in message_list:
            if x['role']==name:
                role_message.append(dict(role='assistant', content=x['content']))
            else:
                tmp1 = x['role'] + ': ' + x['content']
                role_message.append(dict(role='user', content=tmp1))
        response = litellm.completion(messages=role_message, model="gpt-3.5-turbo-1106")
        # response = litellm.completion(messages=role_message, model="ollama/llama2", temperature=0.1, api_base="http://localhost:11434")
        tmp3 = response.choices[0].message.content
        tmp4 = datetime.datetime.now().strftime("%H:%M:%S")
        print(f'[{tmp4}] {name} ({role}):\n{tmp3}\n')
        message_list.append(dict(content=tmp3, role=name))

# litellm.token_counter(model='gpt-3.5-turbo-1106', messges=role_message)
