import io
import os
import json
import base64
import requests
import PIL.Image
import numpy as np

BASE_URL = "http://localhost:11434"

## show model information
question = dict(name='llama2')
response_json = requests.post(f'{BASE_URL}/api/show', json=dict(name='llama2')).json()
# license modelfile parameters template details


## stream=False
question = dict(model='llama2', prompt='What color is the sky at different times of the day? Respond using JSON', format='json', stream=False)
response_json = requests.post(f'{BASE_URL}/api/generate', json=question).json()
message = response_json['response']
json.loads(message) #should be good
response_json['prompt_eval_count'] #number of tokens in the prompt
response_json['eval_count'] #number of tokens the response
response_json['context'] #(list,int)
response_json['eval_count'] / response_json['eval_duration'] * 1e9 #tokens per second

## stream=True (default)
question = dict(model='llama2', prompt='What color is the sky at different times of the day? Respond using JSON', format='json', stream=True)
response = requests.post(f'{BASE_URL}/api/generate', json=question)
z0 = [json.loads(x) for x in response.content.split(b'\n') if x]
message = ''.join(x['response'] for x in z0[:-1])
json.loads(message) #should be good
status = z0[-1]


## https://github.githubassets.com/assets/github-mark-c791e9551fe4.zip
buffered = io.BytesIO()
PIL.Image.open('data/github-mark/github-mark-white.png').convert('RGB').save(buffered, format="JPEG")
image_str = base64.b64encode(buffered.getvalue()).decode('ascii')
question = dict(model='llava', prompt='What is in this picture?', images=[image_str], stream=False)
response_json = requests.post(f'{BASE_URL}/api/generate', json=question).json()
message = response_json['response']


## raw mode

## full options (fail)
tmp0 = dict(num_keep=5, seed=43, num_predict=100, top_k=20, top_p=0.9, tfs_z=0.5, typical_p=0.7,
    repeat_last_n=33, temperature=0.8, repeat_penalty=1.2, presence_penalty=1.5, frequency_penalty=1.0,
    mirostat=1, mirostat_tau=0.8, mirostat_eta=0.6, penalize_newline=True, stop=["\n", "user:"],
    numa=False, num_ctx=1024, num_batch=2, num_gqa=1, low_vram=False, f16_kv=True,
    vocab_only=False, use_mmap=True, use_mlock=False, embedding_only=False, rope_frequency_base=1.1,
    rope_frequency_scale=0.8, num_thread=8) #, num_gpu=0, main_gpu=0
question = dict(model='llama2', prompt='Why is the sky blue?', stream=False, options=tmp0)
response_json = requests.post(f'{BASE_URL}/api/generate', json=question).json()
message = response_json['response']

## chat
message_list = [dict(role='user', content='Why is the sky blue?')]
question = dict(model='llama2', messages=message_list, stream=False)
response_json = requests.post(f'{BASE_URL}/api/chat', json=question).json()
message = response_json['message']

message_list = [
    dict(role='user', content='Why is the sky blue?'),
    dict(role='assistant', content='Due to Rayleigh scattering.'),
    dict(role='user', content='How is that different than Mie scattering?'),
]
question = dict(model='llama2', messages=message_list, stream=False)
response_json = requests.post(f'{BASE_URL}/api/chat', json=question).json()
message = response_json['message']


## chat with image
buffered = io.BytesIO()
PIL.Image.open('data/github-mark/github-mark-white.png').convert('RGB').save(buffered, format="JPEG")
image_str = base64.b64encode(buffered.getvalue()).decode('ascii')
message_list = [dict(role='user', content='what is in this image?', images=[image_str])]
question = dict(model='llava', messages=message_list, stream=False)
response_json = requests.post(f'{BASE_URL}/api/chat', json=question).json()
message = response_json['message']


## embedding
question = dict(model='llama2', prompt='Here is an article about llamas...')
response_json = requests.post(f'{BASE_URL}/api/embeddings', json=question).json()
np0 = np.array(response_json['embedding']) #(np,float64,4096)


## codellama
tmp0 = 'You are an expert programmer that writes simple, concise code and explanations. Write a python function to generate the nth fibonacci number.'
question = dict(model='codellama:7b-instruct', prompt=tmp0, stream=False)
response_json = requests.post(f'{BASE_URL}/api/generate', json=question).json()
message = response_json['response']

tmp0 = '''
Where is the bug in this code?

def fib(n):
    if n <= 0:
        return n
    else:
        return fib(n-1) + fib(n-2)
'''
question = dict(model='codellama:7b-instruct', prompt=tmp0, stream=False)
response_json = requests.post(f'{BASE_URL}/api/generate', json=question).json()
message = response_json['response']

tmp0 = '# A simple python function to remove whitespace from a string:'
question = dict(model='codellama:7b-code', prompt=tmp0, stream=False)
response_json = requests.post(f'{BASE_URL}/api/generate', json=question).json()
message = response_json['response']


pre_text = 'def compute_gcd(x, y):'
suf_text = 'return result'
question = dict(model='codellama:7b-code', prompt=f'<PRE> {pre_text} <SUF>{suf_text} <MID>', stream=False)
response_json = requests.post(f'{BASE_URL}/api/generate', json=question).json()
message = response_json['response']
# <EOT>
