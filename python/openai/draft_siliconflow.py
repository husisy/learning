import openai
import dotenv

ENV_CONFIG = dotenv.dotenv_values('.env') #not sync in .git
API_KEY = ENV_CONFIG['SILICONFLOW_API']
BASE_URL = 'https://api.siliconflow.cn/v1'


client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
response = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-V3',
    messages=[
        {'role': 'user',
        'content': "中国大模型行业2025年将会迎来哪些机遇和挑战"}
    ],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end='')

# to use in open-webui
# export OPENAI_API_BASE_URL="https://api.siliconflow.cn/v1"
# export OPENAI_API_KEY="xxx"
