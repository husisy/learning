import os
import dotenv
import requests

dotenv.load_dotenv()

BAIDU_API_KEY = os.environ['BAIDU_API_KEY']
BAIDU_SECRET_KEY = os.environ['BAIDU_SECRET_KEY']

# access token
api_url = 'https://aip.baidubce.com/oauth/2.0/token'
query_param = {
    'grant_type': 'client_credentials',
    'client_id': BAIDU_API_KEY,
    'client_secret': BAIDU_SECRET_KEY,
}
headers = {'Content-Type': 'application/json', 'accept':'application/json'}
response = requests.post(api_url, params=query_param)
access_token = response.json()['access_token']


# ernie-bot-turbo
api_url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant'
headers = {'Content-Type': 'application/json', 'accept':'application/json'}
data = {
    'messages': [
        {'role': 'user', 'content': '你好'},
    ]
}
query_param = {
    'access_token': access_token,
}
response01 = requests.post(api_url, params=query_param, headers=headers, json=data)
response01.json()

data = {
    'messages': [
        {"role":"user","content":"请介绍一下你自己"},
        {"role":"assistant","content":"我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。"},
        {"role":"user","content": "我在上海，周末可以去哪里玩？"},
        {"role":"assistant","content": "上海是一个充满活力和文化氛围的城市，有很多适合周末游玩的地方。以下是几个值得推荐的地方：\n\n1. 上海科技馆：上海科技馆是中国大陆最大的科技馆之一，有许多互动展览和科学实验室，可以让您了解科技的发展历史和最新成果。\n2. 上海博物馆：上海博物馆是一座集艺术、历史和文化于一体的博物馆，有许多精美的艺术品和文物，可以让您深入了解中国和世界的文化历史。\n3. 外滩：外滩是上海的标志性景点之一，有许多高楼大厦和繁华的商业街区，是欣赏黄浦江美景和城市风貌的好去处。\n4. 上海迪士尼乐园：上海迪士尼乐园是世界上最大的迪士尼主题公园之一，有许多精彩的游乐项目和演出，可以让您尽情享受娱乐和欢乐。\n5. 上海野生动物园：上海野生动物园是一个以自然保护为主题的野生动物园，有许多珍稀动物和植物，可以让您近距离接触大自然。\n\n这些地方都是上海周末游玩的好去处，可以根据自己的兴趣和需求选择合适的行程。"},
        {"role":"user","content": "周末这里的天气怎么样？"},
    ]
}
response02 = requests.post(api_url, params=query_param, headers=headers, json=data)
response02.json()
# {
#     'id': 'xxx',
#     'object': 'chat.completion',
#     'created': 1693978987,
#     'result': '很抱歉，作为一个AI语言模型，我无法获取实时的地理位置信息。您可以通过查看当地的天气预报或应用程序来获取准确的天气信息。',
#     'is_truncated': False,
#     'need_clear_history': False,
#     'usage': {'prompt_tokens': 470, 'completion_tokens': 55, 'total_tokens': 525}
# }
