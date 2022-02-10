import re
import requests

login_headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Length': '196',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Host': 'github.com',
    'Origin': 'https://github.com',
    'Pragma': 'no-cache',
    'Referer': 'https://github.com/login',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
}
login_url = 'https://github.com/login'
post_url = 'https://github.com/session'

username = '233'
password = '233'
session = requests.Session()
response = session.get(login_url)
token = re.findall(r'authenticity_token.*?value="(.*?)"', response.text)[0]
data = {
    'commit': 'Sign in',
    'utf8': '✓',
    'authenticity_token': token,
    'login': username,
    'password': password
}
response = session.post(post_url, headers=login_headers, data=data)
