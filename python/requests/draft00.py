import os
import requests
import webbrowser
from tqdm import tqdm

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())
my_headers = {
    'user-agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'),
}
# "navigator.userAgent" in javascripts

def browser_html(text):
    filepath = hf_file('tbd00.html')
    with open(filepath, 'w') as fid:
        fid.write(text)
    webbrowser.open('file://' + os.path.realpath(filepath))

x0 = requests.get('https://httpbin.org/get') #'https://api.github.com/events'
x0.content
x0.text
x0.url
x0.encoding #x0.encoding='gb2312'
x0.headers
x0.headers['content-type']
x0.json()
x0.status_code
# https://httpbin.org/status/404 https://httpbin.org/status/201
x0.raise_for_status() #x0.status_code==requests.codes.ok

x0 = requests.get('https://httpbin.org/get', params={'key1':'value1', 'key2':'value2'})
x1 = requests.get('https://httpbin.org/get', params={'key1':'value1', 'key2':['value2','value3']})
x0.url

requests.post('https://httpbin.org/post', data={'key':'value'})
requests.put('https://httpbin.org/put', data={'key':'value'})
requests.delete('https://httpbin.org/delete')
requests.head('https://httpbin.org/get')
requests.options('https://httpbin.org/get')

# TODO can we crawl bing picture every-day
url = 'https://cn.bing.com/th?id=OHR.YukonGames_ZH-CN0135612170_UHD.jpg&pid=hp&w=2880&h=1620&rs=1&c=4&r=0'
z0 = requests.get(url, headers=my_headers)
with open(hf_file('tbd00.jpg'), 'wb') as fid:
    fid.write(z0.content)


def download_url_and_save(url, directory='.', headers=None, proxies=None):
    assert os.path.exists(directory)
    response = requests.get(url, headers=headers, proxies=proxies, stream=True)
    response.raise_for_status()
    filename = os.path.join(directory, url.rsplit('/',1)[1])
    if not os.path.exists(filename):
        tmp_filename = filename + '.incomplete'
        tmp0 = {'total':int(response.headers['content-length']), 'unit':'iB', 'unit_scale':True}
        with open(tmp_filename, 'wb') as fid, tqdm(**tmp0) as progress_bar:
            for x in response.iter_content(chunk_size=1024): #1kiB
                progress_bar.update(len(x))
                fid.write(x)
        os.rename(tmp_filename, filename)
    return filename


def download_random_picture(directory='.', headers=None, proxies=None):
    # https://wallpaper.wispx.cn/random
    response = requests.get('https://wallpaper.wispx.cn/api/find?rand=1', headers=headers)
    response.raise_for_status()
    all_info = response.json()
    filename = download_url_and_save(all_info['download_url'], directory, headers, proxies)
    return all_info, filename


# post file
with open('post00.txt', encoding='utf-8') as fid:
    requests.post('https://httpbin.org/post', files={'file':fid})
requests.post('https://httpbin.org/post', files={'file':('post00.txt','hello world.', {'Expires':'0'})})

x1 = requests.get('https://httpbin.org/cookies', cookies={'cookies_are':'working'})
x1.text

tmp1 = requests.cookies.RequestsCookieJar()
tmp1.set('tasty_cookie', 'yum', domain='httpbin.org', path='/cookies')
tmp1.set('gross_cookie', 'blech', domain='httpbin.org', path='/elsewhere')
x1 = requests.get('https://httpbin.org/cookies', cookies=tmp1)
x1.text

x1 = requests.get('http://github.com/')
x1.url
x1.status_code
x1.history

x1 = requests.get('http://github.com/', allow_redirects=False)
x1.status_code
x1.history

x1 = requests.head('http://github.com/', allow_redirects=True)
x1.url
x1.history

x1 = requests.get('https://github.com/', timeout=0.001)


with requests.Session() as sess:
    x1 = sess.get('https://httpbin.org/cookies/set/sessioncookie/23333333')
    x2 = sess.get('https://httpbin.org/cookies')
x1.text
x2.text

with requests.Session() as sess:
    sess.auth = ('user', 'pass')
    sess.headers.update({'x-test':'true'})
    x1 = sess.get('https://httpbin.org/headers', headers={'x-test2':'true'})
    x2 = sess.get('https://httpbin.org/headers', headers={'x-test3':'true'})
x1.text
x2.text

x1 = requests.get('https://en.wikipedia.org/wiki/Monty_Python')
x1.headers
x1.request.headers


def get_public_ip(ipv6:bool=False):
    # https://www.ipify.org/
    # https://stackoverflow.com/a/3097641/7290857
    url = 'https://api64.ipify.org' if ipv6 else 'https://api.ipify.org'
    ret = requests.get(url, params={'format':'json'}).json()['ip']
    # ret = requests.get(url).text
    return ret
print(get_public_ip())
print(get_public_ip(ipv6=True))
