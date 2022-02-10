# download potential
# http://quantum-simulation.org/index.htm


import os
import requests
import webbrowser
from lxml import etree
from tqdm import tqdm


hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())
my_headers = {
    'user-agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'),
}
# "navigator.userAgent" in javascripts

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


def browser_html(text):
    filepath = hf_file('tbd00.html')
    with open(filepath, 'w') as fid:
        fid.write(text)
    webbrowser.open('file://' + os.path.realpath(filepath))


url_dir_list = [
    ('http://quantum-simulation.org/potentials/hscv/xml/', hf_file('hscv','xml')),
    # ('http://quantum-simulation.org/potentials/hscv/output/', hf_file('hscv', 'output')),
]
for url0,dir0 in url_dir_list:
    if not os.path.exists(dir0):
        os.makedirs(dir0)

    z0 = requests.get(url0, headers=my_headers)
    z1 = etree.HTML(z0.text)
    file_list = [str(x) for x in z1.xpath('//tr/td/a/@href') if '.' in x]
    for file_i in file_list:
        filepath = download_url_and_save(url0+file_i, dir0, headers=my_headers)
        print(filepath)
