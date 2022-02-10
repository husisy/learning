import time
from datetime import datetime
import feedparser
from lxml import etree

def get_first(x:list, default=None):
    assert isinstance(x, list)
    if len(x)==0:
        return default
    else:
        return x[0]

hf_clean_html = lambda x: '\n'.join(etree.HTML(x).xpath('.//text()'))
hf_img_src = lambda x: get_first(etree.HTML(x).xpath('.//img/@src'))

# https://journals.aps.org/feeds
url = 'http://feeds.aps.org/rss/recent/physics.xml'

z0 = feedparser.parse(url)

[x['link'] for x in z0['entries']] #http://link.aps.org/doi/10.1103/Physics.12.92
[x['title'] for x in z0['entries']] #Focus: Friction, Not Inertia, Controls Avalanches
[x['summary'] for x in z0['entries']]
[x.get('author') for x in z0['entries']] #could be none
[datetime.fromtimestamp(time.mktime(x['updated_parsed'])) for x in z0['entries']]
[hf_img_src(x['summary']) for x in z0['entries']][:10]

ret = []
feed_title = z0['feed']['title']
feed_url = url
for x in z0['entries']:
    ret_i = {
        'feed_title': feed_title,
        'feed_url': feed_url,
        'title': x['title'],
        'link': x['link'], #UID
        'summary_raw': x['summary'],
        'summary': hf_clean_html(x['summary']),
        'img_src': hf_img_src(x['summary']),
        'updated_time': datetime.fromtimestamp(time.mktime(x['updated_parsed'])),
    }
    if 'author' in x:
        ret_i['author'] = x['author']
    ret.append(ret_i)
