import feedparser
from lxml import etree


hf_clean_html = lambda x: '\n'.join(etree.HTML(x).xpath('.//text()'))

# url = 'http://feeds.aps.org/rss/recent/physics.xml'
url = 'http://feeds.feedburner.com/PythonCentral' #from https://www.pythoncentral.io
# http://feeds.aps.org/rss/recent/physics.xml

z0 = feedparser.parse(url)

z0['headers']
# Last-Modified ETag Content-Type Content-Length Server Connection

z0['feed']
# title subtitle link description sy_updateperiod sy_updatefrequency sy_updatebase
# author publisher updated updated_parsed rights prism_copyright prism_rightsagent

z0['entries'][0]
# id title link summary content authors updated_parsed rights dc_xxx prism_xxx

title = [x['title'] for x in z0['entries']]
link = [x['link'] for x in z0['entries']]
description_raw = [x['description'] for x in z0['entries']]
description = [hf_clean_html(x['description']) for x in z0['entries']]
