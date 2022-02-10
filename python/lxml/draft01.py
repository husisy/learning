import os
from lxml import etree
from io import StringIO, BytesIO

hf_data = lambda *x: os.path.join('data', *x)
hf_html = lambda x: '<!DOCTYPE html><html><body>' + x + '</body></html>'

# XML
_ = etree.fromstring('<img />')
_ = etree.parse(hf_data('test01.xml'))


# HTML
html = hf_html('<img src="http://zhongyisousuo.com/s?q=中药大全&cs=utf-8" />')

z1 = etree.HTML(html)
print(etree.tostring(z1, pretty_print=True, method='html').decode('utf-8'))

parser = etree.HTMLParser()
z1 = etree.parse(StringIO(html), parser).getroot()
z1.xpath('//img/@src')


# DOCTYPE
pub_id  = "-//W3C//DTD XHTML 1.0 Transitional//EN"
sys_url = "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"
doctype_string = '<!DOCTYPE html PUBLIC "%s" "%s">' % (pub_id, sys_url)
xml_header = '<?xml version="1.0" encoding="utf-8"?>'
xhtml = xml_header + doctype_string + '<html><body></body></html>'
z1 = etree.parse(BytesIO(xhtml.encode('utf-8')))
z1.docinfo.public_id
z1.docinfo.system_url
z1.docinfo.xml_version
z1.docinfo.encoding
