from lxml import etree


# basic
root = etree.Element('root')
root.append(etree.Element('child0'))
child2 = etree.SubElement(root, 'child2')
root.insert(1, etree.Element('child1'))

etree.tostring(root).decode()
etree.tostring(root, pretty_print=True).decode()

root.tag
child2.tag

# list like property
len(root)
child0 = root[0]
child1 = root[1]
root[:1]
root[1:]

# basic
z0 = etree.Element('root', prop1='prop1')
z0.append(etree.Element('child1'))
z1 = etree.SubElement(z0, 'child2')
etree.tostring(z0).decode()
etree.tostring(z0, pretty_print=True).decode()

z0.tag
len(z0)
z0[0]
etree.iselement(z0)
z0[0].getparent().tag
z0[1].getprevious().tag
z0[0].getnext().tag

z0.get('prop1')
z0.set('prop1', '233')
z0.keys()
z0.items()
z0.attrib

z0.text = '233'


# copy
etree.Element('x1') == etree.Element('x1') #False
tmp1 = etree.Element('x1')
tmp1 == tmp1 #True
tmp1 is tmp1 #True
# from copy import deepcopy


# mixed-context XML
z0 = etree.fromstring('<html><body>Hello<br/>World</body></html>')

html = etree.Element('html')
body = etree.SubElement(html, 'body')
body.text = 'Hello'
br = etree.SubElement(body, "br")
br.tail = 'World'


# XPath
z0 = etree.fromstring('<html><body>Hello<br/>World<br/>!</body></html>')
z0.xpath('string()')
z0.xpath('//text()')
z1 = etree.XPath('//text()')
z2 = z1(z0)
z2[0].getparent().tag
z2[1].is_tail
z2[0].is_text

z0 = etree.fromstring('<root><child>Child 1</child><child>Child 2</child><another>Child 3</another></root>')
[x.tag for x in z0.iter()]
list(z0.iter('child'))


# xpath
tmp1 = '''<bookstore>

<book>
  <title lang="eng">Harry Potter</title>
  <price>29.99</price>
</book>

<book>
  <title lang="eng">Learning XML</title>
  <price>39.95</price>
</book>

</bookstore>'''
z1 = etree.fromstring(tmp1)

# try to get all book-node
z1.xpath('book')
z1.xpath('./book')
z1.xpath('/bookstore/book')
z1.xpath('//book')
z1[0].xpath('../book')
z1[0].xpath('//book')
z1[0].xpath('/bookstore/book')
z1[0].xpath('bookstore//book')

z1.xpath('//@lang')

z1.xpath('/bookstore/book[1]')
z1.xpath('/bookstore/book[2]')
z1.xpath('/bookstore/book[last()]')
z1.xpath('/bookstore/book[last()-1]')
z1.xpath('/bookstore/book[position()<3]')

z1.xpath('//title[@lang]')
z1.xpath('//title[@lang="eng"]')
z1.xpath('//book[price>35.0]')
z1.xpath('//book[title="Learning XML"]/title')

z1.xpath('/bookstore/*')
z1.xpath('//*')
z1.xpath('//title[@*]')

z1.xpath('//title | //price')

z1.xpath('child::book')
z1.xpath('child::*[price>35]')

z1.xpath('book/title/attribute::*')

z1.xpath('//title/@text()')