import re
import regex
import string

# metacharacter
r'.^$*+?{}[]\|()'
regex.search(r'\.\^\$\*\+\?\{\}\[\]\\\|\(\)', r'233.^$*+?{}[]\|()233')

regex.match(r'\n','\n')
regex.match('\n','\n')

regex.findall(r'\w', string.printable) #\W
regex.findall(r'\d', string.printable) #\D
regex.findall(r'\s', string.whitespace)
regex.match('.', '\n')
regex.match('.', '\n', re.DOTALL)

regex.findall('233', '23 233 2333')
regex.findall('233[a-z]', '233a 233b 233A 233Z')
regex.findall('233[^a-z]', '233a 233b 233A 233Z')
regex.findall('233[$.]', '233a 233b 233$ 233.')
regex.findall(r'[\\\]]','\\]')

re.search('^233', '233')
re.search('^233', 'xx233')
re.findall('^233', '233\n233')
re.findall('^233', '233\n233', re.MULTILINE)

# TODO, below wait for check
