import re
import string


string.ascii_letters
string.ascii_lowercase
string.ascii_uppercase
string.digits
string.whitespace
string.printable
string.punctuation

# metacharacter
r'.^$*+?{}[]\|()'
re.search(r'\.\^\$\*\+\?\{\}\[\]\\\|\(\)', r'233.^$*+?{}[]\|()233')

re.match(r'\n','\n')
re.match('\n','\n')

re.findall(r'\w', string.printable) #\W
re.findall(r'\d', string.printable) #\D
re.findall(r'\s', string.whitespace)
re.match('.', '\n')
re.match('.', '\n', re.DOTALL)

re.findall('233', '23 233 2333')
re.findall('233[a-z]', '233a 233b 233A 233Z')
re.findall('233[^a-z]', '233a 233b 233A 233Z')
re.findall('233[$.]', '233a 233b 233$ 233.')
re.findall(r'[\\\]]','\\]')

re.search('^233', '233')
re.search('^233', 'xx233')
re.findall('^233', '233\n233')
re.findall('^233', '233\n233', re.MULTILINE)

re.search('233$','xx 233')
re.search('233$','xx 233\n')
re.findall('233$', 'xx 233\nxx 233\n')
re.findall('233$', 'xx 233\nxx 233\n', re.MULTILINE)

re.search('[^^]', '^')
re.search('[[]','233') #future warning

re.search('233|23','233')
re.search('23|233','233')

re.search('2(3(3))','233').groups()
re.search('(?:233)','233').groups()

re.search('(?P<xx>233)','233').group('xx')
re.search('(?P<xx>xx)233(?P=xx)', 'xx233xx')
re.sub('(?P<xx>xx)233(?P=xx)', r'\g<1>332\g<1>', 'xx233xx')
re.sub('(?P<xx>xx)233(?P=xx)', r'\g<xx>332\g<xx>', 'xx233xx')
re.findall('[abc]233(?=3)', 'a2333 b2334 c233')
re.findall('[abc]233(?!3)', 'a2333 b2334 c233')
re.search('(?<=a)233[abc]', 'a233a b233b c233c')
re.findall('(?<!a)233[abc]', 'a233a b233b c233c')

re.findall('23*4', '24 234 2334 23334')
re.findall('23+4', '24 234 2334 23334')
re.findall('23?4', '24 234 2334 23334')
re.findall('23{1,2}4', '24 234 2334 23334')
re.findall('23{1,}4', '24 234 2334 23334')
re.findall('23{,2}4', '24 234 2334 23334')
re.findall('23{2}4', '24 234 2334 23334')


re.findall(r'\A233[ab]', '233a 233b')
re.findall(r'\b2[abc]33', '2a33 x2b33 2c33x')
re.findall(r'\B2[abc]33', '2a33 x2b33 2c33x')


hf1 = lambda x: re.search(r'^(<)?(\w+@\w+(?:\.\w+)+)(?(1)>|$)', x)
hf1('<user@host.com>')
hf1('user@host.com')
hf1('<user@host.com')
hf1('user@host.com>')


re.search('23*3','23333').group()
re.search('23*?3','23333').group()
re.search('23+?3','23333').group()
re.search('23??3','23333').group()
re.search('23{1,3}3','23333').group()
re.search('23{1,3}?3','23333').group()

