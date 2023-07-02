import lark


def hf_file(x):
    with open(x, 'r', encoding='utf-8') as f:
        ret = f.read()
    return ret


class MyJsonTransformer(lark.Transformer):
    def string(self, s):
        return s[0][1:-1]
    def number(self, n):
        return float(n[0])
    def null(self, _):
        return None
    def true(self, _):
        return True
    def false(self, _):
        return False
    list = list
    pair = tuple
    dict = dict

json_parser = lark.Lark(hf_file('json.lark'), start='value') #parser='lalr'
tmp0 = '{"key": ["item0", "item1", 3.14, true]}'
x0 = json_parser.parse(tmp0)
print(x0.pretty())
'''
dict
  pair
    string      "key"
    list
      string    "item0"
      string    "item1"
      number    3.14
      true
'''
x1 = MyJsonTransformer().transform(x0)


# http://www.json-generator.com/
# https://next.json-generator.com/
