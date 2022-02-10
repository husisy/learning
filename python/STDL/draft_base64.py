import base64

x1 = base64.b64encode(b'binary\x00string')
x2 = base64.b64decode(x1)

x1 = base64.b64encode(b'i\xb7\x1d\xfb\xef\xff')
x2 = base64.urlsafe_b64encode(b'i\xb7\x1d\xfb\xef\xff')
base64.urlsafe_b64decode(x2)
