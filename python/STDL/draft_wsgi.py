from wsgiref.simple_server import make_server

def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    tmp1 = '<h1>hello, {}</h1>'.format(environ['PATH_INFO'][1:] or 'web')
    return [tmp1.encode('utf-8')]

httpd = make_server('', 8000, application)
print('serving HTTP on port 8000, please visit localhost:8000/2333')
httpd.serve_forever()
