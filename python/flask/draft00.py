import flask
import markupsafe

app = flask.Flask(__name__)

@app.route('/')
def index():
    print(flask.request.method) #'GET'
    return 'my index page'
#localhost:5000/
# localhost:5000/index

@app.route('/hello')
def hello():
    return 'my hello world'
#localhost:5000/hello

@app.route('/user/<name>')
def show_user_profile(name):
    ret = f'user: {markupsafe.escape(name)}'
    # ret = 'user: {}'.format(markupsafe.escape(name))
    return ret
#locahost:5000/user/233
# Jinja will do markupsafe.escape automatically

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return 'post: {}'.format(post_id)
#localhost:5000/post/233

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    return 'subpath: {}'.format(subpath)
#localhost:5000/path/what/ya

@app.route('/userid/<user>_<int:id>')
def show_user_id(user, id_):
    return 'user: {};  id: {}'.format(user, id_)
#locahost:5000/userid/husisy_233

@app.route('/projects')
def projects():
    return 'my projects page'
#localhost:5000/projects/
#localhost:5000/projects

# do not use this
# @app.route('/about/')
# def about():
#     return 'my about page'
#localhost:5000/about
#FAIL:: localhost:5000/about/

# http://127.0.0.1:5000/

# linux: export FLASK_APP=draft00.py
# win-cmd: set FLASK_APP=draft00.py
# powershell: $env:FLASK_APP = "draft00.py"

# linux: EXPORT FLASK_ENV=development
# win-cmd: set FLASK_ENV=development

# python -m flask run
# python -m flask run --host=0.0.0.0 --port=3307


with app.test_request_context():
    # input name of function, return url
    print(flask.url_for('index')) #/
    print(flask.url_for('show_user_profile', name='world')) #/user/world

with app.test_request_context('/hello', method='GET'):
    assert flask.request.path == '/hello'
    assert flask.request.method == 'GET'
