import flask

app = flask.Flask(__name__)

@app.route('/')
def index():
    print(flask.request.method) #'GET'
    return 'my index page'
#localhost:5000/

@app.route('/hello')
def hello():
    return 'my hello world'
#localhost:5000/hello

@app.route('/user/<username>')
def show_user_profile(username):
    return 'user: {}'.format(username)
#locahost:5000/user/233

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

