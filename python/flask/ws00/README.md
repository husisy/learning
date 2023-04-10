# flaskr

1. link
   * [flask/tutorial](https://flask.palletsprojects.com/en/2.2.x/tutorial/)
   * [github/tutorial](https://github.com/pallets/flask/tree/2.2.3/examples/tutorial)
2. code
   * `404` not found
   * `403` forbidden
   * `401` unauthorized
3. `waitress-serve --listen="127.0.0.1:5000" --call "flaskr:create_app"`

```bash
flask --app flaskr init-db
flask --app flaskr run
waitress-serve --listen="127.0.0.1:5000" --call "flaskr:create_app"
```

```text
/home/user/Projects/flask-tutorial
├── flaskr/
│   ├── __init__.py
│   ├── db.py
│   ├── schema.sql
│   ├── auth.py
│   ├── blog.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── auth/
│   │   │   ├── login.html
│   │   │   └── register.html
│   │   └── blog/
│   │       ├── create.html
│   │       ├── index.html
│   │       └── update.html
│   └── static/
│       └── style.css
└── tests/
    ├── conftest.py
    ├── data.sql
    ├── test_factory.py
    ├── test_db.py
    ├── test_auth.py
    └── test_blog.py
```

`.gitignore`

```text
venv/

*.pyc
__pycache__/

instance/

.pytest_cache/
.coverage
htmlcov/

dist/
build/
*.egg-info/
```
