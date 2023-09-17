# dotenv

1. link
   * [github](https://github.com/theskumar/python-dotenv)
   * [documentation](https://saurabh-kumar.com/python-dotenv/)
2. install
   * `pip install python-dotenv`
   * `conda install -c conda-forge python-dotenv`
3. (default) not override existing environment variables

```bash
# Development settings
DOMAIN=example.org
ADMIN_EMAIL=admin@${DOMAIN}
ROOT_URL=${DOMAIN}/app
```

```Python
import os
import dotenv
dotenv.load_dotenv() #default to .env
os.environ['XXX']
os.getenv('XXX')

dotenv.dotenv_values('.env')['xxx'] #not override existing environment variables
```
