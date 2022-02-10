# link

1. link
   * [github](https://github.com/CharlesPikachu/DecryptLogin)
   * [documentation](https://httpsgithubcomcharlespikachudecryptlogin.readthedocs.io/zh/latest/index.html)
2. install `pip install DecryptLogin`

```Python
from DecryptLogin import login

lg = login.Login()
infos_return, session = lg.github(username, password, 'pc')
```
