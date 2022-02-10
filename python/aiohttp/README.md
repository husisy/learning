# aiohttp

1. link
   * [documentation](https://aiohttp.readthedocs.io/en/stable/)
   * [github](https://github.com/aio-libs/aiohttp/)
2. `conda install -c conda-forge aiohttp cchardet aiodns`
3. 配置文件
   * 推荐`yaml`配置文件
   * 建议从如下目录加载配置文件`./config/app_cfg.yaml`, `/etc/app_cfg.yaml`
   * 建议保留命令行覆盖配置文件的能力
   * 建议严格的配置文件校验，如[Trafaret](https://github.com/Deepwalker/trafaret), [colander](https://github.com/Pylons/colander), [jsonschema](https://github.com/Julian/jsonschema)

## ws00

1. 在`ws00/`目录下执行`python aiohttpdemo_polls/main.py`
2. 在浏览器中访问`0.0.0.0:8080`
