from aiohttp import web

from routes import setup_routes
from settings import config

app = web.Application()
setup_routes(app)
app['config'] = config
web.run_app(app)
