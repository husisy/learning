import os
import dotenv
import requests

import cloudflare

dotenv.load_dotenv() #default to .env

# This two environment variable is the default and can be omitted
client = cloudflare.Cloudflare(api_email=os.environ.get("CLOUDFLARE_EMAIL"), api_key=os.environ.get("CLOUDFLARE_API_KEY"))

zone = client.zones.create(
    account={"id": "023e105f4ecef8ad9ca31a8372d0c353"},
    name="example.com",
    type="full",
)
zone.id

# pagination
x0 = list(client.accounts.list())
x1 = list(client.zones.list())
[x.name for x in x1]



def get_public_ip(ipv6:bool=False):
    # https://www.ipify.org/
    # https://stackoverflow.com/a/3097641/7290857
    url = 'https://api64.ipify.org' if ipv6 else 'https://api.ipify.org'
    ret = requests.get(url, params={'format':'json'}).json()['ip']
    # ret = requests.get(url).text
    return ret
print(get_public_ip())
print(get_public_ip(ipv6=True))
