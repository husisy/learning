# caddy

TLS termiantion proxy

1. link
   * [website](https://caddyserver.com/)
   * [github](https://github.com/caddyserver/caddy)
2. install
   * ubuntu-api [caddy-link](https://caddyserver.com/docs/install#debian-ubuntu-raspbian)
3. preferences
   * use `Caddyfile`, and then convert to `JSON` config `caddy adapt --config /path/to/Caddyfile`
   * do **NOT** use CLI interface if possible
4. caddy cloudflare
   * [link](https://caddy.community/t/how-to-install-additional-packages-with-apt/11052/5)
   * [github/caddy-cloudflare](https://github.com/caddy-dns/cloudflare)
   * install caddy with plugin, directly replace `/usr/bin/caddy`
   * cloudflare: only first-level DNS is free `*.example.com`, second-level DNS is not free `*.sub.example.com`
   * cloudflare: require `edit` token, `read` token is not enough

ubuntu service

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

sudo systemctl status caddy
sudo systemctl restart caddy
sudo systemctl edit caddy #add environment variable (see below)
sudo journalctl -u caddy --no-pager | less +G #view logging
/etc/caddy/Caddyfile #config file
/var/lib/caddy #home directory
/var/lib/caddy/.local/share/caddy #data storage
```

```text
[Service]
Environment="CF_API_TOKEN=super-secret-cloudflare-tokenvalue"
# {env.CF_API_TOKEN}
```

```bash
# use systemctl instead
caddy
caddy run
caddy start
caddy upgrade

curl localhost:2019/config/

caddy adapt --config /path/to/Caddyfile
caddy adapt --config /path/to/Caddyfile --pretty
# sudo apt install jq
caddy adapt --config /path/to/Caddyfile | jq
caddy adapt --config /path/to/Caddyfile > caddy.json
curl localhost:2019/load -H "Content-Type: application/json" -d @caddy.json
curl localhost:2015

curl localhost:2019/config/apps/http/servers/srv0/routes/0/handle/0/body -H "Content-Type: application/json" -d '"hello again."'
curl localhost:2015
```

`Caddyfile`

```text
:2015

respond "Hello, world!"
```

for debug purpose [documentation/log](https://caddyserver.com/docs/caddyfile/directives/log)

```text
log {
   format console
   level  INFO
}
```

## cloudflare Warp

1. link
   * cloudflare package [link](https://pkg.cloudflareclient.com/)
   * [documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-devices/warp/)

```bash
curl https://pkg.cloudflareclient.com/pubkey.gpg | sudo gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg
# Add this repo to your apt repositories
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflare-client.list
sudo apt-get update && sudo apt-get install cloudflare-warp

curl https://ipinfo.io/ip

warp-cli registration new
warp-cli mode proxy
warp-cli connect # set-mode proxy first!!!
curl --proxy socks://127.0.0.1:40000/ https://ipinfo.io/ip
# curl https://www.cloudflare.com/cdn-cgi/trace/
```
