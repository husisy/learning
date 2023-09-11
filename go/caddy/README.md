# caddy

TLS termiantion proxy

1. link
   * [website](https://caddyserver.com/)
   * [github](https://github.com/caddyserver/caddy)
2. preferences
   * use `Caddyfile`, and then convert to `JSON` config `caddy adapt --config /path/to/Caddyfile`
   * do **NOT** use CLI interface if possible
3. caddy cloudflare
   * [link](https://caddy.community/t/how-to-install-additional-packages-with-apt/11052/5)
   * [github/caddy-cloudflare](https://github.com/caddy-dns/cloudflare)
   * install caddy with plugin, directly replace `/usr/bin/caddy`
   * cloudflare: only first-level DNS is free `*.example.com`, second-level DNS is not free `*.sub.example.com`
   * cloudflare: require `edit` token, `read` token is not enough

ubuntu service

```bash
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
