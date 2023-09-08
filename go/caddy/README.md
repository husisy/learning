# caddy

TLS termiantion proxy

1. link
   * [website](https://caddyserver.com/)
   * [github](https://github.com/caddyserver/caddy)
2. preferences
   * do **NOT** use `Caddyfile`
   * do **NOT** use CLI interface if possible

```bash
caddy
caddy run
caddy adapt --config /path/to/Caddyfile
caddy start

curl localhost:2019/config/
curl localhost:2019/load -H "Content-Type: application/json" -d @example00.json
curl localhost:2015
```

`Caddyfile`

```text
:2015

respond "Hello, world!"
```
