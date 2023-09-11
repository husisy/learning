# config example

1. replace `example.com` with the true domain name, or use `localhost` for testing
2. example list
   * `Caddyfile00`: hello world
   * `Caddyfile01`: static file server
   * `Caddyfile02`: reverse proxy
3. format
   * indent with tab (not four spaces)

```bash
caddy adapt --config Caddyfile | jq
caddy adapt --config Caddyfile > caddy.json
curl localhost:2019/load -H "Content-Type: application/json" -d @caddy.json
```
