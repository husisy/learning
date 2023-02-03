# v2ray

1. link
   * [official-guide](https://guide.v2fly.org/)
   * [github/v2ray-core](https://github.com/v2fly/v2ray-core)
   * [v2ray-manual](https://v2fly.org/)
   * [github/fhs-install-v2ray](https://github.com/v2fly/fhs-install-v2ray)
   * [uuid-generator](https://www.uuidgenerator.net/)
   * 漫谈各种黑科技式 DNS 技术在代理环境中的应用 [link](https://tachyondevel.medium.com/%E6%BC%AB%E8%B0%88%E5%90%84%E7%A7%8D%E9%BB%91%E7%A7%91%E6%8A%80%E5%BC%8F-dns-%E6%8A%80%E6%9C%AF%E5%9C%A8%E4%BB%A3%E7%90%86%E7%8E%AF%E5%A2%83%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8-62c50e58cbd0)
2. install
3. server config file `/usr/local/etc/v2ray/config.json`

```bash
date -R
apt install curl
bash <(curl -L https://raw.githubusercontent.com/v2fly/fhs-install-v2ray/master/install-release.sh)
systemctl start v2ray
systemctl enable v2ray #enable auto-start after starting machine
systemctl status v2ray
systemctl stop v2ray

cat /proc/sys/kernel/random/uuid

v2ray test -config config.json
```

```json
{
  "inbounds": [
    {
      "port": 19792,
      "protocol": "vmess",
      "settings": {
        "clients": [
          {
            "id": "65f1c4a5-4311-4762-8604-e7573f07bdc6",
            "alterId": 0
          }
        ]
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "freedom",
      "settings": {}
    }
  ]
}
```

```json
{
  "inbounds": [
    {
      "port":   ,
      "protocol": "socks",
      "sniffing": {
        "enabled": true,
        "destOverride": [
          "http",
          "tls"
        ]
      },
      "settings": {
        "auth": "noauth"
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "vmess",
      "settings": {
        "vnext": [
          {
            "address": "127.0.0.1",
            "port": 19792,
            "users": [
              {
                "id": "65f1c4a5-4311-4762-8604-e7573f07bdc6",
                "alterId": 0
              }
            ]
          }
        ]
      }
    }
  ]
}
```
