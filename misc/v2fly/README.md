# v2ray

1. link
   * [v2fly-official-site](https://www.v2fly.org/)
   * [official-guide](https://guide.v2fly.org/)
   * [github/v2ray-core](https://github.com/v2fly/v2ray-core)
   * [github/fhs-install-v2ray](https://github.com/v2fly/fhs-install-v2ray)
   * [uuid-generator](https://www.uuidgenerator.net/)
   * 漫谈各种黑科技式 DNS 技术在代理环境中的应用 [link](https://tachyondevel.medium.com/%E6%BC%AB%E8%B0%88%E5%90%84%E7%A7%8D%E9%BB%91%E7%A7%91%E6%8A%80%E5%BC%8F-dns-%E6%8A%80%E6%9C%AF%E5%9C%A8%E4%BB%A3%E7%90%86%E7%8E%AF%E5%A2%83%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8-62c50e58cbd0)
2. install
   * download from github release page
   * verify `md5 /path/to/v2ray-linux-64.zip`, `cat /path/to/v2ray-linux-64.zip.dgst`
   * `v2ray` support `systemctl` and `journalctl`
3. concept
   * 单服务器模式、桥接模式（墙内VPS+墙外VPS）
   * dispatcher
   * router：路由规则，多个出站协议
   * DNS
   * 入站协议inbound
   * 出站协议outbound
   * 命令`v2ctl`合并至`v2ray`
   * 服务器和客户端的时间误差不超过90s
   * `AlterID`必须为零 [v2ray-core/issue812](https://github.com/233boy/v2ray/issues/812)
4. server config file `/usr/local/etc/v2ray/config.json`
5. connection check
   * server side: ufw, v2ray `0.0.0.0`
6. how to debug
   * startup `v2ray -config /path/to/config.json` manually, **NOT** from `systemctl start` (hard to find the logging info)
   * `loglevel=Debug`, `type=Console`
7. TODO
   * [v2ray-warp](https://www.4spaces.org/3750.html)
   * [github/frp](https://github.com/fatedier/frp#example-usage)

```bash
date -R
bash <(curl -L https://raw.githubusercontent.com/v2fly/fhs-install-v2ray/master/install-release.sh)
# PROXY=xxx

systemctl start v2ray
systemctl enable v2ray #enable auto-start after starting machine
systemctl status v2ray
systemctl stop v2ray
journalctl -u v2ray

cat /proc/sys/kernel/random/uuid

v2ray test -config config.json
v2ray help run
```

```json
{
    "log": {},
    "dns": {},
    "router": {},
    "inbounds": [],
    "outbounds": [],
    "services": {}
}
```

```json
{
  "log": {
    "loglevel": "Debug",
    "type": "Console"
  }
}
```
