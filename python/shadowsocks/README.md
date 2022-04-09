# shadowsocks

1. link
   * [archlinux-wiki](https://wiki.archlinux.org/index.php/Shadowsocks_(%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87))
   * [github](https://github.com/shadowsocks/shadowsocks/tree/master)
   * [shadowsocksrr](https://github.com/shadowsocksrr)
   * [秋水逸冰](https://teddysun.com/)
   * [github/setup-ipsec-vpn](https://github.com/hwdsl2/setup-ipsec-vpn)
   * [github/qwj/python-proxy](https://github.com/qwj/python-proxy)
   * [github/proxy.py](https://github.com/abhinavsingh/proxy.py)
   * [v2ray-documentation](https://www.v2ray.com/)

TODO

1. docker对系统要求过高（4GB RAM），用于部署实在不现实
2. 测试v2ray

```bash
sudo -i #ssserver cannot run without sudo privilege, see https://github.com/shadowsocks/shadowsocks-libev/issues/1724
apt update
apt upgrade
apt install python3-pip libsodium-dev
mkdir ~/shadowsocks #create ss folder for log, config, etc.
pip3 install -U git+https://github.com/shadowsocks/shadowsocks.git@master #install ss from github, not pypi which is out-dated
nano ~/shadowsocks/config.json
# its okay to use "server":"127.0.0.1", then you need to ssh port forwarding

ssserver -c ~/shadowsocks/config.json --user nobody -d start --log-file ~/shadowsocks/log
sslocal -c ~/shadowsocks/config.json --user=nobody -d start --log-file ~/shadowsocks/log #TODO maybe sslocal not need sudo priviledge
```

`config.json`

```json
{
    "server":"xxx.xxx.xxx.xxx",
    "server_port":23333,
    "local_address":"127.0.0.1",
    "local_port":1080,
    "password":"xxxxxxxxxxxxxxxxx",
    "timeout":300,
    "method":"chacha20-ietf-poly1305",
    "fast_open":false
}
```
