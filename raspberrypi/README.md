# raspberrypi

1. link
   * [official site](https://www.raspberrypi.org/)
   * [raspberry pi project](https://projects.raspberrypi.org/en)
   * [树莓派实验室](http://shumeipai.nxez.com/)
   * [magpi-raspberry pi beginner's guide](https://magpi.raspberrypi.org/books/beginners-guide-3rd-ed/pdf)
2. [系统下载](https://www.raspberrypi.org/downloads/)
   * raspbian也是使用apt管理package
   * [ubuntu for raspberrypi](https://ubuntu.com/download/raspberry-pi)
3. software
   * [balenaetcher](https://www.balena.io/etcher/)

```bash
sudo apt update
sudo apt upgrade

# ssh
sudo apt install openssh-server
sudo systemctl enable ssh
# systemctl disable sshd
sudo apt install xrdp #for windows remote desktop

sudo raspi-config

# user
sudo adduser xxx
sudo usermod -a -G sudo xxx
sudo usermod -a -G video xxx #for camera access
passwd xxx
# ssh-copy-id xxx@remotehost #run in host-pc
```
