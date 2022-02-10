# docker

1. link
   * [github-prakhar1989/docker-curriculum](https://github.com/prakhar1989/docker-curriculum)
   * [gitbook-docker从入门到实践](https://github.com/yeasy/docker_practice)
   * [Docker-login-credential](https://youendless.com/post/docker_login_pass/)
   * [Docker / Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
   * docker gpu support: [github/nvidia/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
2. 术语
   * image，分层存储
   * container，容器存储层，数据卷
   * Registry (eg. docker hub), Tepository, Tag, Mirror：一个Registry包含多个Repository，每个Repository可以包含多个Tag
   * docker daemon, docker client
   * docker user group
3. 常用命令
   * `docker version`
   * `docker info`
   * `docker image ls`
   * `docker container run -it busybox sh`
   * `docker container ls -a`
   * `docker image rm xxx:tag` 删除image
   * `docker container rm xxx` 删除container
   * `docker container prune`
4. docker hub
   * [official site](https://hub.docker.com/)
   * [USTC mirror](https://mirrors.ustc.edu.cn/help/dockerhub.html): `"registry-mirrors": ["https://docker.mirrors.ustc.edu.cn/"]`
   * [tencent mirror](https://cloud.tencent.com/document/product/457/9113)
5. 私有仓库
   * `docker run -d -p 6000:5000 -v /media/docker_registry:/var/lib/registry registry`
   * `docker tag zctest00:latest 192.168.120.31:6000/zctest00:latest`
   * `docker push 192.168.120.31:6000/zctest00:latest`
6. misc
   * 在`arm`平台拉取`amd64`平台的镜像 [stackoverflow](https://stackoverflow.com/a/60116565)
7. cuda配置
   * [github/nvidia/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
   * [github/nvidia/nvidia-docker/doc](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# frequently used parameter
--mount type=bind,src=/host/path,dst=/container/path
-u $(id -u):$(id -g)
```

## Dockerfile

1. `COPY package.json /usr/src/app/`
   * `COPY hom* /mydir/`
   * `COPY hom?.txt /mydir/`
2. 不推荐使用`ADD`，可以完全被`COPY`与`RUN`取代

## mwe: hello-world

1. `docker image pull library/hello-world`
   * `library`是默认docker group，故可省略为`docker image pull hello-world`
   * `image`也是可省略的，故可省略未`docker pull hello-world`
   * 因为`docker container run xxx`会自动pull未下载的image，所以该命令可省去
2. `docker container run hello-world`
   * `container`是可省略的，故可省略为`docker run hello-world`
3. `docker container ls -a`
   * 上一行`docker container run hello-world`产生的容器已经停止了，故须加上`-a`参数时使之打印
4. `docker container rm xxxid`
   * 将`xxxid`替换为`docker container ls -a`中的`CONTAINER ID`
   * 亦可使用命令`docker container prune`

## mwe: ubuntu

1. `docker run -it --rm ubuntu bash`
   * `-i --interactive`: 交互式操作
   * `-t --tty`: 终端
   * `--rm`: 容器停止后自动删除
   * `bash`是默认行为，故可省略为`docker run -it --rm ubuntu`
   * `docker run -it --rm ubuntu cat /etc/os-release`
2. `cat /etc/os-release`

## mwe: nginx

1. `docker run --name webserver -d -p 2333:80 nginx`
   * `-d --detach`
   * `-p --publish`：端口映射规则，前者`2333`为host端口（被监听），后者`80`为container端口，也可写作`-p 2333:80/tcp`
   * 此时可在host访问`http://127.0.0.1:2333`网页内容，如果是`80`则可省略为`http://127.0.0.1`
2. `docker exec -it webserver bash`
   * 从`docker exec`中`exit`不会终止容器运行
3. `echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html`
   * 此时刷新`http://127.0.0.1:2333`便可看到新的内容
4. `docker diff webserver`
5. `docker commit --author "xxx@gmail.com" --message "update default index.html" webserver nginx:v2`
6. `docker history nginx:v2`
7. 清理
   * `docker container stop xxxid`
   * `docker image rm nginx:v2`
8. **慎用**`docker commit`

## mwe: dockerfile

see `./ws_dockerfile`

```Dockerfile
FROM nginx
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
```

1. `FROM nginx`
   * 其它常见镜像：服务类`nginx redis mongo mysql httpd php tomcat`，语言类`node openjdk python ruby golang`，操作系统类`ubuntu debian centos fedora alphine`，空白镜像`scratch`
2. `RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html`
   * 写成单行命令
   * 删除不必要的文件`rm -rf /var/lib/apt/lists/*`
   * 卸载不必要的包`apt-get purge -y --auto-remove gcc libc6-dev make wget`
3. `docker build -t nginx:v3 .`
   * 上下文路径
   * `.dockerignore`
   * 直接从git repo进行构建 `docker build -t self/hello-world https://github.com/docker-archive/dockercloud-hello-world.git`
   * 用给定的tar压缩包构建
   * 从标准输入构建（无上下文路径） `docker build - < Dockerfile` `cat Dockerfile | docker build -`

一个推荐示例Dockerfile

```Dockerfile
FROM debian:stretch

RUN buildDeps='gcc libc6-dev make wget' \
    && apt-get update \
    && apt-get install -y $buildDeps \
    && wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
    && mkdir -p /usr/src/redis \
    && tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1 \
    && make -C /usr/src/redis \
    && make -C /usr/src/redis install \
    && rm -rf /var/lib/apt/lists/* \
    && rm redis.tar.gz \
    && rm -r /usr/src/redis \
    && apt-get purge -y --auto-remove $buildDeps
```

## misc

```bash
docker save ubuntu:latext > docker_ubuntu.tar
docker load --input docker_ubuntu.tar
```

deploy docker registry

```bash
# https://docs.docker.com/registry/deploying/
docker run -d -p 15000:5000 --restart=always --name registry-zc registry:2
docker pull ubuntu
docker tag ubuntu:latest localhost:15000/my-ubuntu
docker push localhost:15000/my-ubuntu
docker image remove ubuntu:latest localhost:15000/my-ubuntu
docker pull localhost:15000/my-ubuntu
docker tag localhost:15000/my-ubuntu ubuntu
docker image ls
docker image rm localhost:15000/my-ubuntu
```
