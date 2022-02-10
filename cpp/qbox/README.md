# qbox

1. link
   * [documentation](http://qboxcode.org/doc/html/index.html)
   * [official-site](http://qboxcode.org/)
2. download potential [quantum-simulation.org](http://quantum-simulation.org/index.htm)
3. Rydberg constant `2 Ry=1 Hartree= 27.211386245988(53) eV`
4. time unit `a.u.=2.4188843265857E-17s` $\frac{\hbar}{E_{hartree}}$

## installation

`centos7.Dockerfile`

```Dockerfile
# docker build -t qbox:latest .
FROM centos:centos7
COPY . /root/qbox/
ENV PATH="/usr/lib64/openmpi/bin:/root/qbox/bin:$PATH" TARGET=centos7
RUN yum -y install epel-release \
    && yum install -y xerces-c xerces-c-devel openmpi openmpi-devel lapack lapack-devel fftw fftw-devel scalapack-common scalapack-openmpi scalapack-openmpi-devel scalapack-openmpi-static libuuid libuuid-devel make gcc-c++ which nano less python3-pip \
    && pip3 install scipy lxml \
    && cp /root/qbox/build/centos7.mk /root/qbox/src/centos7.mk \
    && cd /root/qbox/src \
    && make -j8 \
    && mkdir /root/qbox/bin \
    && mv /root/qbox/src/qb /root/qbox/bin
```

`ubuntu.Dockerfile`

```Dockerfile
# docker build -t qbox:latest .
FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y libxerces-c-dev libopenmpi-dev liblapack-dev libfftw3-dev libscalapack-mpi-dev g++ gcc make nano less python3-pip uuid-dev \
    && pip3 install scipy lxml \
    && ln -s /usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so /usr/lib/x86_64-linux-gnu/libscalapack.so
ENV PATH="/root/qbox/bin:$PATH" TARGET=centos7
COPY . /root/qbox/
RUN cd /root/qbox/src \
    && cp /root/qbox/build/centos7.mk /root/qbox/src/centos7.mk \
    && make -j8 \
    && mkdir /root/qbox/bin \
    && mv /root/qbox/src/qb /root/qbox/bin \
```

`.dockerignore`

```txt
.git
.gitignore
Dockerfile
.dockerignore
COPYING
```

1. `docker build -t myqbox:latest .`
2. `docker container run -it --rm myqbox bash`
   * `export PATH="/usr/lib64/openmpi/bin:/root/qbox/bin:$PATH"`
   * `mpirun --allow-run-as-root -np 2 qb gs.i > gs.r` 见documentation-example [link](http://qboxcode.org/doc/html/usage/intro.html)
