FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
COPY main.sh /root/main.sh
RUN . /root/main.sh
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
