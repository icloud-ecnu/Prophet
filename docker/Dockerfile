FROM nvidia/cuda:10.0-devel-ubuntu18.04

ARG https_proxy
ARG http_proxy

ENV USE_CUDA_PATH /usr/local/cuda:/usr/local/cudnn/lib64
ENV PATH /usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/cudnn/lib64:/usr/local/cuda/lib64:$LIBRARY_PATH

ENV BYTEPS_BASE_PATH /usr/local
ENV BYTEPS_PATH $BYTEPS_BASE_PATH/byteps
ENV BYTEPS_GIT_LINK https://github.com/icloud-ecnu/Prophet

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq
RUN apt-get install -y --allow-unauthenticated --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        tzdata \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        cmake \
        lsb-release \
        libcudnn7=7.6.0.64-1+cuda10.0 \
        libnuma-dev \
        ibverbs-providers \
        librdmacm-dev \
        ibverbs-utils \
        rdmacm-utils \
        libibverbs-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools

# Install NCCL
ENV NCCL_VERSION=d7a58cfa5865c4f627a128c3238cc72502649881
RUN cd / && \
    wget -q -O - https://github.com/NVIDIA/nccl/archive/$NCCL_VERSION.tar.gz | tar -xzf - && \
    cd nccl-$NCCL_VERSION && make -j src.build && make pkg.txz.build && \
    mkdir -p /usr/local/nccl && \
    tar -Jxf /nccl-$NCCL_VERSION/build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1 && \
    echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig && rm -rf /nccl-$NCCL_VERSION

WORKDIR /root/

# install gcc 4.9
RUN mkdir -p /root/gcc/ && cd /root/gcc &&\
    wget http://launchpadlibrarian.net/247707088/libmpfr4_3.1.4-1_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728424/libasan1_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728426/libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728314/gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728399/cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728404/gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728432/libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728401/g++-4.9_4.9.3-13ubuntu2_amd64.deb

RUN cd /root/gcc &&\
    dpkg -i gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libmpfr4_3.1.4-1_amd64.deb &&\
    dpkg -i libasan1_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i g++-4.9_4.9.3-13ubuntu2_amd64.deb

# Pin GCC to 4.9 (priority 200) to compile correctly against TensorFlow, PyTorch, and MXNet.
RUN update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200


RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cudnn/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig

RUN ln -sf /usr/local/cudnn/include/cudnn.h /usr/local/cuda/include/ && \
    ln -sf /usr/local/cudnn/lib64/libcudnn* /usr/local/cuda/lib64 &&\
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    ln -sf /usr/local/cuda/lib64/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

# install mxnet
ARG FRAMEWORK_VERSION=1.5.0
RUN python3 -m pip --no-cache-dir install mxnet-cu100==$FRAMEWORK_VERSION

# Install BytePS
ARG BYTEPS_NCCL_LINK=shared
ARG BYTEPS_USE_RDMA=1
ARG BYTEPS_WITHOUT_PYTORCH=1
ARG BYTEPS_WITHOUT_TENSORFLOW=1
ARG BYTEPS_BRANCH=master
RUN cd $BYTEPS_BASE_PATH &&\
    git clone --recursive -b $BYTEPS_BRANCH $BYTEPS_GIT_LINK
RUN cd $BYTEPS_PATH &&\
    python3 setup.py install

# Remove GCC pinning
RUN update-alternatives --remove gcc /usr/bin/gcc-4.9 && \
    update-alternatives --remove x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 && \
    update-alternatives --remove g++ /usr/bin/g++-4.9 && \
    update-alternatives --remove x86_64-linux-gnu-g++ /usr/bin/g++-4.9

RUN rm -rf /usr/local/cuda/lib64/libcuda.so && \
    rm -rf /usr/local/cuda/lib64/libcuda.so.1
