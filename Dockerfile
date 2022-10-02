FROM nvcr.io/nvidia/tensorrt:22.08-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul  apt-get install -y \
        cmake \
        g++ \
        wget \
        unzip \
        vim \
        libopencv-dev \
        libboost-all-dev \
        python3 \
        python3-pip \
        libasio-dev


RUN wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu116.zip -P /opt/
RUN unzip /opt/libtorch-cxx11-abi-shared-with-deps-1.12.1+cu116.zip -d /usr/local/include/
WORKDIR /app
COPY . .