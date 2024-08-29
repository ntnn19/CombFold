FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    make \
    g++ \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir biopython==1.84 py3Dmol scipy numpy

RUN wget -qnc -O CombFold-master.zip https://github.com/dina-lab3D/CombFold/archive/refs/heads/master.zip && \
    unzip -q CombFold-master.zip && \
    echo "Downloaded CombFold, Installing" && \
    cd CombFold-master/CombinatorialAssembler && \
    make --silent && \
    echo "CombFold Installed!"

