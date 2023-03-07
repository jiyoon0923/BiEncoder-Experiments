FROM nvcr.io/nvidia/pytorch:22.07-py3

ARG DEBIAN_FRONTEND=noninteractive
# apex, jupyter, jupyterlab ... are already installed in the container

RUN apt-get update && apt-get install -y --no-install-recommends \
            git \
            ssh \
            build-essential \
            locales \
            ca-certificates \
            curl \
            unzip \
            vim \
            wget \
            tmux \
            pciutils \
            gnome-keyring \
            htop \
            fonts-nanum \
            ## clean up 
            && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# encoding setup
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# install python libraries
RUN pip install --upgrade pip
RUN pip install omegaconf wandb 
WORKDIR /workspace
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

# install elastic search
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz \
    && wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz.sha512 \
    && shasum -a 512 -c elasticsearch-7.17.9-linux-x86_64.tar.gz.sha512 \
    && tar -xzf elasticsearch-7.17.9-linux-x86_64.tar.gz --directory /workspace \
    && rm *.gz \
    && rm *.gz.sha512 \
    && chown -R $UNAME:$UNAME /workspace/elasticsearch-7.17.9

USER $UNAME
