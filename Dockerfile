FROM nvcr.io/nvidia/pytorch:22.08-py3 as nemo

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && python3 -m pip install --upgrade pip \
    && git clone https://github.com/NVIDIA/NeMo.git /workspace/nemo \
    && cd /workspace/nemo \
    && git checkout v1.11.0 \
    && ./reinstall.sh

FROM nemo as service

ARG DEBIAN_FRONTEND=noninteractive

COPY . /opt/nmt
RUN python3 -m pip install -r /opt/nmt/requirements.txt
WORKDIR /opt/nmt

ENTRYPOINT [ "python3", "server.py" ]
