FROM nvcr.io/nvidia/nemo:22.07 as nemo

ARG DEBIAN_FRONTEND=noninteractive

FROM nemo as service

ARG DEBIAN_FRONTEND=noninteractive

COPY . /opt/nmt
RUN python3 -m pip install -r /opt/nmt/requirements.txt
WORKDIR /opt/nmt

ENTRYPOINT [ "python3", "server.py" ]
