# NMT english-slovene translator

## NMT API

Implements the NMT API for NeMo AAYN Base models. For more details about building such models, see the official [NVIDIA NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/machine_translation/machine_translation.html) and [NVIDIA NeMo GitHub](https://github.com/NVIDIA/NeMo).

The API provides two endpoints **GET** `/api/healthCheck` (note capital C), to retrieve the service status, and **POST** `/api/translate` to request a translation. The service accepts either a single string or list of strings. The result will be in the same format as the request, either as a single string or list of strings.

Defalut port is **4000**.

Example request body:

```json
{
  "src_language": "en",
  "tgt_language": "sl",
  "text": [
    "Hey hello",
    "this is a test"
  ]
}
```

Example curl request:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"src_language": "en", "tgt_language": "sl", "text": ["Hey hello", "this is a test"]}' http://localhost:4000/api/translate
```

The maximal accepted text length is 5000c. Note that transcription of one 5000c text block on cpu will take advantage of all available cores, consume up to 3GB RAM and may take ~200s (on a system with 24 vCPU).

The parameters can be configured in `server.py`:

```python
_TEXT_LEN_LIMIT = 5000 # characters
_TEXT_SPLIT_THRESHOLD = 1024 # characters
_SPLIT_LEN = 512 # characters
```

## Prerequisites

- docker >= 20.10.17
- docker compose >= 2.6.0
- NeMo model and `model.info`

### Prerequisites for gpu deployment

- [nvidia cuda](https://developer.nvidia.com/cuda-downloads)
- [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - configured docker runtime to use with nvidia container toolkit

## Model.info

The expected format for `model.info` is:
```yml
language_pair: # source and target two-letter ISO 639-1 Language Code, lowercase, eg. slen or ensl
domain: # model domain
version: # model version
info:
  build: # build time in YYYYMMDD-HHSS format
  framework: nemo:aayn:base
  ... # aditional info, optional
features: # optional
  ... # information about special features
```

The NeMo model file is expected in the same folder, named as `aayn_base.nemo`.

The Neural Machine Translation model for language pair SL-EN developed as part of work package 4 of the Development of Slovene in a Digital Environment, RSDO, project (https://www.slovenscina.eu/en/machine-translation), can be downloaded from http://hdl.handle.net/11356/1736.

## Deployment

Run `docker compose up -d` to deploy on cpu or `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` to run on gpu.

## Approximate memory consumption for CPU deployment

- 5GB RAM for service and models
- 3GB RAM per 5000c request

## Approximate memory consumption for GPU deployment

- 5GB VRAM for service and models
