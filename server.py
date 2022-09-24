# silence all tqdm progress bars
from platform import platform
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from version import __version__
import arrow
from typing import Union, List, Dict, Optional, Any
from pydantic import BaseModel, Field
from time import time
from glob import glob
from re import findall
import yaml
import os

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging
import contextlib


if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield

from nltk import download, sent_tokenize
download('punkt')

_TEXT_LEN_LIMIT = 5000
_TEXT_SPLIT_THRESHOLD = 1024
_SPLIT_LEN = 512
_use_gpu_if_available = True

class NMTModel(BaseModel):
  class Config:
    arbitrary_types_allowed = True
  tag: str
  nemo: ModelPT
  platform: str
  active: int

start_time: str = None
models: Dict[str, Dict[str, NMTModel]] = {}
num_requests_processed: int = None



app = FastAPI(
  title='NMT API',
  version=__version__,
  contact={
      "name": "Vitasis Inc.",
      "url": "https://vitasis.si/",
      "email": "info@vitasis.si",
  }
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
  src_language: str = Field( ..., title="Source language", description="ISO 639-1 two-letter language code, lowercase")
  tgt_language: str = Field( ..., title="Target language", description="ISO 639-1 two-letter language code, lowercase")
  text: Union[str,List[str]]

translateRequestExamples = {
  "text: string": {
    "value": {
      "src_language": "sl",
      "tgt_language": "en",
      "text": "Danes prijetno sneži. Jutri bo pa še lepše."
    }
  },
  "text: [string]": {
    "value": {
      "src_language": "sl",
      "tgt_language": "en",
      "text": [ "Danes prijetno sneži.", "Jutri bo pa še lepše." ]
    }
  },
}


class TranslateResponse(BaseModel):
  result: Union[str,List[str]]

translateResponseExamples = {
  "result: string": {
    "value": {
      "result": "It snows well today, and tomorrow it will be even better."
    }
  },
  "result: [string]": {
    "value": {
      "result": [ "It snows pleasantly today.","Tomorrow will be even better." ]
    }
  },
}


class Model(BaseModel):
  tag: str
  workers: Dict[str,Any]
  features: Optional[Dict[str,Any]]
  info: Optional[Dict[str,Any]]

class HealthCheckResponse(BaseModel):
  status: int
  start_time: Optional[str]
  models: Optional[List[Model]]
  num_requests_processed: Optional[int]

healthCheckResponseExamples = {
  "serving": {
    "value": {
      "status": 0,
      "start_time": arrow.utcnow().isoformat(),
      "models": [
        { "tag": "slen:GEN:nemo-1.2.6", "workers": { "platform": "gpu", "active": 2 } },
        { "tag": "ensl:GEN:nemo-1.2.6", "workers": { "platform": "gpu", "active": 0 } },
      ]
    }
  },
  "failed state": {
    "value": {
      "status": 2,
    }
  },
}


@app.get(
  "/api/healthCheck",
  description="Retrieve service health info.",
  response_model=HealthCheckResponse,
  responses={ 200: { "description": "Success", "content": { "application/json": { "examples": healthCheckResponseExamples } } } }
)
def health_check():
  _SERVICE_UNAVAILABLE_ = -1
  _PASS_ = 0
  _WARN_ = 1
  _FAIL_ = 2

  status: HealthCheckResponse = {'status': _SERVICE_UNAVAILABLE_}
  if not models:
    status = {'status': _FAIL_}
  else:
    status = {'status': _PASS_}
    min_workers_available = 1 # min([ workers_info['available'] for workers_info in _response['workers_info'] ]) if len(_response['workers_info']) > 0 else 0
    if min_workers_available <= -1: # config['workers']['fail']
      status = {'status': _FAIL_}
    elif min_workers_available <= 0: # config['workers']['warn']:
      status = {'status': _WARN_}
    status['models'] = [ { "tag": models[src_lang][tgt_lang].tag, "workers": { "platform": models[src_lang][tgt_lang].platform, "active": models[src_lang][tgt_lang].active } } for src_lang in models for tgt_lang in models[src_lang] ]
    status['start_time'] = start_time
    status['num_requests_processed'] = num_requests_processed

  return status

@app.post(
  "/api/translate",
  description=f"Translate text. Maximum text lenght is {_TEXT_LEN_LIMIT}c.\n\nInput: Text.\n\nOutput: Translation.",
  response_model=TranslateResponse,
  responses={ 200: { "description": "Success", "content": { "application/json": { "examples": translateResponseExamples } } } }
)
def translate_text(item: TranslateRequest = Body(..., examples=translateRequestExamples)):
  time0 = time()
  if item.src_language.lower() not in models:
    raise HTTPException(status_code=404, detail=f"Source language {item.src_language} unsupported")
  if item.tgt_language.lower() not in models[item.src_language.lower()]:
    raise HTTPException(status_code=404, detail=f"Target language {item.tgt_language} unsupported")

  logging.info(f" Q: {item.text}")

  if isinstance(item.text, str):
    text = [item.text]
  else:
    text = item.text
  text_len = sum(len(_text) for _text in text)
  if text_len > _TEXT_LEN_LIMIT:
    logging.warning(f'{text}, text length exceded {text_len}c [max {_TEXT_LEN_LIMIT}c]')
    raise HTTPException(status_code=400, detail=f"Bad request.")

  text_batch = []
  text_batch_split = []
  for _text in text:
    if len(_text) > _TEXT_SPLIT_THRESHOLD:
      _split_start = len(text_batch)
      _sent = sent_tokenize(_text)
      i = 0
      while i < len(_sent):
        j = i+1
        while j < len(_sent) and len(' '.join(_sent[i:j])) < _SPLIT_LEN: j+=1
        if len(' '.join(_sent[i:j])) > _TEXT_SPLIT_THRESHOLD:
          _split=findall(rf'(.{{1,{_SPLIT_LEN}}})(?:\s|$)',' '.join(_sent[i:j]))
          text_batch.extend(_split)
        else:
          text_batch.append(' '.join(_sent[i:j]))
        i = j
      _split_end = len(text_batch)
      text_batch_split.append((_split_start,_split_end))
    else:
      text_batch.append(_text)

  logging.debug(f' B: {text_batch}, BS: {text_batch_split}')

  if _use_gpu_if_available and torch.cuda.is_available():
      models[item.src_language.lower()][item.tgt_language.lower()].nemo = models[item.src_language.lower()][item.tgt_language.lower()].nemo.cuda()

  models[item.src_language.lower()][item.tgt_language.lower()].active += 1
  translation_batch = models[item.src_language.lower()][item.tgt_language.lower()].nemo.translate(text_batch)
  logging.debug(f' BT: {translation_batch}')
  models[item.src_language.lower()][item.tgt_language.lower()].active -= 1

  translation = []
  _start = 0
  for _split_start,_split_end in text_batch_split:
    if _split_start != _start:
      translation.extend(translation_batch[_start:_split_start])
    translation.append(' '.join(translation_batch[_split_start:_split_end]))
    _start = _split_end
  if _start < len(translation_batch):
    translation.extend(translation_batch[_start:])

  result: TranslateResponse = { "result": ' '.join(translation) if isinstance(item.text, str) else translation }

  logging.info(f' R: {result}')
  logging.debug(f'text_length: {text_len}c, duration: {round(time()-time0,2)}s')
  global num_requests_processed
  num_requests_processed += 1

  if num_requests_processed == 0:
    if _use_gpu_if_available and torch.cuda.is_available():
      # Force onto CPU
      models[item.src_language.lower()][item.tgt_language.lower()].nemo = models[item.src_language.lower()][item.tgt_language.lower()].nemo.cpu()
      torch.cuda.empty_cache()

  return result


def initialize():
  time0 = time()
  models: Dict[str, Dict[str, NMTModel]] = {}
  for _model_info_path in glob(f"./models/**/model.info",recursive=True):
    with open(_model_info_path) as f:
      _model_info = yaml.safe_load(f)

    lang_pair = _model_info.get('language_pair', None)
    if lang_pair:
      _model_tag = f"{_model_info['language_pair']}:{_model_info['domain']}:{_model_info['version']}"
      _model_platform = "gpu" if _use_gpu_if_available and torch.cuda.is_available() else "cpu"
      _model_path = f"{os.path.dirname(_model_info_path)}/{_model_info['info']['framework'].partition(':')[-1].replace(':','_')}.{_model_info['info']['framework'].partition(':')[0]}"

      model = ModelPT.restore_from(_model_path,map_location="cuda" if _model_platform == "gpu" else "cpu")
      model.freeze()
      model.eval()

      if lang_pair != f"{model.src_language.lower()}{model.tgt_language.lower()}":
        logging.warning(f"Invalid model.info; language_pair '{lang_pair}', {_model_info['info']['framework'].partition(':')[-1].replace(':','_')}.{_model_info['info']['framework'].partition(':')[0]} '{model.src_language.lower()}{model.tgt_language.lower()}', unloading")
        del model
        continue

      models[model.src_language.lower()] = {}
      models[model.src_language.lower()][model.tgt_language.lower()] = NMTModel(
        tag = _model_tag,
        nemo = model,
        platform = _model_platform,
        active = 0,
      )

  logging.info(f'Loaded models {[ (models[src_lang][tgt_lang].tag,models[src_lang][tgt_lang].platform) for src_lang in models for tgt_lang in models[src_lang] ]}')
  logging.info(f'Initialization finished in {round(time()-time0,2)}s')

  start_time = arrow.utcnow().isoformat()
  num_requests_processed = 0
  return start_time, models, num_requests_processed

def start_service():
  uvicorn.run(app, host="0.0.0.0", port=4000)

if __name__ == "__main__":
  logging.setLevel(logging.DEBUG)
  start_time, models, num_requests_processed = initialize()
  start_service()
