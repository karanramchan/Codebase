import sys
# sys.modules['sqlite3'] = __import__('pysqlite3')
# import chromadb
# from chromadb.config import Settings
# import stanza
# import scispacy
# import spacy
import os
import requests
import copy

from config import OCR_THRESHOLD, VD_PATH, NLP_MODEL, CONFIG_PATH, LABEL_MAP
#from layoutparser.models import Detectron2LayoutModel, detectron2
# import layoutparser as lp
import joblib



model=None
ocr_agent = None
model =joblib.load("model_files/lgb_model")
vectorizer =joblib.load("model_files/vectorizer")
le =joblib.load("model_files/label_encoder")
model_v2 =joblib.load("model_files/lgb_model_v2")
vectorizer_v2 =joblib.load("model_files/vectorizer_v2")
onehot_encoder_v2 =joblib.load("model_files/onehot_encoder_v2")
scaler_v2 =joblib.load("model_files/scaler_v2")

