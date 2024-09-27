import pandas as pd
import traceback
from .config import DATA_PATH, document_download_config
from .classifier import predict_branch_code
from ClaimProcessingAI.model import document_classfier
from ClaimProcessingAI.settings import model, ocr_agent, spacy_nlp, nlp, client, collection, classifier, mlb
from tqdm import tqdm 
import requests
import json
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")
import pybase64
from io import BytesIO
import base64
import pdf2image
# conda install -c conda-forge poppler
import json
import requests # request img from web
import shutil
# import boto3
import io, os, sys, requests, pandas, threading, csv, time
from operator import itemgetter
from functools import reduce
import numpy as np
import time
import traceback
import layoutparser as lp
import cv2
import time

def detected_icds(detected_icds):
        detected_codes = []
        for icds in detected_icds:
            for code in icds.values():
                 detected_codes.append(code[:3])
        return detected_codes

def extract_entities(fid, ds_pages):
    # Define an empty list to store results
    """
        Get text from files.
        
        Parameters:
        ds_pages (list): List of page numbers.
        docBase64 (str): Base64 encoded document.
        log_arr_1 (list): Log array.
    """
    text_block_list = []
    ds_text_block_list = []
    entity_list = []
    code_list = []
    ds_entity_list = []
    ds_code_list = []
    max_score_list =[]
    
    # Iterate over the range of page numbers
    for page_number in ds_pages:  # from 1 to 100
        # Construct the file path
        error=None
        try:
            
            file_path = "saved/"+str(fid)+"_"+str(page_number)+".jpeg"
            ds_pages = [2]
            docBase64 = ""
            log_arr_1 = []
            print("------",file_path)
            get_files_text(fid, [page_number],"",[])
            # Perform your processing
            strt = time.time()
            doc_obj = document_classfier(model, ocr_agent, nlp, collection,spacy_nlp,classifier,mlb)
            all_ents, all_text,ds_ents, ds_text,all_codes,ds_codes, max_score,_ = doc_obj.main_func(file_path)
            end = time.time()
            text_block_list.append(all_text) 
            ds_text_block_list.append(ds_text) 
            entity_list.append(all_ents) 
            code_list.append(all_codes) 
            ds_entity_list.append(ds_ents) 
            ds_code_list.append(ds_codes) 
            max_score_list.append(max_score) 
    
            # Append results to the list
            # results_list.append((page_number, str(text_blocks),str(set(all_ents)),str(all_codes), end-strt))
        except Exception as e:
            # text_blocks,all_ents,all_codes=None,None,None
            error = traceback.print_exc()
            print(traceback.print_exc())
            continue
    
    new_detected_code_list = detected_icds(code_list)
    primary_code = predict_branch_code(new_detected_code_list, classifier, mlb) 
    
    return text_block_list, ds_text_block_list,entity_list, code_list, ds_entity_list,ds_code_list, error, max_score_list, primary_code

def get_files_text(fid, ds_pages,docBase64,log_arr_1):
    log_arr_1.append("Entered get_files_text function")
    
    base64flag = 0
    if docBase64=="":
    
        URL = document_download_config['URL']

        PARAMS = {"fileIdentifier": fid}

        r = requests.post(url = URL, params = PARAMS, headers=document_download_config['header'], stream = True)
        
        x = r.text
        
        indx = x.find('var model')
        indy = x[indx:].find('}')
        j = x[indx:][:indy+1]
        d = json.loads(j.split('=',1)[1].lstrip())
        
        docBase64 = d['fileBytes']
        base64flag = 1
        
        
    ds_text = ''
#
    threads = []
    
    if docBase64!="" or (base64flag == 1 and 'fileBytes' in d.keys() and '.pdf' in d['fileName'].lower()):
        s= time.time()
        pages = pdf2image.convert_from_bytes(BytesIO(pybase64.b64decode(docBase64)).read())
        ds = [p for i,p in enumerate(pages) if i+1 in ds_pages]
        file_path = "saved/"+str(fid)+"_"+str(ds_pages[0])+".jpeg"
        print(file_path)
        cv2.imread(ds[0].save(file_path,"jpeg"))



# Assuming extract_entities function is defined elsewhere
def extract_ds_pages(x):
    """
    Extract discharge summary pages from a dictionary.
    
    Parameters:
    x (str): Input dictionary.
    
    Returns:
    int: Discharge summary page number.
    """
    for i in eval(x):
        if i["Name"]=="discharge_summary":
            return i["PageNumber"]
    return []

def run(test):
    """
    Run the extraction process on the test data.
    
    Parameters:
    test (pandas.DataFrame): Test data.
    """
    test['ds_pages'] = test["page"].apply(lambda x:extract_ds_pages(x))
    batch_size = 20
    total_fids = test.shape[0]
    print(total_fids)
    for i in tqdm(range(0, total_fids, batch_size)):
        batch_test = test.iloc[i:i+batch_size].copy()  # Get a batch of 20 FIDs
        batch_test[['text_block_list', 'ds_text_block_list','entity_list', 'code_list', 'ds_entity_list','ds_code_list', 'error', 'max_score_list',"primary_code_list"]] =    batch_test.progress_apply(lambda x:pd.Series(extract_entities(x.fid, x.ds_pages)), axis=1)
        batch_test.to_csv(f'batch_ds_block2_{(i+1)//batch_size}_test.csv', index=False)  # Save t


if '__name__'=='__main__':
    test = pd.read_csv(DATA_PATH)
    run(test)
    