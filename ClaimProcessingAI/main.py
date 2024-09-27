import pandas as pd
import traceback
from config import DATA_PATH, document_download_config
from classifier import predict_branch_code, predict_branch_code_v2
from model import document_classfier
from settings import model, vectorizer, le, model_v2, vectorizer_v2, onehot_encoder_v2, scaler_v2
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
# import layoutparser as lp
import cv2
import time
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Body, Response
import uvicorn
import time
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()

nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
import numpy as np
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pyodbc
import traceback

connection_string = connection_string_prod="DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.1.246;UID=webmis_140;PWD=Mediassist@123"
connection_string_prod = connection_string
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
engine = create_engine(connection_url, echo=False, pool_pre_ping=True, pool_size=150, pool_recycle=3600)


def get_data(claimid) :

    df_sql = pd.read_sql_query(text(''' select  clmid, clmamount, BenefAge, BenefSex, hospid, HospName 
                            from mediassist..tblmaclaim
                            join mediassist..tblmahospital on clmhospid = hospid
                            join mediassist..tblmabeneficiary on BenefUserID = ClmBenefid
                            where clmid = {clmid}
                            union all
                            select  clmid, clmamount, BenefAge, BenefSex, hospid, HospName 
                            from MEDIASSISTUHS..tblmaclaim
                            join MEDIASSISTUHS..tblmahospital on clmhospid = hospid
                            join MEDIASSISTUHS..tblmabeneficiary on BenefUserID = ClmBenefid
                            where clmid = {clmid}

                       '''.format(clmid = claimid)), engine)
    
    return df_sql




app = FastAPI()

def detected_icds(detected_icds):
        detected_codes = []
        for icds in detected_icds:
            for code in icds.values():
                 detected_codes.append(code[:3])
        return detected_codes

def preprocess_documents(doc):
    preprocessed_documents = []
    stop_words = set(stopwords.words('english'))
    

    doc = doc.lower()
    
    # Remove numbers
    doc = re.sub(r'\d+', '', doc)
    
    # Remove punctuation
    doc = re.sub(r'[^\w\s]', '', doc)
    
    # Remove whitespace (including newline characters)
    doc = doc.strip()
    
    # Tokenize the document (you can modify tokenization as per your needs)
    tokens = nltk.word_tokenize(doc)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into a single string
    preprocessed_doc = ' '.join(tokens)
        
    file_path = 'preprocssed.txt'
    with open(file_path, 'w') as file:
        file.write(preprocessed_doc)
        
    
     
    return preprocessed_doc

@app.post("/extractentities/")
async def extract_entities(request_data: Dict):  # Define an empty list to store results
    """
        Get text from files.
        
        Parameters:
        ds_pages (list): List of page numbers.
        docBase64 (str): Base64 encoded document.
        log_arr_1 (list): Log array.
        
    """
    final_result = []
    
    claim_id = request_data["ClaimId"]
    for docs in request_data["documentClassification"]:
        # if "fid" not in request_data or "ds_pages" not in request_data:
        #     raise HTTPException(status_code=400, detail="Missing 'fid' or 'ds_pages' in request data.")
    
        
        fid = docs["docid"]
        docbase64 = docs["docBase64"]
        ds_pages = eval(docs["DischargeSummaryPageNo"])
	 
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
                
                file_path = "img.jpeg"
                ds_pages = [2]
                docBase64 = ""
                log_arr_1 = []
                get_files_text(fid, [page_number],docbase64,[])
                # Perform your processing
                strt = time.time()
                doc_obj = document_classfier()
                all_text = doc_obj.main_func(file_path)
                end = time.time()
                text_block_list.append(preprocess_documents(all_text))
                # Append results to the list
                # results_list.append((page_number, str(text_blocks),str(set(all_ents)),str(all_codes), end-strt))
            except Exception as e:
                # text_blocks,all_ents,all_codes=None,None,None
                error = traceback.print_exc()
                print(traceback.print_exc())
                continue
        
        text = " ".join(text_block_list)
        rest_info = ""
        try:
            rest_info = get_data(claim_id)
           
        except:
            
            print(traceback.format_exc())
            print("additional info not found")
            
        print("restinfo------>",rest_info)
        
        if len(rest_info)>0:
             print(rest_info)
             rest_info.columns = ['clmid' ,'amount',  'age', 'gender' , 'hospid','hospname']
             primary_code, primary_prob, secondary_code, secondary_prob = predict_branch_code_v2(model_v2, text, vectorizer_v2, onehot_encoder_v2, scaler_v2, le, rest_info.head(1))
        else:
            print("length of rest info",len(rest_info))
            primary_code, primary_prob, secondary_code, secondary_prob = predict_branch_code(model, text, vectorizer, le)


        
        if len(text)==0:
            final_result.append({
        "primary_code": "Others",
	     "primary_condfidence_score": "LOW",
        "secondary_code": "Others",
	     "secondary_prob": "LOW",
	"document_id": fid
              })
        else:
    
            if primary_prob>0.97:
                p_confidence = "VERY_HIGH"
            elif primary_prob>0.95:
                p_confidence = "HIGH"
            elif primary_prob<=0.95 and primary_prob>0.75:
                p_confidence = "MEDIUM"
            else:
                p_confidence = "LOW"

            if secondary_prob>0.97:
                s_confidence = "VERY_HIGH"
            elif secondary_prob>0.95:
                s_confidence = "HIGH"
            elif secondary_prob<=0.95 and secondary_prob>0.75:
                s_confidence = "MEDIUM"
            else:
                s_confidence = "LOW"
            final_result.append({
        "primary_code": primary_code,
	     "primary_score": p_confidence,
         "secondary_code": secondary_code,
	     "secondary_score": s_confidence,
	"document_id": fid
              })
            

    return {"claimid":claim_id, "icdinfo":final_result}


def get_files_text(fid, ds_pages,docBase64,log_arr_1):
    log_arr_1.append("Entered get_files_text function")
    
    base64flag = 0
    if docBase64!="":
        s= time.time()
        pages = pdf2image.convert_from_bytes(BytesIO(pybase64.b64decode(docBase64)).read())
        ds = [p for i,p in enumerate(pages) if i+1 in ds_pages]
        file_path = "img.jpeg"
        cv2.imread(ds[0].save(file_path,"jpeg"))
        print("Successfully read doc base 64")
        return
        
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
        print()
        base64flag = 1
                
        ds_text = ''
        threads = []
        
        if docBase64!="" or (base64flag == 1 and 'fileBytes' in d.keys() and '.pdf' in d['fileName'].lower()):
            s= time.time()
            pages = pdf2image.convert_from_bytes(BytesIO(pybase64.b64decode(docBase64)).read())
            ds = [p for i,p in enumerate(pages) if i+1 in ds_pages]
            file_path = "img.jpeg"
            
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



@app.get("/healthcheckup")
def wrapper3(response:Response):
    response.headers["Cache-Control"] = "no-store,no-cache"
    return "API working beautifully"

@app.get("/cache/refresh")
def wrapper4(response:Response):
    response.headers["Cache-Control"] = "no-store,no-cache"
    return "Cache updated successfully."




if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8085)
    
