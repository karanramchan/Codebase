import pandas as pd
from tqdm import tqdm
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
# import ocr_aws as ocr_aws
# import ocr_gcp as ocr_gcp
from operator import itemgetter
from functools import reduce
import numpy as np
import time
import traceback
# import layoutparser as lp
import cv2
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
from sklearn.metrics.pairwise import cosine_similarity
from classifier import predict_branch_code
import nltk
nltk.download('punkt')
import pytesseract
from PIL import Image
import cv2



class document_classfier:
    def __init__(self):
        pass
        

    def image_to_text(self,input_path):
       """
       A function to read text from images.
       """
       
       img = cv2.imread(input_path)
       img = cv2.imread(input_path)
       gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
       smoothed = cv2.medianBlur(gray, 5)
    
        
       enhanced = cv2.equalizeHist(smoothed)
       
        # Normalize the image
       normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

       # file_path = "normalized.jpeg"
            
       # cv2.imread(normalized.save(file_path,"jpeg"))
    
       text = pytesseract.image_to_string(normalized)

       file_path = 'tesseract.txt'
       with open(file_path, 'w') as file:
           file.write(text)
        
    
       return text.strip()
        
        
    def create_layout(self,file_path,model):
        image = cv2.imread(file_path)

        return text

    def max_cosine_similarity_with_discharge(self,text):
        
        """Check if any of the first 10 words has high cosine similarity with 'discharge'.
        Return the maximum cosine similarity."""
        
        # Process the input text
        doc = self.spacy_nlp(text)
        
        # Get the vector for the word 'discharge'
        discharge_vector = self.spacy_nlp("diagnosis").vector
        
        # Initialize maximum similarity
        max_similarity = 0
        
        # Iterate through the first 10 words or fewer
        for token in doc[:5]:
        # Skip if the token is not a valid word (e.g., punctuation)
            if not token.has_vector:
                continue
            # Compute cosine similarity
            similarity = cosine_similarity(np.array([token.vector]), np.array([discharge_vector]))
            # Update maximum similarity if necessary
            if similarity > max_similarity:
                max_similarity = similarity
        
        return max_similarity


    def discharge_block_probab(self, text, keywords=["discharge summary","diagnosis","discharge notes"]):
        # Tokenize the text block
        doc = self.spacy_nlp(text)
        
        # Calculate word vectors for the text block
        text_vector = doc.vector
        
        # Calculate word vectors for each keyword
        keyword_vectors = [self.spacy_nlp(keyword).vector for keyword in keywords]
        
        # Calculate cosine similarity between text vector and each keyword vector
        similarities = [cosine_similarity(np.array([text_vector]), np.array([keyword_vector]))[0][0] for keyword_vector in keyword_vectors]
        
        # Sort similarities in descending order and take top three
        top_three_similarities = sorted(similarities, reverse=True)[:3]
        
        # Calculate average similarity of top three words
        average_similarity = np.mean(top_three_similarities)
        
        return average_similarity

    def extract_entities(self,text_blocks,threshold=0.6):
        all_ents = []
        all_text = []
        ds_ents = []
        ds_text = []
        max_score = -1
        # tokenized_text = nltk.sent_tokenize(text_blocks)
        tokenized_text = text_blocks.split('\n')
        # for i in tokenized_text:
        #     print(i)
        #     print("-------------------------")
        #     print('\n')
        for txt in tokenized_text:
            doc = self.nlp(txt)
            med_ents = list(set([ents.text for ents in doc.ents if ents.type in ["PROBLEM"]]))
            all_ents.extend(med_ents)
            all_text.append(txt)
            ds_score =  float(self.max_cosine_similarity_with_discharge(txt))
            if (ds_score>max_score) & (len(med_ents)>=1):
                max_score = ds_score
                ds_ents = med_ents
                ds_text = txt
           
        # print("MAX SCORE------>",max_score,ds_text,ds_ents)
        # if max_score>threshold:
            
        #     return ds_ents,ds_text, max_score
            
        return all_ents, all_text,ds_ents, ds_text, max_score


    def extract_codes(self,all_ents,ds_ents):
        all_codes = {}
        for ents in set(all_ents):
            results = self.collection.query(
                        query_texts=str(ents),
                        n_results=1
                            )
            all_codes[ents] = results["ids"][0][0]
        ds_codes = {}
        for ents in set(ds_ents):
            results = self.collection.query(
                        query_texts=str(ents),
                        n_results=1
                            )
            ds_codes[ents] = results["ids"][0][0]
        return all_codes, ds_codes

    def detected_icds(self, detected_icds):
        detected_codes = []
        
        for code in detected_icds.values():
             detected_codes.append(code[:3])
        return detected_codes

    def main_func(self,file_path):
        # text_blocks = self.create_layout(file_path,self.model)
        text_blocks = self.image_to_text(file_path)
        print("text_block_extraction_done")
        return  text_blocks
   