import os
import pdfplumber
import pandas as pd

def get_claim_ids(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

def get_pdfs(claim_id, folder_path):
    claim_path = os.path.join(folder_path, claim_id)
    return [f for f in os.listdir(claim_path) if f.endswith('.pdf')]

def extract_text_from_first_two_pages(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(2, len(pdf.pages))):
            page = pdf.pages[i]
            page_text = page.extract_text(layout=True)
            if page_text:
                text += page_text + "\n"
            else:
                tables = page.extract_tables()
                for table in tables:
                    text += "\n".join(["\t".join(row) for row in table]) + "\n"
    return text.strip()

def load_claims_data(csv_path):
    return pd.read_csv(csv_path)

def get_claim_details(claims_data, claim_id):
    claim_row = claims_data[claims_data['CLM_ID'].astype(str) == str(claim_id)]
    if not claim_row.empty:
        
        claim_icd = claim_row['ADMTNG_ICD9_DGNS_CD'].values[0]
        claim_amount = claim_row['CLM_PMT_AMT'].values[0]
        created_date = claim_row['CLM_FROM_DT'].values[0]
        return claim_icd, claim_amount, created_date
    return None, None, None
