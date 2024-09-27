CONFIG_PATH = 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config'
OCR_THRESHOLD = 0.1
NLP_MODEL = "en_core_sci_lg"
VD_PATH ="db"
LABEL_MAP = {0: "Text", 1: "Title"}
IMAGE_PATH = "img.jpeg"
DATA_PATH = 'Discharge_summary_tesing_1.csv'
CLASSES_TO_KEEP = ['C50', 'H25', 'K80', 'M17', 'N18', 'N20', 'O82', 'Others']
document_download_config = { 
    "URL": "http://coredms.mediassist.in/DownloadFile",
    "header": {"mbappurl":"http://coredms.mediassist.in/","mbappid":"5386"}
}