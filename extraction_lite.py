import argparse

parser = argparse.ArgumentParser()                                               
parser.add_argument("--path", "-p", type=str, required=True)
args = parser.parse_args()

import os
rname = []
jname = []
dire = args.path       #'/Users/cheril/Downloads/Data'    #your directory path here
for path, subdirs, files in os.walk(dire):
    for name in files:
        if 'Resumes' in os.path.join(path, name):
            if '.pdf'in os.path.join(path, name) or '.doc' in os.path.join(path, name):
                    rname.append(os.path.join(path, name))
        if 'JobDesc' in os.path.join(path, name):
            if '.pdf'in os.path.join(path, name) or '.doc' in os.path.join(path, name):
                    jname.append(os.path.join(path, name))

import torch
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

import re
import fitz
import docx2txt

jobd = []
for i in jname:
    if '.doc' in i and '$' not in i:
        text = docx2txt.process(i)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        jobd.append(text)
    elif '.pdf' in i:
        text = ''
        DIGITIZED_FILE = i
        with fitz.open(DIGITIZED_FILE) as doc:
            for page in doc:
                text += page.get_text()
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            #ptext += text
            jobd.append(text)

import re

resd = []
for i in rname:
    if '.doc' in i:
        text = docx2txt.process(i)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        resd.append(text)
    elif '.pdf' in i:
        text = ''
        DIGITIZED_FILE = i
        with fitz.open(DIGITIZED_FILE) as doc:
            for page in doc:
                text += page.get_text()
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            #ptext += text
            resd.append(text)

import pandas as pd
df = pd.DataFrame()
df['Name'] = [rname[i][rname[i].rfind("/")+1:] for i in range(len(rname))]
df['Text'] = resd

df.to_csv('Resume_Data_lite.csv',index=False)

import pandas as pd
dfa = pd.DataFrame()
dfa['Name'] = [jname[i][jname[i].rfind("/")+1:] for i in range(len(jname))]
dfa['Text'] = jobd

dfa.to_csv('Jobs_Data_lite.csv',index=False)

print('Extraction Complete')