import os
import argparse

parser = argparse.ArgumentParser()                                               
parser.add_argument("--path", "-p", type=str, required=True)
args = parser.parse_args()

rname = []
jname = []
dire = args.path        #'/Users/cheril/Downloads/Data'    #your directory path here
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

print(device)

import re
import fitz
import docx2txt

jobd = []
for i in jname:
    if '.docx' in i and '$' not in i:
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
    if '.docx' in i:
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoConfig

tokenizer = AutoTokenizer.from_pretrained("has-abi/distilBERT-finetuned-resumes-sections")

model = AutoModelForSequenceClassification.from_pretrained("has-abi/distilBERT-finetuned-resumes-sections")

ace = []
es = []
model = model.to(device)
for j in resd:
    acei = []
    esi = []
    for i in j.split('. '):
        inputs = tokenizer(i, return_tensors="pt",truncation=True).to(device)
        output = model(**inputs)
        logits = output.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        if predicted_class[0].detach().cpu().numpy() == 0 or predicted_class[0].detach().cpu().numpy() == 1 or predicted_class[0].detach().cpu().numpy() == 3:
            acei.append(i)
        if predicted_class[0].detach().cpu().numpy() == 7 or predicted_class[0].detach().cpu().numpy() == 8 or predicted_class[0].detach().cpu().numpy() == 9 or predicted_class[0].detach().cpu().numpy() == 10:
            esi.append(i)
    ace.append(''.join(acei))
    es.append(''.join(esi))

acej = []
esj = []
model = model.to(device)
for j in jobd:
    acei = []
    esi = []
    for i in j.split('. '):
        inputs = tokenizer(i, return_tensors="pt",truncation=True).to(device)
        output = model(**inputs)
        logits = output.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        if predicted_class[0].detach().cpu().numpy() == 0 or predicted_class[0].detach().cpu().numpy() == 1 or predicted_class[0].detach().cpu().numpy() == 3:
            acei.append(i)
        if predicted_class[0].detach().cpu().numpy() == 7 or predicted_class[0].detach().cpu().numpy() == 8 or predicted_class[0].detach().cpu().numpy() == 9 or predicted_class[0].detach().cpu().numpy() == 10:
            esi.append(i)
    acej.append(''.join(acei))
    esj.append(''.join(esi))

import pandas as pd
df = pd.DataFrame()
df['Name'] = [rname[i][rname[i].rfind("/")+1:] for i in range(len(rname))]
df['Text'] = resd
df['ace'] = ace
df['es'] = es

df.to_csv('Resume_Data.csv',index=False)

import pandas as pd
dfa = pd.DataFrame()
dfa['Name'] = [jname[i][jname[i].rfind("/")+1:] for i in range(len(jname))]
dfa['Text'] = jobd
dfa['ace'] = acej
dfa['es'] = esj

dfa.to_csv('Jobs_Data.csv',index=False)

print('Extraction Complete')