import pandas as pd
import os
import argparse

Resume = pd.read_csv('Resume_Data_lite.csv')
Jobs = pd.read_csv('Jobs_Data_lite.csv')

Jobs = Jobs.fillna('')
Resume = Resume.fillna('')

import torch
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)

parser = argparse.ArgumentParser()                                               
parser.add_argument("--index", "-i", type=int, required=True)
args = parser.parse_args()

index = int(args.index)   #Select the index of the Job Description you want
query = Jobs['Text'].iloc[index]

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model = model.to(device)

#Compute embedding for both lists
resume_embedding = model.encode(Resume['Text'].to_list(), convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)
whole_ranks = util.cos_sim(query_embedding, resume_embedding)[0].detach().cpu().numpy()
#whole_ranks[query] = cos_scores.detach().cpu().numpy()

Resume['Scores'] = whole_ranks

Ranked_resumes = Resume.sort_values(
    by=['Scores'], ascending=False).reset_index(drop=True)

Ranked_resumes['Rank'] = pd.DataFrame(
    [i for i in range(1, len(Ranked_resumes['Scores'])+1)])

Ranked_resumes.to_csv('Ranked_resumes.csv',index=False)

print('Scoring Finished, CSV named "Ranked_resumes.csv" created!')