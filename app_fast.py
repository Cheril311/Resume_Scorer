import streamlit as st
import os
import docx2txt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer, util
from wordcloud import WordCloud
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoConfig
import re
import fitz
import docx2txt
import re
import pandas as pd

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)

st.title("Resume Matcher")

dire = "Data"
uploaded_resume = st.file_uploader("please upload Resumes", type=["pdf","docx"], accept_multiple_files=True)
if uploaded_resume is not None:
    for i in range(len(uploaded_resume)):
      file_path = os.path.join(dire,"Resumes", uploaded_resume[i].name)
      with open(file_path, "wb") as f:
             f.write(uploaded_resume[i].getbuffer())  
    #st.success("Resumes Uploaded!")

uploaded_jd = st.file_uploader("please upload JDs", type=["pdf","docx"], accept_multiple_files=True)
if uploaded_jd is not None:
        for i in range(len(uploaded_jd)):
          file_path = os.path.join(dire,"JobDesc", uploaded_jd[i].name)
          with open(file_path, "wb") as f:
                 f.write(uploaded_jd[i].getbuffer())  
        #st.success("JDs Uploaded, Starting Parsing!")
        
rname = []
jname = []  
for path, subdirs, files in os.walk(dire):
        for name in files:
            if 'Resumes' in os.path.join(path, name):
                if '.pdf'in os.path.join(path, name) or '.doc' in os.path.join(path, name):
                        rname.append(os.path.join(path, name))
            if 'JobDesc' in os.path.join(path, name):
                if '.pdf'in os.path.join(path, name) or '.doc' in os.path.join(path, name):
                        jname.append(os.path.join(path, name))

@st.cache_data
def extraction(dire,jname,rname):








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

    df = pd.DataFrame()
    df['Name'] = [rname[i][rname[i].rfind("/")+1:] for i in range(len(rname))]
    df['Text'] = resd
    df['ace'] = ace
    df['es'] = es

    #df.to_csv('Resume_Data1.csv',index=False)


    dfa = pd.DataFrame()
    dfa['Name'] = [jname[i][jname[i].rfind("/")+1:] for i in range(len(jname))]
    dfa['Text'] = jobd
    dfa['ace'] = acej
    dfa['es'] = esj

    #dfa.to_csv('Jobs_Data1.csv',index=False)
    return df,dfa

#st.write('Extraction Complete!')

Resume,Jobs = extraction(dire,jname,rname)
Resume.to_csv('rnew.csv',index=False)
Jobs.to_csv('jnew.csv',index=False)

############################### JOB DESCRIPTION CODE ######################################
# Checking for Multiple Job Descriptions
# If more than one Job Descriptions are available, it asks user to select one as well.
if len(Jobs['Name']) <= 1:
    st.write(
        "There is only 1 Job Description present. It will be used to create scores.")
else:
    st.write("There are ", len(Jobs['Name']),
             "Job Descriptions available. Please select one.")


# Asking to Print the Job Desciption Names
if len(Jobs['Name']) > 1:
    option_yn = st.selectbox(
        "Show the Job Description Names?", options=['YES', 'NO'])
    if option_yn == 'YES':
#         index = [a for a in range(len(Jobs['Name']))]
#         fig = go.Figure(data=[go.Table(header=dict(values=["Job No.", "Job Desc. Name"], line_color='darkslategray',
#                                                    ),
#                                        cells=dict(values=[index, Jobs['Name']], line_color='darkslategray',
#                                                   ))
#                               ])
#         fig.update_layout(width=700, height=400)
        st.dataframe(Jobs['Name'],700,400)


# Asking to chose the Job Description
index = st.slider("Which JD to select ? : ", 0,
                  len(Jobs['Name'])-1, 1)


option_yn = st.selectbox("Show the Job Description ?", options=['YES', 'NO'])
if option_yn == 'YES':
    st.markdown("---")
    st.markdown("### Job Description :")
#     fig = go.Figure(data=[go.Table(
#         header=dict(values=["Job Description"],
#                     fill_color='#f0a500',
#                     align='center', font=dict(color='white', size=16)),
#         cells=dict(values=[Jobs['Text'][index]],
#                    fill_color='#f4f4f4',
#                    align='left'))])

#     fig.update_layout(width=800, height=500)
    st.write(Jobs['Text'].iloc[index])
    st.markdown("---")

#@st.cache()

Jobs = Jobs.fillna('')
Resume = Resume.fillna('')


query = Jobs['Text'].iloc[index]

@st.cache_data
def scoring(df,query):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model = model.to(device)

    #Compute embedding for both lists
    resume_embedding = model.encode(Resume['Text'].to_list(), convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    whole_ranks = util.cos_sim(query_embedding, resume_embedding)[0].detach().cpu().numpy()
    #whole_ranks[query] = cos_scores.detach().cpu().numpy()

    resume_embedding = model.encode(Resume['ace'].to_list(), convert_to_tensor=True)
    query_embedding = model.encode(Jobs['ace'].iloc[index], convert_to_tensor=True)
    ace_ranks = util.cos_sim(query_embedding, resume_embedding)[0].detach().cpu().numpy()

    if Jobs['ace'].iloc[index]=='':
        for i in range(len(ace_ranks)):
                ace_ranks[i] = 0.0

    #Compute embedding for both lists
    resume_embedding = model.encode(Resume['es'].to_list(), convert_to_tensor=True)
    query_embedding = model.encode(Jobs['es'].iloc[index], convert_to_tensor=True)
    es_ranks = util.cos_sim(query_embedding, resume_embedding)[0].detach().cpu().numpy()

    if Jobs['es'].iloc[index]=='':
        for i in range(len(es_ranks)):
                es_ranks[i] = 0.0

    new_ranks = ace_ranks+es_ranks+whole_ranks

    Resume['Scores'] = new_ranks

    Ranked_resumes = Resume.sort_values(
        by=['Scores'], ascending=False).reset_index(drop=True)

    Ranked_resumes['Rank'] = pd.DataFrame(
        [i for i in range(1, len(Ranked_resumes['Scores'])+1)])

    return Ranked_resumes

Ranked_resumes = scoring(Resume,query)

st.write("Top 10 Ranked Resumes: ")
st.dataframe(Ranked_resumes[["Rank","Name","Scores"]].iloc[:10],700)

st.markdown("---")

fig2 = px.bar(Ranked_resumes,
              x=Ranked_resumes['Name'], y=Ranked_resumes['Scores'], color='Scores',
              color_continuous_scale='haline', title="Score and Rank Distribution")
# fig.update_layout(width=700, height=700)
st.write(fig2)


st.markdown("---")

option_2 = st.selectbox("Show the Best Matching Resumes?", options=[
    'YES', 'NO'])
if option_2 == 'YES':
    indx = st.slider("Which rank resume to display ?:",
                     1, Ranked_resumes.shape[0], 1)


    st.write("Displaying Resume with Rank: ", indx)
    st.markdown("---")
    st.markdown("## **Resume** ")
    value = Ranked_resumes['Text'].iloc[indx-1]


    st.write("With a Match Score of :", Ranked_resumes['Scores'].iloc[indx-1])
#     fig = go.Figure(data=[go.Table(
#         header=dict(values=["Resume"],
#                     fill_color='#f0a500',
#                     align='center', font=dict(color='white', size=16)),
#         cells=dict(values=[str(value)],
#                    fill_color='#f4f4f4',
#                    align='left'))])

#     fig.update_layout(width=800, height=1200)
#     st.write(fig)
    st.write(value)

    st.markdown("#### The Word Cloud For the Resume")
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          colormap='viridis', collocations=False,
                          min_font_size=10).generate(value)
    plt.figure(figsize=(7, 7), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)
    # st.text(df_sorted.iloc[indx-1, 1])
    st.markdown("---")
