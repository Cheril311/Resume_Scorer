# Resume_Scoring
Resume Scoring Project
This is a project to upload resume/CVs and rank them according to job description. Uses sentence-transformers to understand text, a finetuned sentence transformer model to extract information and mathematical rules to rank resumes

First run 

```
pip install -r requirements.txt
```



You can use the simple GUI interface for the task. The only thing you need to do before using GUI is create a folder named ```Data``` in the same location as this code and make two subfolders named  ```Resumes``` and ```JobDesc``` in the ```Data``` folder. Once done simply run the following code on command-line or terminal:

```
streamlit run app_fast.py
```

If you want to use a program that is faster but less accurate use:

```
streamlit run app_fast_lite.py
```


