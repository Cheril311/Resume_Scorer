# Resume_Freelance
Freelance Resume Scoring Project

First run 

```
pip install -r requirements.txt
```

## There are 3 ways to run this program

### 1) GUI

You can use the simple GUI interface for the task. The only thing you need to do before using GUI is create a folder named ```Data``` in the same location as this code and make two subfolders named  ```Resumes``` and ```JobDesc``` in the ```Data``` folder. Once done simply run the following code on command-line or terminal:

```
streamlit run app.py
```

If you want to use a program that is faster but less accurate use:

```
streamlit run app_lite.py
```


### 2) Through CLI/Terminal

First run ```extraction.py``` to extract information from JDs and CVs. You have to input the path of the Data Folder in the notebook.<b> Remember that the structure of Data Folder should be as it is. It should have two subfolders named ```Resumes``` and ```JobDesc```.</b> If you want to run a faster but less accurate version, run ```extraction_lite.py```.

This can be done using:

```
python extraction.py -p folder_path
```

you should put the path name of your folder in the folder_path placeholder.

if running lite file 

```
python extraction_lite.py -p folder_path
```
once you run extraction file, its time to score your resumes.

To do this use:

```
python extraction.py -p folder_path
```

you should put the path name of your folder in the folder_path placeholder.

if running lite file 

```
python extraction_lite.py -p folder_path
```

once you run extraction file, its time to score your resumes. 

Use:

```
python scoring.py -i index
```
Here the index placeholder should be changed with index of the JD for example 1 for the JD at 1st index.

if used the extraction_lite version then

```
python scoring_lite.py -i index
```


### Using Jupyter Notebook

Run ```Extraction.ipynb``` file cell by cell to get csv files for Resume and Jobs Data. You have to input the path of the Data Folder in the notebook.<b> Remember that the structure of Data Folder should be as it is. It should have two subfolders named ```Resumes``` and ```JobDesc```.</b> If you want to run a faster but less accurate version, run ```Extraction_lite.ipynb```

Once extraction is done run the ```Resume_Scoring.ipynb``` or the ```Resume_Scoring_lite.ipynb``` depending on the type of extraction used and run the file cell by cell to obtain a CSV file of ranked resumes, remember to change the ```index``` variable to the index of job description of your choice. 
