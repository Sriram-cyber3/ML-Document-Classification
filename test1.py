#-------------------------------------------------------------------------------
#prepro
#test1
#train_model
#demo_run

import os
import PyPDF2
import pandas as pd
import win32com.client
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import fitz  # PyMuPDF - for handling PDFs
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split        
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from prepro import preprocess, mow, get_directory, summary_visualization, extract_text_word

folder_path = get_directory()  #Input

def main(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):       #checks if the given path is both 'valid' and a 'directory'
        print("Your data has been recieved, we will shortly provide you the output. Thank you.")
        et_data=[]   # DataFrame 

        total_files = 0
        total_words = 0
        folder_stats = defaultdict(lambda: {'file_count': 0, 'word_count': 0, 'page_count': 0})

        for root, dir, files in os.walk(folder_path):                    # traverses a directory tree
            for file in files:                                           # iterates over the files in the directory
                file_path = os.path.join(root, file)                        
                folder_name = os.path.basename(os.path.dirname(file_path))  
                file_name = os.path.basename(file_path)                     
                folder_and_file = os.path.join(folder_name, file_name)      
                print(f"Processing file: {folder_and_file}")
                # extracts text from WORD files
                if file_path.endswith((".doc",".docx")):
                    text = extract_text_word(file_path)
                # extracts text from PDF files
                elif file_path.endswith(".pdf"):
                    reader = PyPDF2.PdfReader(file_path)                  # Initialize a PdfReader object for the PDF file
                    text = ""
                    contains_images=False         
                    for page_num, page in enumerate(reader.pages):        
                        page_text = page.extract_text()               
                        if page_text.strip():                      
                            text += page_text              
                        else:     
                            # Check if the page contains images
                            pdf_page = fitz.open(file_path)[page_num]
                            if pdf_page.get_images(full=True):
                                contains_images = True
                                break
                    
                if text.strip():  # If there's any text content
                    preprocessed_text = preprocess(text)
                    top_words=mow(preprocessed_text)
                    first_five_words = [word for word, _ in top_words[:5]]  # returns a list of the first five words
                    result = ' '.join(preprocessed_text)
                    w5 = ', '.join(first_five_words)
                    et_data.append({'Extracted content': result, 'Label':folder_name, 'Most Occuring word': w5})
                    # Update stats
                    num_words = len(preprocessed_text)
                    total_files += 1
                    total_words += num_words
                    pages = len(reader.pages)
                    folder_stats[folder_name]['file_count'] += 1
                    folder_stats[folder_name]['word_count'] += num_words
                    folder_stats[folder_name]['page_count'] += pages
                    #data_vis(top_words)
        df=pd.DataFrame(et_data)
        df.to_excel("example_pandas.xlsx", index=False)
    
        # Visualize summary
        summary_visualization(folder_stats, total_files, total_words)
    return df

df=main(folder_path)


