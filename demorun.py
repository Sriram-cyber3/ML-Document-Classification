import os
import PyPDF2
import pandas as pd
from prepro import extract_text_word, preprocess, get_directory
import fitz
import pickle

# Load the saved model, tfidfconverter and the labelencoder
with open(r"C:\Users\pegas\OneDrive\Desktop\project\rgmodel.pkl" , "rb") as f:
    savedmodel4 = pickle.load(f)
with open(r"C:\Users\pegas\OneDrive\Desktop\project\tfidfconverter.pkl", "rb") as f1:
    tfidfconverter = pickle.load(f1)
encoder = pickle.load(open(r"C:\Users\pegas\OneDrive\Desktop\project\labelencoder.pkl", "rb"))



folder_path = get_directory()  #Input

if folder_path:
    if os.path.exists(folder_path) and os.path.isdir(folder_path):       #checks if the given path is both 'valid' and a 'directory'
        et_data=[]   # DataFrame 

        for root, dir, files in os.walk(folder_path):                    # traverses a directory tree
            for file in files:                                           # iterates over the files in the directory
                file_path = os.path.join(root, file)                        
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
                    result = ' '.join(preprocessed_text)
                    df = pd.DataFrame([result], columns=['Data'])
                    X = pd.DataFrame(tfidfconverter.transform(df['Data']).toarray())
                    output_category = savedmodel4.predict(X)
                    output_category_name = encoder.inverse_transform(output_category)   # Convert predicted numerical labels back into original category names
                    et_data.append({"File": file_path,"Model's prediction":output_category_name})
        df=pd.DataFrame(et_data)
        df.to_excel("demo_run.xlsx", index=False)
        print(df)
