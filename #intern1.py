import os
import PyPDF2
from collections import Counter
import pandas as pd
import win32com.client


def mow(text):
    # Convert text to lowercase and split by whitespace to get words
    words = text.lower().split()
    
    # Count the frequency of each word using Counter
    word_counts = Counter(words)
    
    # Find the most common word and its count
    most_common_word = word_counts.most_common(1)[0][0]
    return most_common_word

def convert_doc_to_pdf(doc_path, pdf_path):
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False  # Optionally, hide the Word window
    doc = word.Documents.Open(doc_path)  # Open the document (Important! Without this, we can't interact with the document)
    doc.SaveAs(pdf_path, FileFormat=17)  # 17 is the PDF format # Save as PDF (this will work only after opening the document)
    doc.Close()  # Close the document and quit Word
    word.Quit()
    
folder_path = input("Enter the folder path: ")

if os.path.exists(folder_path) and os.path.isdir(folder_path):
    print("Your data has been recieved, we will shortly provide you the output. Thank you.")
    et_data=[]
    for root, dir, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            sup=os.path.dirname(file_path)
            if file_path.endswith((".doc",".docx")):
                output_path = os.path.splitext(file_path)[0] + '.pdf'
                convert_doc_to_pdf(file_path, output_path)
                file_path = output_path          
            reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            et_data.append({'Extracted content':text, 'Label':sup, 'Most Occuring word': mow(text)})
    df=pd.DataFrame(et_data)
    df.to_excel("example_pandas.xlsx", index=False)
else:
    print("Invalid folder path. Please try again.")
