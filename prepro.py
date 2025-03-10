from collections import Counter           # A dictionary subclass for counting hashable objects
import re                                 # regular expression (regex) library
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from nltk.stem import WordNetLemmatizer 
import matplotlib.pyplot as plt
import win32com.client                    # A library to interact with Windows applications
import tkinter as tk
from tkinter import filedialog            # A submodule in tkinter used for opening file/folder selection dialogs.
import os
import PyPDF2 

def preprocess(text):
    # Convert text to lowercase to ensure uniformity
    text = text.lower()
    
    # Remove special characters and numbers - `[^a-z\s]` keeps only lowercase letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)         # a replace function
    
    # Tokenize the cleaned text into individual words
    words = word_tokenize(text)
    
    # Remove stopwords (common words like "the", "is", "and") to focus on meaningful words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words to reduce them to their base form
    lemmatizer = WordNetLemmatizer()                                         # Create an instance of the lemmatizer
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def mow(words):
    # Count word frequencies
    word_counts = Counter(words)   # returns a dictionary with word frequencies
    
    # Get the top 5 most common words
    top_words = word_counts.most_common(5)  # returns a list of tuples with the top n most frequently occurring words
    return top_words

def data_vis(top_words):
    words, frequencies = zip(*top_words)  # Unzip the data into two lists
    # Create a bar chart
    plt.bar(words, frequencies, color='skyblue')
    plt.title('Word Frequencies')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show(block=False)    # The script does not pause here
    plt.pause(3)
    plt.close()

def extract_text_word(file_path):
    word = win32com.client.Dispatch("Word.Application")  # Create a Word application instance
    doc = word.Documents.Open(file_path)  # Open the Word document
    text = doc.Content.Text  # Extract the full text content of the document
    doc.Close()  # Close the document
    word.Quit()  # Quit the Word application to free system resources
    return text  # Return the extracted text

def get_directory():
    root = tk.Tk()                                                  # initializes a new Tkinter application.
    root.withdraw()                                                 # Hide the main tkinter window
    folder_path = filedialog.askdirectory(title="Select a Folder")
    folder_path = os.path.normpath(folder_path)                     # Normalize path format
    root.destroy()                                                  # close the tkinter instance properly 
    if folder_path:
        print(f"Selected folder: {folder_path}")
    else:
        print("No folder selected.")
    return folder_path

def summary_visualization(folder_stats, total_files, total_words):
    # Visualize folder-wise file count
    labels = list(folder_stats.keys())                                       # list of folder names
    file_counts = [folder_stats[label]['file_count'] for label in labels]    # list of file counts per folder
    word_counts = [folder_stats[label]['word_count'] for label in labels]    # list of word counts per folder
    page_counts = [folder_stats[label]['page_count'] for label in labels]    # list of page counts per folder

    # Bar chart for files per folder
    plt.bar(labels, file_counts, color='green')
    plt.title('Number of Files Processed Per Folder')
    plt.xlabel('Folder Name')
    plt.ylabel('File Count')
    plt.xticks(rotation=45)
    plt.tight_layout()                        # Adjust the plot to ensure everything fits without overlapping
    plt.show()
    
    # Bar chart for total words per folder
    plt.bar(labels, word_counts, color='orange')
    plt.title('Total Words Extracted Per Folder')
    plt.xlabel('Folder Name')
    plt.ylabel('Word Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Bar chart for total pages per folder
    plt.bar(labels, page_counts, color='red')
    plt.title('Total Pages Per Folder')
    plt.xlabel('Folder Name')
    plt.ylabel('Page Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Display summary statistics
    print("\n=== Summary ===")
    print(f"Total Files Processed: {total_files}")
    print(f"Total Words Extracted: {total_words}")
    #print(f"Total Pages Extracted: {total_pages}")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:          # Open the PDF file in binary read mode ('rb')
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:  
            text += page.extract_text()
    return text

