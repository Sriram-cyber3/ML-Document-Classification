from prepro import extract_text_from_pdf, preprocess, extract_text_word
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the saved model, tfidfconverter and the labelencoder
with open(r"C:\Users\pegas\OneDrive\Desktop\project\nbmodel.pkl", "rb") as f:
    savedmodel4 = pickle.load(f)
with open(r"C:\Users\pegas\OneDrive\Desktop\project\tfidfconverter.pkl", "rb") as f1:
    tfidfconverter = pickle.load(f1)
encoder = pickle.load(open(r"C:\Users\pegas\OneDrive\Desktop\project\labelencoder.pkl", "rb"))

# Demo data processing
demof = r"C:\Users\pegas\OneDrive\Desktop\sample\PSA.pdf"
if demof.endswith(".pdf"):
    text = extract_text_from_pdf(demof)
elif demof.endswith((".doc",".docx")):
    text = extract_text_word(demof)
words = preprocess(text)
text_processed = ' '.join(words)
dfdemo = pd.DataFrame([text_processed], columns=['Data'])
X = pd.DataFrame(tfidfconverter.transform(dfdemo['Data']).toarray())
output_category = savedmodel4.predict(X)
output_category_name = encoder.inverse_transform(output_category)   # Convert predicted numerical labels back into original category names
print(output_category_name)