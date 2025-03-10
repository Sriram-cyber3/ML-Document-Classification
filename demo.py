import pickle 
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import PyPDF2

   
 # Load the model
savedmodel = pickle.load(open('nbmodel.pkl', 'rb'))
encoder = LabelEncoder()
tfidfconverter = TfidfVectorizer(max_features=1000)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Demo data processing
demof = r"C:\Users\pegas\OneDrive\Desktop\sample\Instructions_on_contracts_and_agreement.pdf"
text = extract_text_from_pdf(demof)
text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)
words = word_tokenize(text)
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word) for word in words]
text_processed = ' '.join(words)
dfdemo = pd.DataFrame([text_processed], columns=['Data'])
inputs = pd.DataFrame(tfidfconverter.transform(dfdemo['Data']).toarray())
output_category = savedmodel.predict(inputs)
# Decode the predicted label to get the category name
output_category_name = encoder.inverse_transform(output_category)
print(output_category_name)
