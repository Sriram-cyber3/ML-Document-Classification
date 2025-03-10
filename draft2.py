import os
import PyPDF2
from collections import Counter
import pandas as pd
import win32com.client
import nltk
import re
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split        
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def mow(text):
    words = text.lower().split()
    word_counts = Counter(words)
    most_common_word = word_counts.most_common(1)[0][0]
    return most_common_word

def convert_doc_to_pdf(doc_path, pdf_path):
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(doc_path)
    doc.SaveAs(pdf_path, FileFormat=17)
    doc.Close()
    word.Quit()

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

folder_path = r"C:\Users\pegas\OneDrive\Desktop\sample"

if os.path.exists(folder_path) and os.path.isdir(folder_path):
    print("Your data has been received, we will shortly provide you the output. Thank you.")
    et_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            sup = os.path.dirname(file_path)
            if file_path.endswith((".doc", ".docx")):
                output_path = os.path.splitext(file_path)[0] + '.pdf'
                convert_doc_to_pdf(file_path, output_path)
                file_path = output_path
            reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            et_data.append({'cleaned_text': text, 'folder': sup, 'Most Occuring word': mow(text)})
    df = pd.DataFrame(et_data)
    df.to_excel("example_pandas.xlsx", index=False)

    tfidfconverter = TfidfVectorizer(max_features=1000)
    X = pd.DataFrame(tfidfconverter.fit_transform(df['cleaned_text']).toarray())

    encoder = LabelEncoder()
    y = encoder.fit_transform(df['folder'])

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Unique classes in y_test:", set(y_test))
    print("Unique classes in y_pred:", set(y_pred))
    print("Classes in encoder:", encoder.classes_)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

    # Save the model
    pickle.dump(model, open('nbmodel.pkl', 'wb'))

    # Load the model
    savedmodel = pickle.load(open('nbmodel.pkl', 'rb'))

else:
    print("Invalid folder path. Please try again.")

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