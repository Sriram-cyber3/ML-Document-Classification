import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
import xgboost
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split        
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer

from test1 import df


my_tags=df['Label'].unique()                         # gets the list of unique folder names

# TF-IDF Vectorization
tfidfconverter = TfidfVectorizer(max_features=1000)  # TF-IDF (Term Frequency-Inverse Document Frequency) measures the importance of a word in a document relative to a collection of documents (corpus)
X = pd.DataFrame(tfidfconverter.fit_transform(df['Extracted content']).toarray())  
pickle.dump(tfidfconverter, open('tfidfconverter.pkl', 'wb'))

# Label Encoding
encoder = LabelEncoder()                             # Initialize the LabelEncoder
y = encoder.fit_transform(df['Label'])               # Convert categorical values into numerical values
pickle.dump(encoder, open('labelencoder.pkl', 'wb')) 

# Random Over-sampling
ros = RandomOverSampler(random_state=42)             # setting random state to ensure reproducibility
X_resampled, y_resampled = ros.fit_resample(X, y)    # applying oversampling

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)  

results = {}

def nvb(X_train, X_test, y_train, y_test):
    #Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred, target_names=my_tags, zero_division=0))
    # Confusion Matrix for Naive Bayes
    cm_nb = confusion_matrix(y_test, y_pred)
    print("Naive Bayes Confusion Matrix:\n", cm_nb)
    # Visualize Confusion Matrix for Naive Bayes
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Naive Bayes Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Calculate metrics for Naive Bayes
    results['Naive Bayes'] = {"Accuracy": accuracy_score(y_test, y_pred), "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0), 
    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)}   # F1 score is the Harmonic Mean of Precision & Recall

    # Save the model
    with open("nbmodel.pkl", "wb") as f:
        pickle.dump(model, f)

def xgb(X_train, X_test, y_train, y_test):
    #Xgboost
    model2 = xgboost.XGBClassifier()
    model2.fit(X_train, y_train)

    # Predict on test data
    y_pred2 = model2.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred2))
    print("Classification Report:\n", classification_report(y_test, y_pred2, target_names=my_tags, zero_division=0))
    # Confusion Matrix for XGBoost
    cm_xgb = confusion_matrix(y_test, y_pred2)
    print("XGBoost Confusion Matrix:\n", cm_xgb)
    # Visualize Confusion Matrix for XGBoost
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Calculate metrics for XGBoost
    results['Xgboost'] = {"Accuracy": accuracy_score(y_test, y_pred2), "Precision": precision_score(y_test, y_pred2, average="weighted", zero_division=0), 
    "Recall": recall_score(y_test, y_pred2, average="weighted", zero_division=0),
    "F1-Score": f1_score(y_test, y_pred2, average="weighted", zero_division=0)}

    pickle.dump(model2, open('xgbmodel.pkl','wb'))

def rf(X_train, X_test, y_train, y_test):
    # Initialize the Random Forest model
    model3 = RandomForestClassifier(n_estimators=1200, random_state=42)  

    # Train the model
    model3.fit(X_train, y_train)
    # Predict on test data
    y_pred3 = model3.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred3))
    print("Classification Report:\n", classification_report(y_test, y_pred3, target_names=my_tags, zero_division=0))
    # Confusion Matrix for random forest
    cm_rf = confusion_matrix(y_test, y_pred3)
    print("Random Forest Confusion Matrix:\n", cm_rf)
    # Visualize Confusion Matrix for random forest
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Calculate metrics for Random Forest
    results['Random forest'] = {"Accuracy": accuracy_score(y_test, y_pred3), "Precision": precision_score(y_test, y_pred3, average="weighted", zero_division=0), 
    "Recall": recall_score(y_test, y_pred3, average="weighted", zero_division=0),
    "F1-Score": f1_score(y_test, y_pred3, average="weighted", zero_division=0)}

    pickle.dump(model3, open('rgmodel.pkl','wb'))

nvb(X_train, X_test, y_train, y_test)
xgb(X_train, X_test, y_train, y_test)
rf(X_train, X_test, y_train, y_test)

# Create a DataFrame from the results dictionary
results_df = pd.DataFrame(results).T  

# Display the comparison table
print(results_df)