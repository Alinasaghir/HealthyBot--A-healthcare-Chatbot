from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
# Assuming df is your labeled dataset with 'Symptoms' and 'Disease' columns
df = pd.read_excel('dataset.xlsx')
# Data preprocessing
symptoms = df['Symptoms'].apply(lambda x: ', '.join(x.split(',')))
X = symptoms
y = df['Disease']

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save the model to disk
from joblib import dump
dump(clf, "model/random_forest.joblib")

# Model prediction function
def predict_disease_from_symptom(symptom_list):
    user_symptoms = symptom_list
    user_X = vectorizer.transform([', '.join(user_symptoms)])
    result = clf.predict(user_X)
    return result[0], result[0]
