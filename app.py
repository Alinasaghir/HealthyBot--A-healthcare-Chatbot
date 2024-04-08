#import libraries
import random
import re
from word2number import w2n
from textblob import TextBlob
from markupsafe import escape
from sklearn.ensemble import RandomForestClassifier
from flask import jsonify
import secrets
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, session
from flask_sqlalchemy import SQLAlchemy
import os 

# Configuration and setup
# Get the absolute path of the current directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Define the path for the SQLite database file
db_file_path = os.path.join(base_dir, 'new_database.sql')

# Create a Flask web application
app = Flask(__name__)

# Configure the SQLAlchemy database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_file_path

# Initialize the SQLAlchemy database object
db = SQLAlchemy(app)

# Define utility functions
 # A function to extract age information from text using regular expressions and word-to-number conversion
def extract_age_from_text(text):
    try:
        # Try to extract numeric values using regular expression
        pattern = r'\d+'
        result = re.findall(pattern, text)
        
        if result:
            return float(result[0])

        # If no numeric values are found, try to convert text to a numeric value
        return float(w2n.word_to_num(text))
    except ValueError:
        return "Please give a valid age"

 # Analyze sentiment of feedback using TextBlob
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    # Classify the sentiment as positive, negative, or neutral
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

def make_token():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16) 
 
# Define User Model
class user(db.Model):
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120))
    password = db.Column(db.String(80))

# Define routes and views
@app.route("/")
def index():
    return render_template("index.html")

userSession = {}

@app.route("/user")
def index_auth():
    my_id = make_token()
    userSession[my_id] = -1
    return render_template("index_auth.html",sessionId=my_id)

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")

# Main login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            return redirect(url_for("index_auth"))
    return render_template("login.html")

# Main registration route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        register = user(username=uname, email=mail, password=passw)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")

import re

all_result = {
    'name':'',
    'age':0,
    'symptoms':[]
}

# Import Dependencies
import pandas as pd
import numpy as np
from joblib import load
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def predict_symptom(user_input, symptom_list):
    # Convert user input to lowercase and split into tokens
    user_input_tokens = user_input.lower().replace("_"," ","and",".").split()
    negation_words =['not', 'no', 'without', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'cannot' ]

    negated_user_input_tokens = [token for token in user_input_tokens if token not in negation_words]

    # Calculate cosine similarity between user input and each symptom
    similarity_scores = []
    for symptom in symptom_list:
        symptom_tokens = symptom.lower().replace("_", " ").split()

        count_vector = np.zeros((2, len(set(negated_user_input_tokens + symptom_tokens))))

        for i, token in enumerate(set(negated_user_input_tokens + symptom_tokens)):
            count_vector[0][i] = negated_user_input_tokens.count(token)
            count_vector[1][i] = symptom_tokens.count(token)

        similarity = cosine_similarity(count_vector)[0][1]
        similarity_scores.append(similarity)

    # Return symptom with highest similarity score
    max_score_index = np.argmax(similarity_scores)
    return symptom_list[max_score_index]

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset into a pandas dataframe
df = pd.read_excel('dataset.xlsx')

# Get all unique symptoms
symptoms = set()

for s in df['Symptoms']:
    for symptom in s.split(','):
        symptoms.add(symptom.strip())

def predict_disease_from_symptom(symptom_list):
    user_symptoms = symptom_list
    # Vectorize symptoms using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Symptoms'])
    user_X = vectorizer.transform([', '.join(user_symptoms)])

    # Compute cosine similarity between user symptoms and dataset symptoms
    similarity_scores = cosine_similarity(X, user_X)

    # Find the most similar disease(s)
    max_score = similarity_scores.max()
    max_indices = similarity_scores.argmax(axis=0)
    diseases = set()
    for i in max_indices:
        if similarity_scores[i] == max_score:
            diseases.add(df.iloc[i]['Disease'])

    # Output results
    if len(diseases) == 0:
        return "<b>No matching diseases found</b>",""
    elif len(diseases) == 1:
        print("The most likely disease is:", list(diseases)[0])
        disease_details = getDiseaseInfo(list(diseases)[0])
        return f"<b>{list(diseases)[0]}</b><br>{disease_details}",list(diseases)[0]
    else:
        return "The most likely diseases are<br><b>"+ ', '.join(list(diseases))+"</b>",""

    symptoms = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
                'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
                'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
                'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
                'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
                'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
                'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
                'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0, 'mild_fever': 0,
                'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0, 'swelling_of_stomach': 0,
                'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0, 'throat_irritation': 0,
                'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0, 'weakness_in_limbs': 0,
                'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
                'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0, 'swollen_legs': 0,
                'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0, 'swollen_extremeties': 0,
                'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
                'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0, 'movement_stiffness': 0,
                'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0, 'loss_of_smell': 0,
                'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0, 'internal_itching': 0,
                'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
                'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
                'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
                'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
                'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0, 'fluid_overload.1': 0,
                'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
                'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
                'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0}
    
    # Set value to 1 for corresponding symptoms
    
    for s in symptom_list:
        index = predict_symptom(s, list(symptoms.keys()))
        print('User Input: ',s," Index: ",index)
        symptoms[index] = 1
    
    # Put all data in a test dataset
    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))
    print(df_test.head()) 
    # Load pre-trained model
    clf = load(str("model/random_forest.joblib"))
    result = clf.predict(df_test)
    disease_details = getDiseaseInfo(result[0])
    # Cleanup
    del df_test
    
    return f"<b>{result[0]}</b><br>{disease_details}",result[0]

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get all unique diseases
diseases = set(df['Disease'])

def get_symtoms(user_disease):
    # Vectorize diseases using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([user_disease])

    # Compute cosine similarity between user disease and dataset diseases
    similarity_scores = cosine_similarity(X, user_X)

    # Find the most similar disease(s)
    max_score = similarity_scores.max()
    print(max_score)
    if max_score < 0.7:
        print("No matching diseases found")
        return False,"No matching diseases found"
    else:
        max_indices = similarity_scores.argmax(axis=0)
        symptoms = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                symptoms.update(set(df.iloc[i]['Symptoms'].split(',')))
        # Output results

        print("The symptoms of", user_disease, "are:")
        for sym in symptoms:
            print(str(sym).capitalize())

        return True,symptoms

from duckduckgo_search import DDGS

def getDiseaseInfo(keywords):
    results = list(DDGS().text(keywords))
    return results[0]['body']

# Disease prediction route
@app.route('/ask',methods=['GET','POST'])
def chat_msg():

    user_message = request.args["message"].lower()
    sessionId = request.args["sessionId"]

    rand_num = random.randint(0,4)
    response = []
    if request.args["message"]=="undefined":

        response.append("<b><center> WELCOME TO HEALTHY BOT BUDDY</b>")
        response.append("What is your good name?")
        return jsonify({'status': 'OK', 'answer': response})
    else:

        currentState = userSession.get(sessionId)

        if currentState ==-1:
            response.append("Hi "+user_message+",Please be honest with us and provide required information accordingly.")
            userSession[sessionId] = userSession.get(sessionId) +1
            all_result['name'] = user_message            

        if currentState==0:
            username = all_result['name']
            response.append(username+", what is your age?")
            userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==1:
        # Extract age from the user message
            extracted_age = extract_age_from_text(user_message)

            try:
                if 0 < extracted_age < 130:
                    all_result['age'] = extracted_age
                    username = all_result['name']
                    response.append(username + ", Choose Option ?")
                    response.append("1. Predict Disease")
                    response.append("2. Check Disease Symptoms")
                    userSession[sessionId] = userSession.get(sessionId) + 1
                else:
                    response.append("Invalid input. Please provide a valid age.")
            except:
                response.append("Invalid input. Please provide a valid age.")

        if currentState == 2:
            if '2' in user_message.lower() or 'check' in user_message.lower():
                username = all_result['name']
                response.append(username + ", What's Disease Name?")
                userSession[sessionId] = 6
            else:
                # Handle the case when the user chooses option 1
                username = all_result['name']
                response.append(username + ", What symptoms are you experiencing?")
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
                userSession[sessionId] = userSession.get(sessionId) + 1


        if currentState==3:
            all_result['symptoms'].extend(user_message.split(","))
            username = all_result['name']
            response.append(username+"If you want to add more symptoms, please press anything other than 1 or enter your symptoms as a response:")            
            response.append("1. Check Disease")   
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>') 
            userSession[sessionId]=4

        if currentState==4:
            if '1' in user_message.lower() or 'predict' in user_message.lower():
                disease,type = predict_disease_from_symptom(all_result['symptoms'])
                response.append("<b>The following disease may be causing your discomfort</b>")
                response.append(disease)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                all_result['symptoms'] = []
                userSession[sessionId] = 5
            else:
                new_symptoms = user_message.split(",")
                for symptom in new_symptoms:
                    if symptom not in all_result['symptoms']:
                        all_result['symptoms'].append(symptom.strip())
                username = all_result['name']    
                disease,type = predict_disease_from_symptom(all_result['symptoms'])
                print(all_result['symptoms'])
                response.append("<b>The following disease may be causing your discomfort</b>")
                response.append(disease)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                all_result['symptoms'] = []
                userSession[sessionId] = 5
                
        if currentState==5:
            response.append("")
            response.append("Do you want to get started again?")
            userSession[sessionId] = 7

        if currentState==6:

            result,data = get_symtoms(user_message)
            if result:
                response.append(f"The symptoms of {user_message} are")
                for sym in data:
                    if sym.lower() != user_message.lower():
                        response.append(sym.capitalize())
                response.append("")
                response.append("Do you want to get started again?")
                userSession[sessionId] = 7

            else:response.append(data)
        if currentState==7:
            if "yes" in user_message:
                response.append("Choose Option ?")            
                response.append("1. Predict Disease")
                response.append("2. Check Disease Symtoms")
                userSession[sessionId] = 2
            else:
                response.append("Please provide feedback:")
                userSession[sessionId]=8
        if currentState==8:
            feedback_sentiment = analyze_sentiment(user_message)
            if feedback_sentiment == "positive":
                response.append("Thank you for your positive feedback! We're glad we could assist you.")
            elif feedback_sentiment == "negative":
                response.append("We're sorry to hear about your experience. Please let us know how we can improve.")
            elif feedback_sentiment =="neutral":
                response.append("Thank you for your feedback.")
            else:
                response.append("See you soon,"+username)

        return jsonify({'status': 'OK', 'answer': response})

# Run the application
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False, port=3000)

 