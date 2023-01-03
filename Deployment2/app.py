from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask_cors import CORS
from sklearn.svm import SVC

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("NLP.csv")

    # Features and Labels

    X = df['Translation']
    y = df['Target']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer(stop_words='english')
    X = cv.fit_transform(X)  # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Naive Bayes Classifier
    # from sklearn.naive_bayes import MultinomialNB

    clf = SVC()
    clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)
    # Alternative Usage of Saved Model
    joblib.dump(clf, 'SVC_info_model.pkl')
    SVC_model = open('SVC_info_model.pkl', 'rb')
    clf = joblib.load(SVC_model)

    if request.method == 'POST':
        # message = request.json['message']
        message = request.get_json(force=True)
        message = message['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return {"prediction": str(my_prediction)}


@app.route('/predict2', methods=['POST'])
def predict2():
    df1 = pd.read_csv("Bangladataset - Sheet1.csv")

    # Features and Labels


    y = df1['Target']
    X= df1['Text'].fillna('').apply(str)


    # Extract Feature With CountVectorizer
    cv = CountVectorizer(stop_words='english')
    X = cv.fit_transform(X)  # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB

    clf = SVC()
    clf.fit(X_train, y_train)
    #clf.score(X_test, y_test)
    # Alternative Usage of Saved Model
    joblib.dump(clf, 'SVC_Bangla_model.pkl')
    SVC_model = open('SVC_Bangla_model.pkl','rb')
    clf = joblib.load(SVC_model)

    if request.method == 'POST':
        # message = request.json['message']
        message = request.get_json(force=True)
        message = message['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return {"prediction": str(my_prediction)}


if __name__ == '__main__':

    app.run( debug=True, port=8000)
