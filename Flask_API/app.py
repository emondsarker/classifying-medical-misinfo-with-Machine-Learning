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


# API ROUTE, method is post cause sending info
@app.route('/predict', methods=['POST'])
def predict():
    # initiate csv file for ENGLISH
    df = pd.read_csv("NLP.csv")

    # Features and Labels
    # X is input
    # Y is output
    X = df['Translation']
    y = df['Target']

    # Extract Feature With CountVectorizer, converts csv file params to vectors
    # model only accepts vectors

    # CV is the same as TFiDF
    cv = CountVectorizer(stop_words='english')  # remove stopword on input data
    X = cv.fit_transform(X)  # Fit the Data, aka making it suitable to data
    from sklearn.model_selection import train_test_split

    # Train test split is 80/20 therefore testsize=0.2 (20%)
    # random state is 42 (default) because when we rerun we get the same split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = SVC()  # model initialize
    clf.fit(X_train, y_train)  # fitting training to model

    # pickle file is the trained model weights from Google colab
    # we dump so the previous weights are dynamically used
    joblib.dump(clf, 'SVC_info_model.pkl')
    SVC_model = open('SVC_info_model.pkl', 'rb')  # read pickle to SVC model
    clf = joblib.load(SVC_model)  # loads model weights to clf

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
    X = df1['Text'].fillna('').apply(str)

    # Extract Feature With CountVectorizer
    cv = CountVectorizer(stop_words='english')
    X = cv.fit_transform(X)  # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB

    clf = SVC()
    clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)
    # Alternative Usage of Saved Model
    joblib.dump(clf, 'SVC_Bangla_model.pkl')
    SVC_model = open('SVC_Bangla_model.pkl', 'rb')
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
    app.run(debug=True)
