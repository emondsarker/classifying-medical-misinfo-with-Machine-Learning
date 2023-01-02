from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)
# tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
vectorizer = CountVectorizer(max_features=10000)
loaded_model = pickle.load(open('modelsvm.pkl', 'rb'))
dataframe = pd.read_csv('dataset.csv')
x = dataframe['Translation']
y = dataframe['Target']

# x_train, x_test, y_train, y_test = train_test_split(BOW, y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


def fake_news_det(user_input):

    # tfid_x_train = tfvect.fit_transform(x_train)
    # tfid_x_test = tfvect.transform(x_test)
    input_data = [user_input]
    input2 = vectorizer.fit_transform(input_data)
    # print(BOW.shape)
    # vectorized_input_data = tfvect.transform([user_input])
    # vectorized_input_data = input_data
    # vectorized_input_data2 = tfvect.fit_transform(vectorized_input_data)
    prediction = loaded_model.predict(input2)
    # return prediction
    if prediction[0] == 1:
        return "information"
    else:
        return "misinformation"


@app.route('/')
def home():
    # return render_template('index.html')
    return 'My first API call!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_json = request.get_json(force=True)
        print('input json', input_json)
        # message = request.form['message']
        pred = fake_news_det(input_json['text'])
        print(pred)
    #     return render_template('index.html', prediction=pred)
    # else:
    #     return render_template('index.html', prediction="Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)
