import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add', methods=['POST'])
def POST():

    data = pd.read_csv('comments.csv')
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(data['tweet'])
    y = data['Toxicity']

    model = RandomForestRegressor()
    model.fit(X, y)


    #comment = input("Type your comment: ")
    comment_tfidf = tfidf.transform([request.form["comment"]])

    prediction = model.predict(comment_tfidf)

    if prediction==0:
        result="The comment is not toxic."
    else:
        result="The comment is toxic."
        
    return render_template('result.html', result={"result":result,"comment":request.form["comment"]})

@app.route('/result')
def result():
    result = request.args.get('result')
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True,port=8080)

