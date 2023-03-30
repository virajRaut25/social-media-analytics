from flask import Flask, render_template, request
import pandas as pd
from model import classify, doAnalyze

app = Flask(__name__)
dataset = pd.read_csv("data/dataset.csv")

@app.route('/')
def home():
    ratio_wing = dataset['wing'].value_counts().reindex(['left', 'right'])
    left_perct = round((ratio_wing['left']/len(dataset))*100)
    right_perct = round((ratio_wing['right']/len(dataset))*100)
    return render_template('index.html', left_perct=left_perct, right_perct=right_perct)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    wing = None
    sentiment = None
    if request.method == "POST":
        tweet = request.form['wttweet']
        wing, sentiment = classify(tweet)

    return render_template('predict.html', wing = wing, sentiment = sentiment)

@app.route('/analyze', methods=["GET", "POST"])
def analyze():
    msg = None
    if request.method == "POST":
        topic = request.form['topic']
        msg = doAnalyze(topic)

    return render_template('analyze.html', msg = msg)

if __name__ == "__main__":
    app.run(debug=True)