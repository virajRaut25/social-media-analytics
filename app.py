from flask import Flask, render_template, request, redirect
import pandas as pd

app = Flask(__name__)
dataset = pd.read_csv("static/data/dataset.csv")

@app.route('/')
def home():
    ratio_wing = dataset['wing'].value_counts().reindex(['left', 'right'])
    left_perct = round((ratio_wing['left']/len(dataset))*100)
    right_perct = round((ratio_wing['right']/len(dataset))*100)
    return render_template('index.html', left_perct=left_perct, right_perct=right_perct)

if __name__ == "__main__":
    app.run(debug=True)