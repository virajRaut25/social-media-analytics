import re
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem import	WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib
import joblib

matplotlib.use('agg')

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

dataset = pd.read_csv("data/dataset.csv")

all_stopwords = stopwords.words('english')
exception_stopwords = ["not", "no", "nor", "don't", "isn't", "isn"]
for word in exception_stopwords:
  all_stopwords.remove(word)

def cleanText(text, remove_words):
  tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(^| ).( |$)", " ", text)
  tweet = tweet.lower()
  tweet = tweet.split()
  lemma_function = WordNetLemmatizer()
  process = []
  for token, tag in pos_tag(tweet):
    if not token in set(remove_words):
      process.append(lemma_function.lemmatize(token, tag_map[tag[0]]))
  process = " ".join(process)
  return process

def classify(tweet):
    tweet = cleanText(tweet, all_stopwords)
    sia = SentimentIntensityAnalyzer()
    sen = sia.polarity_scores(tweet)
    s = max(zip(sen.values(), sen.keys()))[1]
    if s == "neu":
      sent = "Neutral"
    elif s == "pos":
      sent = "Positive"
    else:
      sent = "Negative"

    pickled_vectorizer = joblib.load('vectorizer.pkl')
    tweet = pickled_vectorizer.transform([tweet]).toarray()
    pickled_model = joblib.load('classifier.pkl')
    wing = ""
    if pickled_model.predict(tweet) == 0:
        wing = "Left Wing"
    else:
        wing = "Right Wing"

    return wing, sent

def isPresent(tweet, keyword):
  flag = False
  for word in keyword:
    if tweet.find(word.lower()) != -1:
      flag = True
      break
  return flag


def get_tfidf_top_features(documents,n_top=10):
  tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  stop_words='english')
  tfidf = tfidf_vectorizer.fit_transform(documents)
  importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
  tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
  return (importance[:n_top], tfidf_feature_names[importance[:n_top]])


def doAnalyze(topic):
  rm_words = all_stopwords
  rm_words.extend(["politics", "news"]) 
  topic = topic.split()
  filtered_tweet = dataset[dataset['tweet'].apply(lambda x: isPresent(x, topic))]['tweet'].values

  corpus = []
  for i in range(len(filtered_tweet)):
    corpus.append(cleanText(filtered_tweet[i], rm_words))

  msg = "success"
  try:
    Z, labels = get_tfidf_top_features(corpus,30)
  except Exception as e:
    print(e)
    msg = "insufficient_data"
    return msg

  all_words = " ".join(corpus).split()
  counted = Counter(all_words)
  word_freq = pd.DataFrame(counted.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)
  fig, axes = plt.subplots(figsize=(5, 7))
  sns.barplot(ax=axes,x='frequency',y='word',data=word_freq.head(30))
  plt.tight_layout()
  plt.savefig('static/images/analyze_frequency.png', transparent=True) 
  plt.close(fig)

  fig, axes = plt.subplots(figsize=(16, 6))
  axes.tick_params(axis='x', rotation=60, labelsize=15)
  axes.tick_params(axis='y', labelsize=15)
  b = sns.barplot(ax=axes,x='word',y='frequency',data=word_freq.head(30))
  b.set_xlabel("Word", fontsize=15)
  b.set_ylabel("Frequency", fontsize=15)
  plt.tight_layout()
  plt.savefig('static/images/analyze_frequency_c.png', transparent=True)
  plt.close(fig)

  linkage_matrix = ward(np.reshape(Z, (len(Z), 1)))

  fig, ax = plt.subplots(figsize=(5, 7)) 
  ax = dendrogram(linkage_matrix, orientation="right", labels=labels)
  plt.tight_layout()
  plt.savefig('static/images/analyze_ward_clusters.png', transparent=True) 
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(16, 6)) 
  ax = dendrogram(linkage_matrix, labels=labels)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.tight_layout()
  plt.savefig('static/images/analyze_ward_clusters_c.png', transparent=True) 
  plt.close(fig)

  return msg


if __name__ == "__main__":
  single_tweet = "support communism"
  print(classify(single_tweet))
  topic = "modi"
  res = doAnalyze(topic)
  
