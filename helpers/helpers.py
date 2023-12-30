import re
import pickle as pk
import tensorflow as tf
import numpy as np

# gotten with nltk.download("stopwords") and stopwords.words("english"). This is done to reduce dependency list
stopword_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
                 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
                 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
                 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
                 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
                'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
                "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', 
                "weren't", 'won', "won't", 'wouldn', "wouldn't"]

subj_enc = {'politicsnews': 0,
 'news': 1,
 'politics': 0,
 'government news': 3,
 'world news': 4,
 'us-news': 5,
 'left-news': 6,
 'middle-east': 7}


def url_agency_strip(text):
  patterns = [ "http","href", "https", "www", "Reuters"]
  for pat in patterns:
    stripped = re.sub(pat,'',text)
  return stripped

def preprocess_text(text):
    text = url_agency_strip(text)
    text = text.lower()
    text = text.replace("'", " ")
    text = text.replace("\\", " ")
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

def remove_stopwords(data_list, stopword_list = stopword_list):
    for i in range(len(data_list)):
        data_list[i] = " ".join(
            [word for word in data_list[i].split() if word not in (stopword_list)]
        )
    return data_list

def process_req(text, subject, vectorizer):
  req = []
  req.append(text)
  req= [preprocess_text(i) for i in req]
  req = remove_stopwords(req)
  req_vectorized = vectorizer(req)
  req_vectorized_f = np.array([tf.concat(
      [req_vectorized[i], np.array([subj_enc[subject]])], 0
  ) for i in range(req_vectorized.shape[0])])
  return req_vectorized_f

def load_vectorizer(f):
  vectorizer_dets = pk.load(open(f, "rb"))
  vectorizer = tf.keras.layers.TextVectorization.from_config(vectorizer_dets['config'])
  vectorizer.set_weights(vectorizer_dets['weights'])
  return vectorizer

def load_model(f):
  model = tf.keras.models.load_model(f)
  return model

