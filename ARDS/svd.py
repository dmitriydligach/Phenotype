#!/usr/bin/env python

import sys, pickle
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import os, numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

train_path = '/Users/Dima/Loyola/Data/Ards/Train/'
test_path = '/Users/Dima/Loyola/Data/Ards/Test/'

def load(path):
  """Assume each subdir is a separate class"""

  labels = []    # string labels
  examples = []  # examples as strings

  for sub_dir in os.listdir(path):
    sub_dir_path = os.path.join(path, sub_dir)
    for f in os.listdir(sub_dir_path):
      file_path = os.path.join(path, sub_dir, f)
      text = open(file_path).read().rstrip()
      examples.append(text)
      labels.append(sub_dir)

  return examples, labels

def roc(positive_class='Yes'):
  """Get ROC curve"""

  train_examples, train_labels = load(train_path)
  test_examples, test_labels = load(test_path)

  labelEncoder = LabelEncoder()
  labelEncoder.fit(train_labels)
  y_train = labelEncoder.transform(train_labels)
  y_test = labelEncoder.transform(test_labels)
  pos_class_ind = labelEncoder.transform([positive_class])[0]

  # load tfidf vectorizer trained on mimic
  vectorizer = pickle.load(open('../Svd/Model/tfidf.p', 'rb'))
  x_train = vectorizer.transform(train_examples)
  x_test = vectorizer.transform(test_examples)

  # load svd model and map train/test to low dimensions
  print 'input shape:', x_train.shape
  svd = pickle.load(open('../Svd/Model/svd.p', 'rb'))
  x_train = svd.transform(x_train)
  x_test = svd.transform(x_test)
  print 'output shape:', x_train.shape

  classifier = LogisticRegression(class_weight='balanced')
  model = classifier.fit(x_train, y_train)
  predicted = classifier.predict_proba(x_test)

  roc_auc = roc_auc_score(y_test, predicted[:, pos_class_ind])

  print 'roc auc:', roc_auc

if __name__ == "__main__":

  roc()
