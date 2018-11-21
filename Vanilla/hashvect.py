#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import os, numpy
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

train_path = '/Users/Dima/Loyola/Data/Opioids/Train'
test_path = '/Users/Dima/Loyola/Data/Opioids/Test'

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

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

def grid_search(x, y, scoring='f1_macro'):
  """Find best model and fit it"""

  param_grid = {
    'penalty': ['l1', 'l2'],
    'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
  lr = LogisticRegression(class_weight='balanced')
  gs = GridSearchCV(lr, param_grid, scoring=scoring, cv=10)
  gs.fit(x, y)

  return gs.best_estimator_

def f1(use_hash_vect=True):
  """Train SVM and compute p, r, and f1"""

  train_x, train_y = load(train_path)
  test_x, test_y = load(test_path)

  if use_hash_vect:
    vectorizer = HashingVectorizer(n_features=50000)
    train_x = vectorizer.transform(train_x)
    test_x = vectorizer.transform(test_x)
  else:
    vectorizer = TfidfVectorizer()
    train_x = vectorizer.fit_transform(train_x)
    test_x = vectorizer.transform(test_x)
    print('vocabulary size:', len(vectorizer.vocabulary_))

  classifier = grid_search(train_x, train_y)
  predicted = classifier.predict(test_x)

  p = precision_score(test_y, predicted, average='macro')
  r = recall_score(test_y, predicted, average='macro')
  f1 = f1_score(test_y, predicted, average='macro')

  print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

if __name__ == "__main__":

  f1()
