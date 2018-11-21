#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import os, numpy, random
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import utils

train_path = '/Users/Dima/Loyola/Data/Opioids/Train'
test_path = '/Users/Dima/Loyola/Data/Opioids/Test'

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def get_paths_to_files(path):
  """Path to training files"""

  paths = []
  for sub_dir in os.listdir(path):
    sub_dir_path = os.path.join(path, sub_dir)
    for f in os.listdir(sub_dir_path):
      file_path = os.path.join(path, sub_dir, f)
      paths.append(file_path)

  return paths

def get_minibatch(paths):
  """Simulate batch fetching"""

  texts = []
  labels = []

  for _ in range(10):
    for path in random.sample(paths, 50):

      text = open(path).read().rstrip()
      label = path.split('/')[-2]
      texts.append(text)
      labels.append(label)

    yield texts, labels

def grid_search(x, y, scoring='f1_macro'):
  """Find best model and fit it"""

  param_grid = {
    'penalty': ['l1', 'l2'],
    'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
  lr = LogisticRegression(class_weight='balanced')
  gs = GridSearchCV(lr, param_grid, scoring=scoring, cv=5)
  gs.fit(x, y)

  return gs.best_estimator_

def f1(use_hash_vect=True):
  """Train SVM and compute p, r, and f1"""

  train_x, train_y = utils.load(train_path)
  test_x, test_y = utils.load(test_path)

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

def train_and_test():
  """Train using mini-batches"""

  classes = numpy.array(['Yes', 'No'])
  all_paths = get_paths_to_files(train_path)
  vectorizer = HashingVectorizer(n_features=50000)

  for train_x, train_y in get_minibatch(all_paths):
    train_x = vectorizer.transform(train_x)
    cls = SGDClassifier(loss='log')
    cls.partial_fit(train_x, train_y, classes=classes)

  test_x, test_y = utils.load(test_path)
  test_x = vectorizer.transform(test_x)
  predicted = cls.predict(test_x)
  p = precision_score(test_y, predicted, average='macro')
  r = recall_score(test_y, predicted, average='macro')
  f1 = f1_score(test_y, predicted, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

if __name__ == "__main__":

  train_and_test()
