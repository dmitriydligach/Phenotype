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

def get_minibatch(paths, iterations=100, batch_size=10):
  """Simulate batch fetching"""

  texts = []
  labels = []

  for _ in range(iterations):
    for path in random.sample(paths, batch_size):
      text = open(path).read().rstrip()
      label = path.split('/')[-2]
      texts.append(text)
      labels.append(label)

    yield texts, labels

def train_and_test():
  """Train using mini-batches"""

  classes = numpy.array(['Yes', 'No'])
  all_paths = get_paths_to_files(train_path)

  classifier = SGDClassifier(loss='log')
  vectorizer = HashingVectorizer(n_features=25000)

  test_x, test_y = utils.load(test_path)
  test_x = vectorizer.transform(test_x)

  for train_x, train_y in get_minibatch(all_paths):

    train_x = vectorizer.transform(train_x)
    classifier.partial_fit(train_x, train_y, classes=classes)
    predicted = classifier.predict(test_x)

    p = precision_score(test_y, predicted, average='macro')
    r = recall_score(test_y, predicted, average='macro')
    f1 = f1_score(test_y, predicted, average='macro')

    print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

if __name__ == "__main__":

  train_and_test()
