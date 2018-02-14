#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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

def run_experiment():
  """Split into train and test and run an experiment"""

  train_examples, train_labels = load(train_path)
  test_examples, test_labels = load(test_path)

  vectorizer = TfidfVectorizer()
  train_tfidf_matrix = vectorizer.fit_transform(train_examples)
  test_tfidf_matrix = vectorizer.transform(test_examples)

  classifier = LinearSVC(class_weight='balanced')
  model = classifier.fit(train_tfidf_matrix, train_labels)
  predicted = classifier.predict(test_tfidf_matrix)

  precision = precision_score(test_labels, predicted, pos_label='Yes')
  recall = recall_score(test_labels, predicted, pos_label='Yes')
  f1 = f1_score(test_labels, predicted, pos_label='Yes')
  print 'p =', precision
  print 'r =', recall
  print 'f1 =', f1

if __name__ == "__main__":

  run_experiment()
