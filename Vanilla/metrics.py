#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import os, numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

train_path = '/Users/Dima/Loyola/Data/Alcohol/anc_notes_cuis/'
test_path = '/Users/Dima/Loyola/Data/Alcohol/anc_notes_test_cuis/'

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

def nfoldcv(metric='f1', pos_class='yes'):
  """Run n-fold cross-validation"""

  train_examples, train_labels = load(train_path)

  labelEncoder = LabelEncoder()
  labelEncoder.fit(train_labels)
  y_train = labelEncoder.transform(train_labels)
  pos_class_ind = labelEncoder.transform([pos_class])[0]

  vectorizer = TfidfVectorizer()
  x_train = vectorizer.fit_transform(train_examples)

  classifier = LogisticRegression()
  cv_scores = cross_val_score(
    classifier,
    x_train,
    y_train,
    scoring=metric,
    cv=10)
  print('%s (n-fold cv) = %.3f' % (metric, numpy.mean(cv_scores)))

def roc(pos_class='yes'):
  """Get ROC curve"""

  train_examples, train_labels = load(train_path)
  test_examples, test_labels = load(test_path)

  labelEncoder = LabelEncoder()
  labelEncoder.fit(train_labels)
  y_train = labelEncoder.transform(train_labels)
  y_test = labelEncoder.transform(test_labels)
  pos_class_ind = labelEncoder.transform([pos_class])[0]

  vectorizer = TfidfVectorizer()
  x_train = vectorizer.fit_transform(train_examples)
  x_test = vectorizer.transform(test_examples)

  classifier = LogisticRegression(class_weight='balanced', C=1)
  model = classifier.fit(x_train, y_train)
  predicted = classifier.predict_proba(x_test)

  roc_auc = roc_auc_score(y_test, predicted[:, pos_class_ind])
  print('roc auc (test) = %.3f' % roc_auc)

def f1(pos_class='yes'):
  """Train SVM and compute p, r, and f1"""

  train_examples, train_labels = load(train_path)
  test_examples, test_labels = load(test_path)

  vectorizer = TfidfVectorizer()
  train_tfidf_matrix = vectorizer.fit_transform(train_examples)
  test_tfidf_matrix = vectorizer.transform(test_examples)

  classifier = LogisticRegression(class_weight='balanced')
  model = classifier.fit(train_tfidf_matrix, train_labels)
  predicted = classifier.predict(test_tfidf_matrix)

  precision = precision_score(test_labels, predicted, pos_label=pos_class)
  recall = recall_score(test_labels, predicted, pos_label=pos_class)
  f1 = f1_score(test_labels, predicted, pos_label=pos_class)

  print('p (test) = %.3f' % precision)
  print('r (test) = %.3f' % recall)
  print('f1 (test) = %.3f' % f1)

if __name__ == "__main__":

  nfoldcv()
  f1()
  roc()
