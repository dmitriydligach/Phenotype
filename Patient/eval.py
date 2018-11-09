#!/usr/bin/env python3

# reproducible results
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
from keras import backend as bke
s = tf.Session(graph=tf.get_default_graph())
bke.set_session(s)

# the rest of the imports
import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True

import configparser
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import keras as k
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model
import dataset

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def grid_search(x, y, scoring):
  """Find best model and fit it"""

  param_grid = {
    'penalty': ['l1', 'l2'],
    'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
  lr = LogisticRegression(class_weight='balanced')
  gs = GridSearchCV(lr, param_grid, scoring=scoring, cv=10)
  gs.fit(x, y)

  return gs.best_estimator_

def run_eval(x_train, y_train, x_test, y_test, search=True):
  """Evaluation on test set"""

  if search:
    classifier = grid_search(x_train, y_train, 'f1_macro')
  else:
    classifier = LogisticRegression(class_weight='balanced')
    model = classifier.fit(x_train, y_train)

  predictions = classifier.predict(x_test)
  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

  probs = classifier.predict_proba(x_test)
  roc_auc = roc_auc_score(y_test, probs[:, 1])
  accuracy = accuracy_score(y_test, predictions)
  print("auc: %.3f - accuracy: %.3f" % (roc_auc, accuracy))

def data_dense():
  """Data to feed into code prediction model"""

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  test_dir = os.path.join(base, cfg.get('data', 'test'))

  # load pre-trained model
  rl = cfg.get('data', 'rep_layer')
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer(rl).output)
  maxlen = model.get_layer(name='EL').get_config()['input_length']

  # load target task training data
  dataset_provider = dataset.DatasetProvider(
    train_dir,
    cfg.get('data', 'alphabet_pickle'))
  x_train, y_train = dataset_provider.load_keras(maxlen=maxlen)
  x_train = pad_sequences(x_train, maxlen=maxlen)

  # make training vectors for target task
  print('x_train shape (original):', x_train.shape)
  x_train = interm_layer_model.predict(x_train)
  print('x_train shape (new):', x_train.shape)

  # now load the test set
  dataset_provider = dataset.DatasetProvider(
    test_dir,
    cfg.get('data', 'alphabet_pickle'))
  x_test, y_test = dataset_provider.load_keras(maxlen=maxlen)
  x_test = pad_sequences(x_test, maxlen=maxlen)

  # make test vectors for target task
  x_test = interm_layer_model.predict(x_test)

  return x_train, y_train, x_test, y_test

def data_sparse():
  """Bag-of-cuis data for sparse evaluation"""

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  test_dir = os.path.join(base, cfg.get('data', 'test'))

  # load training data
  dataset_provider = dataset.DatasetProvider(train_dir)
  x_train, y_train = dataset_provider.load_sklearn()

  # load test data
  dataset_provider = dataset.DatasetProvider(test_dir)
  x_test, y_test = dataset_provider.load_sklearn()

  # turn xs into tfidf vectors
  vectorizer = TfidfVectorizer()
  x_train = vectorizer.fit_transform(x_train)
  x_test = vectorizer.transform(x_test)

  return x_train.toarray(), y_train, x_test.toarray(), y_test

def data_hybrid():
  """Concatenate dense and sparse vectors and eval"""

  x_train_sparse, y_train, x_test_sparse, y_test = data_sparse()
  x_train_dense, y_train, x_test_dense, y_test = data_dense()

  x_train = np.concatenate((x_train_dense, x_train_sparse), axis=1)
  x_test = np.concatenate((x_test_dense, x_test_sparse), axis=1)

  return x_train, y_train, x_test, y_test

def main():
  """Driver function"""

  if cfg.get('data', 'rep') == 'sparse':
    x_train, y_train, x_test, y_test = data_sparse()
  elif cfg.get('data', 'rep') == 'dense':
    x_train, y_train, x_test, y_test = data_dense()
  elif cfg.get('data', 'rep') == 'hybrid':
    x_train, y_train, x_test, y_test = data_hybrid()
  else:
    exit()

  run_eval(x_train, y_train, x_test, y_test)
if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  main()
