#!/usr/bin/env python3

import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True

import configparser, os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import keras as k
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model
import dataset

def run_eval():
  """Evaluation on test set"""

  x_train, y_train, x_test, y_test = data_dense()

  classifier = LinearSVC(class_weight='balanced')
  model = classifier.fit(x_train, y_train)
  predictions = classifier.predict(x_test)
  p = precision_score(y_test, predictions, pos_label=1)
  r = recall_score(y_test, predictions, pos_label=1)
  f1 = f1_score(y_test, predictions, pos_label=1)

  print('\np = %.3f' % p)
  print('r = %.3f' % r)
  print('f1 = %.3f' % f1)

  classifier = LogisticRegression(class_weight='balanced')
  model = classifier.fit(x_train, y_train)
  predicted = classifier.predict_proba(x_test)
  roc_auc = roc_auc_score(y_test, predicted[:, 1])

  print('roc auc = %.3f' % roc_auc)

def data_dense():
  """Data to feed into code prediction model"""

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
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
  x_train, y_train = dataset_provider.load(maxlen=maxlen)
  x_train = pad_sequences(x_train, maxlen=maxlen)

  # make training vectors for target task
  print('x_train shape (original):', x_train.shape)
  x_train = interm_layer_model.predict(x_train)
  print('x_train shape (new):', x_train.shape)

  # now load the test set
  dataset_provider = dataset.DatasetProvider(
    test_dir,
    cfg.get('data', 'alphabet_pickle'))
  x_test, y_test = dataset_provider.load(maxlen=maxlen)
  x_test = pad_sequences(x_test, maxlen=maxlen)

  # make test vectors for target task
  print('x_test shape (original):', x_test.shape)
  x_test = interm_layer_model.predict(x_test)
  print('x_test shape (new):', x_test.shape)

  return x_train, y_train, x_test, y_test

def run_nfold_cv():
  """N-fold cross validation"""

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'train'))

  # load target task data
  dataset_provider = dataset.DatasetProvider(
    data_dir,
    cfg.get('data', 'alphabet_pickle'))

  x, y = dataset_provider.load()
  # pad to same maxlen as data in source model
  x = pad_sequences(x, maxlen=cfg.getint('data', 'maxlen'))
  print('x shape (original):', x.shape)

  # make vectors for target task
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer('HL').output)
  x = interm_layer_model.predict(x)
  print('x shape (new):', x.shape)

  # ready for svm train/test now

  if cfg.getfloat('data', 'test_size') == 0:
    # run n-fold cross validation
    classifier = LinearSVC(class_weight='balanced', C=0.001)
    cv_scores = cross_val_score(classifier, x, y, scoring='f1', cv=5)
    print('fold f1s:', cv_scores)
    print('average f1:', np.mean(cv_scores))
    print('standard devitation:', np.std(cv_scores))

  else:
    # randomly allocate a test set
    x_train, x_test, y_train, y_test = train_test_split(
      x,
      y,
      test_size=cfg.getfloat('data', 'test_size'),
      random_state=1)
    classifier = LinearSVC(class_weight='balanced')
    model = classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    precision = precision_score(y_test, predicted, pos_label=1)
    recall = recall_score(y_test, predicted, pos_label=1)
    f1 = f1_score(y_test, predicted, pos_label=1)
    print('p =', precision)
    print('r =', recall)
    print('f1 =', f1)

if __name__ == "__main__":

  run_eval()
