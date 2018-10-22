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
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers.core import Dense
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import keras as k
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
import dataset

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def make_model(C):
  """Model definition"""

  # load pretrained code prediction model
  rl = cfg.get('data', 'rep_layer')
  pretrained_model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(inputs=pretrained_model.input,
                             outputs=pretrained_model.get_layer(rl).output)

  # add logistic regression layer
  model = Sequential()
  model.add(interm_layer_model)
  model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(C)))

  model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

  return model

def get_maxlen():
  """Obtain max sequence length from saved model"""

  pretrained_model = load_model(cfg.get('data', 'model_file'))
  return pretrained_model.get_layer(name='EL').get_config()['input_length']

def fine_tune(C=1):
  """Fine tuning dense vectors"""

  # load target task train and test data
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  test_dir = os.path.join(base, cfg.get('data', 'test'))

  maxlen = get_maxlen()

  dataset_provider = dataset.DatasetProvider(
    train_dir,
    cfg.get('data', 'alphabet_pickle'))
  x_train, y_train = dataset_provider.load_keras(maxlen=maxlen)
  x_train = pad_sequences(x_train, maxlen=maxlen)

  dataset_provider = dataset.DatasetProvider(
    test_dir,
    cfg.get('data', 'alphabet_pickle'))
  x_test, y_test = dataset_provider.load_keras(maxlen=maxlen)
  x_test = pad_sequences(x_test, maxlen=maxlen)

  # train and evaluate
  model = make_model(C)
  model.fit(x_train, y_train, epochs=3, validation_split=0.0)

  predictions = model.predict_classes(x_test)
  probs = model.predict(x_test)

  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

  accuracy = accuracy_score(y_test, predictions)
  roc_auc = roc_auc_score(y_test, probs)
  print("auc: %.3f - accuracy: %.3f" % (roc_auc, accuracy))

def grid_search():
  """Grid search using sklearn API"""

  # load target task train and test data
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  test_dir = os.path.join(base, cfg.get('data', 'test'))

  maxlen = get_maxlen()
  dataset_provider = dataset.DatasetProvider(
    train_dir,
    cfg.get('data', 'alphabet_pickle'))
  x_train, y_train = dataset_provider.load_keras(maxlen=maxlen)
  x_train = pad_sequences(x_train, maxlen=maxlen)

  classifier = KerasClassifier(make_model)

  param_grid = {
    'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'epochs':[3, 5, 7]}

  validator = GridSearchCV(
    classifier,
    param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=1)

  validator.fit(x_train, y_train)

  print('best param:', validator.best_params_)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  grid_search()
  # fine_tune(0.0001)
