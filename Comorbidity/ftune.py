#!/usr/bin/env python3

import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.dont_write_bytecode = True
import configparser, pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras import regularizers
from dataset import DatasetProvider
import i2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

FEATURE_LIST = 'Model/features.txt'
NGRAM_RANGE = (1, 1) # use unigrams for cuis
MIN_DF = 0

def make_model(cfg, interm_layer_model_trainable=True):
  """Model definition"""

  rl = cfg.get('data', 'rep_layer')
  c = cfg.getfloat('args', 'c')

  # load pretrained code prediction model
  pretrained_model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(inputs=pretrained_model.input,
                             outputs=pretrained_model.get_layer(rl).output)

  # freeze the pretrained weights if specified
  if not interm_layer_model_trainable:
    for layer in interm_layer_model.layers:
      layer.trainable = False

  # add logistic regression layer
  model = Sequential()
  model.add(interm_layer_model)
  model.add(Dense(
    1,
    activation='sigmoid',
    kernel_regularizer=regularizers.l2(c)))

  model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

  return model

def get_maxlen(cfg):
  """Obtain max sequence length from saved model"""

  pretrained_model = load_model(cfg.get('data', 'model_file'))
  return pretrained_model.get_layer(name='EL').get_config()['input_length']

def run_evaluation(cfg, disease, judgement):
  """Use pre-trained patient representations"""

  print('disease:', disease)
  x_train, y_train, x_test, y_test = get_data(cfg, disease, judgement)

  # train and evaluate
  model = make_model(cfg)
  epochs = cfg.getint('args', 'epochs')
  model.fit(x_train, y_train, epochs=epochs)
  predictions = model.predict_classes(x_test)

  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

  return p, r, f1

def get_data(cfg, disease, judgement):
  """Sequences of tokens to feed into code prediction model"""

  base = os.environ['DATA_ROOT']
  train_data = os.path.join(base, cfg.get('data', 'train_data'))
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))
  test_data = os.path.join(base, cfg.get('data', 'test_data'))
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  # determine whether to treat input tokens as a sequence or set
  if cfg.get('data', 'model_type') == 'dan':
    use_cuis = True
    tokens_as_set = True
  else:
    use_cuis = False
    tokens_as_set = False

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'),
    use_cuis=use_cuis)
  x_train, y_train = train_data_provider.load(tokens_as_set=tokens_as_set)
  x_train = pad_sequences(x_train, maxlen=get_maxlen(cfg))

  # now load the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'),
    use_cuis=use_cuis)
  x_test, y_test = test_data_provider.load(tokens_as_set=tokens_as_set)
  x_test = pad_sequences(x_test, maxlen=get_maxlen(cfg))

  return x_train, y_train, x_test, y_test

def run_evaluation_all_diseases():
  """Evaluate classifier performance for all 16 comorbidities"""

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  judgement = cfg.get('data', 'judgement')
  evaluation = cfg.get('data', 'evaluation')
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  ps = []; rs = []; f1s = []
  for disease in i2b2.get_disease_names(test_annot, set()):
    p, r, f1 = run_evaluation(cfg, disease, judgement)
    ps.append(p)
    rs.append(r)
    f1s.append(f1)

  print('average p = %.3f' % np.mean(ps))
  print('average r = %.3f' % np.mean(rs))
  print('average f1 = %.3f' % np.mean(f1s))

if __name__ == "__main__":

  run_evaluation_all_diseases()
