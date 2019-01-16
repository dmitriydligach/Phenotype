#!/usr/bin/env python3

import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.dont_write_bytecode = True
import configparser, pickle, gc
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras import backend as K
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop, SGD
from scikit_learn import FixedKerasClassifier
from scipy.stats import uniform
from scipy.stats import randint
from keras import regularizers
from dataset import DatasetProvider
import i2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def get_data(disease, judgement):
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

  # load training data
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
  x_train = pad_sequences(x_train, maxlen=get_maxlen())

  # load the test set
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
  x_test = pad_sequences(x_test, maxlen=get_maxlen())

  return x_train, y_train, x_test, y_test

def get_maxlen():
  """Obtain max sequence length from saved model"""

  pretrained_model = load_model(cfg.get('data', 'model_file'))
  return pretrained_model.get_layer(name='EL').get_config()['input_length']

def make_model(
  output_classes=3,
  dropout=0.25,
  lr=0.001):
  """Model definition"""

  gc.collect()
  K.clear_session()

  # load pretrained code prediction model
  rl = cfg.get('data', 'rep_layer')
  pretrained_model = load_model(cfg.get('data', 'model_file'))
  base_model = Model(inputs=pretrained_model.input,
                     outputs=pretrained_model.get_layer(rl).output)

  # freeze pretrained weights
  for layer in base_model.layers:
    layer.trainable = False

  # add logistic regression layer
  model = Sequential()
  model.add(base_model)
  model.add(Dropout(dropout))
  model.add(Dense(output_classes, activation='softmax'))

  model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=RMSprop(lr=lr),
    metrics=['accuracy'])

  return model

def thaw_layer(model, layer_name):
  """Make specified layer trainable"""

  for layer in model.layers:
    if type(layer) == Model: # base model
      for base_layer in layer.layers:
        if base_layer.name == layer_name:
          base_layer.trainable = True

def list_layers(model):
  """List layers and show if they are trainable"""

  for layer in model.layers:
    if type(layer) == Model:
      for base_layer in layer.layers:
        print('%s: %s' % (base_layer.name, base_layer.trainable))
    else:
      print('%s: %s' % (layer.name, layer.trainable))

def run_evaluation(disease, judgement):
  """Use pre-trained patient representations"""

  print('disease:', disease)
  x_train, y_train, x_test, y_test = get_data(disease, judgement)
  print('x_train shape:', x_train.shape)
  print('x_test shape:', x_test.shape)
  print('classes:', len(set(y_train)))

  classifier = FixedKerasClassifier(
    build_fn=make_model,
    output_classes=len(set(y_train)),
    verbose=0)

  param_space = {
    'dropout': uniform(0, 0.75),
    'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'epochs': randint(3, 75),
    'batch_size': [2, 4, 8, 16, 32]}

  validator = RandomizedSearchCV(
    classifier,
    param_space,
    n_iter=cfg.getint('data', 'n_iter'),
    scoring='f1_macro',
    refit=False,
    cv=cfg.getint('data', 'cv'),
    verbose=cfg.getint('data', 'verbose'))
  validator.fit(x_train, y_train)
  print('best params:', validator.best_params_)

  # train with best params and evaluate
  model = make_model(
    output_classes=len(set(y_train)),
    dropout=validator.best_params_['dropout'],
    lr=validator.best_params_['lr'])
  model.fit(
    x_train,
    y_train,
    epochs=validator.best_params_['epochs'],
    batch_size=validator.best_params_['batch_size'],
    verbose=0)

  distribution = model.predict(x_test)
  predictions = np.argmax(distribution, axis=1)
  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

  # unfreeze a base model layer
  print()
  thaw_layer(model, 'HL')
  list_layers(model)

  # now fine-tune the model
  model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=SGD(lr=1e-5, momentum=0.9),
    metrics=['accuracy'])
  model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=2,
    verbose=0)

  distribution = model.predict(x_test)
  predictions = np.argmax(distribution, axis=1)
  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))
  print()

  return p, r, f1

def run_evaluation_all_diseases():
  """Evaluate classifier performance for all 16 comorbidities"""

  base = os.environ['DATA_ROOT']
  judgement = cfg.get('data', 'judgement')
  evaluation = cfg.get('data', 'evaluation')
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  ps = []; rs = []; f1s = []
  for disease in i2b2.get_disease_names(test_annot, set()):
    p, r, f1 = run_evaluation(disease, judgement)
    ps.append(p)
    rs.append(r)
    f1s.append(f1)

  print('average p = %.3f' % np.mean(ps))
  print('average r = %.3f' % np.mean(rs))
  print('average f1 = %.3f' % np.mean(f1s))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  run_evaluation_all_diseases()
