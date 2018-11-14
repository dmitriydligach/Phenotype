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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from dataset import DatasetProvider
import i2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def make_model(
  c,
  dropout,
  lr,
  output_classes,
  code_model_trainable=False):
  """Model definition"""

  # load pretrained code prediction model
  rl = cfg.get('data', 'rep_layer')
  pretrained_model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(inputs=pretrained_model.input,
                             outputs=pretrained_model.get_layer(rl).output)

  # freeze the pretrained weights if specified
  for layer in interm_layer_model.layers:
    layer.trainable = code_model_trainable

  # add logistic regression layer
  model = Sequential()
  model.add(interm_layer_model)
  model.add(Dropout(dropout))
  model.add(Dense(
    output_classes,
    activation='softmax',
    kernel_regularizer=regularizers.l2(c)))

  model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=RMSprop(lr=lr),
    metrics=['accuracy'])

  return model

def get_maxlen():
  """Obtain max sequence length from saved model"""

  pretrained_model = load_model(cfg.get('data', 'model_file'))
  return pretrained_model.get_layer(name='EL').get_config()['input_length']

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
  x_train = pad_sequences(x_train, maxlen=get_maxlen())

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
  x_test = pad_sequences(x_test, maxlen=get_maxlen())

  return x_train, y_train, x_test, y_test

def run_evaluation(disease, judgement):
  """Use pre-trained patient representations"""

  print('disease:', disease)
  x_train, y_train, x_test, y_test = get_data(disease, judgement)
  num_classes = len(set(y_train))

  # run a grid search
  classifier = KerasClassifier(
    build_fn=make_model,
    output_classes=num_classes,
    batch_size=2,
    verbose=0)

  # TODO: add batch size?
  param_grid = {
    'c':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'dropout':[0.0, 0.25, 0.5],
    'lr':[0.001, 0.01],
    'epochs':[3, 5, 10, 15, 20, 25]}

  if cfg.get('data', 'search') == 'grid':
    # run a grid search
    validator = GridSearchCV(
      classifier,
      param_grid,
      scoring='f1_macro',
      refit=False,
      cv=2,
      n_jobs=1)

    validator.fit(x_train, y_train, batch_size=2)
    c = validator.best_params_['c']
    dropout = validator.best_params_['dropout']
    lr = validator.best_params_['lr']
    epochs = validator.best_params_['epochs']

  elif cfg.get('data', 'search') == 'random':
    # run a random search
    validator = RandomizedSearchCV(
      classifier,
      param_grid,
      n_iter=10,
      scoring='f1_macro',
      refit=False,
      n_jobs=1,
      cv=2)

    validator.fit(x_train, y_train, batch_size=2)
    c = validator.best_params_['c']
    dropout = validator.best_params_['dropout']
    lr = validator.best_params_['lr']
    epochs = validator.best_params_['epochs']

  else:
    # use default hyperparameters
    c = 1
    dropout = 0.25
    lr = 0.001
    epochs = 10

  # train with best params and evaluate
  model = make_model(c, dropout, lr, num_classes)
  model.fit(
    x_train,
    y_train,
    batch_size=2,
    epochs=epochs,
    verbose=0)
  distribution = model.predict(x_test)
  predictions = np.argmax(distribution, axis=1)

  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

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
