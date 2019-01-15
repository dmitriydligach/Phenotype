#!/usr/bin/env python3

import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.dont_write_bytecode = True
import configparser, pickle, gc, keras
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
import i2b2, rndsearch

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

def make_model(args):
  """Model definition"""

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
  model.add(Dropout(args['dropout']))
  model.add(Dense(args['output_classes'], activation='softmax'))

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

  x_train, y_train, x_test, y_test = get_data(disease, judgement)
  print('\ndisease: %s, classes: %d' % (disease, len(set(y_train))))

  fixed_args = {
    'output_classes': len(set(y_train)),
    'loss': 'sparse_categorical_crossentropy'}

  param_space = {
    'dropout': uniform(0, 0.5),
    'optimizer': ('RMSprop', 'Adam'),
    # 'log10lr': uniform(-4, 3), # 1e-4, ..., 1e-1
    'log10lr': (-5, -4, -3, -2, -1),
    'batch': (2, 4, 8, 16, 32)}

  results = rndsearch.run(
    make_model,
    fixed_args,
    param_space,
    x_train,
    y_train,
    n=cfg.getint('search', 'n'))

  # display configs sorted by f1
  print('\nconfigurations sorted by score:')
  sorted_by_value = sorted(results, key=results.get)
  for config in sorted_by_value:
    print('%s: %.3f' % (config, results[config]))

  best_config = dict(sorted_by_value[-1])
  print('best:', best_config)
  print('best score:', results[sorted_by_value[-1]])

  # train with best params and evaluate
  args = best_config.copy()
  args.update(fixed_args)
  model = make_model(args)
  optim = getattr(keras.optimizers, args['optimizer'])
  model.compile(
    loss=fixed_args['loss'],
    optimizer=optim(lr=10**args['log10lr']),
    metrics=['accuracy']
  )
  model.fit(
    x_train, y_train,
    epochs=args['epochs'],
    batch_size=args['batch'],
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

  print()
  print('average p = %.3f' % np.mean(ps))
  print('average r = %.3f' % np.mean(rs))
  print('average f1 = %.3f' % np.mean(f1s))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  run_evaluation_all_diseases()
