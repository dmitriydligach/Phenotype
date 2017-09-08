#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../../Neural/Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import keras as k
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras import regularizers
from dataset import DatasetProvider
import word2vec, i2b2

NUM_FOLDS = 5

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def print_config(cfg):
  """Print configuration settings"""

  print 'train:', cfg.get('data', 'train_data')
  if cfg.has_option('data', 'embed'):
    print 'embeddings:', cfg.get('data', 'embed')
  print 'batch:', cfg.get('nn', 'batch')
  print 'epochs:', cfg.get('nn', 'epochs')
  print 'embdims:', cfg.get('nn', 'embdims')
  print 'hidden:', cfg.get('nn', 'hidden')
  print 'regcoef', cfg.get('nn', 'regcoef')
  print 'learnrt:', cfg.get('nn', 'learnrt')

def get_model(cfg, token2int, max_input_len, classes):
  """Model definition"""

  model = Sequential()
  model.add(Embedding(input_dim=len(token2int),
                      output_dim=cfg.getint('nn', 'embdims'),
                      input_length=max_input_len,
                      trainable=True,
                      weights=get_embeddings(cfg, token2int)))
  model.add(GlobalAveragePooling1D())

  reg_coef = cfg.getfloat('nn', 'regcoef')

  model.add(Dropout(cfg.getfloat('nn', 'dropout')))
  model.add(Dense(
    units=cfg.getint('nn', 'hidden'),
    kernel_regularizer=regularizers.l2(reg_coef)))
  model.add(Activation('relu'))

  model.add(Dense(
    units=classes,
    kernel_regularizer=regularizers.l2(reg_coef)))
  model.add(Activation('sigmoid'))

  return model

def get_embeddings(cfg, token2int):
  """Initial weights for embedding layer"""

  init_vectors = None
  base = os.environ['DATA_ROOT']

  if cfg.has_option('data', 'embed'):
    embed_file = os.path.join(base, cfg.get('data', 'embed'))
    w2v = word2vec.Model(embed_file)
    init_vectors = [w2v.select_vectors(token2int)]

  return init_vectors

def run_evaluation(exclude, judgement='intuitive'):
  """Train on train set and evaluate on test set"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  print_config(cfg)
  base = os.environ['DATA_ROOT']
  train_data = os.path.join(base, cfg.get('data', 'train_data'))
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))
  test_data = os.path.join(base, cfg.get('data', 'test_data'))
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease=None,
    judgement=judgement,
    use_pickled_alphabet=False,
    min_token_freq=cfg.getint('args', 'min_token_freq'))
  x_train, y_train = train_data_provider.load_vectorized(exclude)

  classes = len(y_train[0])
  maxlen = max([len(seq) for seq in x_train])
  x_train = pad_sequences(x_train, maxlen=maxlen)
  y_train = np.array(y_train);

  # now load the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease=None,
    judgement=judgement,
    use_pickled_alphabet=True,
    min_token_freq=cfg.getint('args', 'min_token_freq'))
  x_test, y_test = test_data_provider.load() # pass maxlen
  x_test = pad_sequences(x_test, maxlen=maxlen)
  y_test = np.array(y_test)

  model = get_model(
    cfg,
    train_data_provider.token2int,
    maxlen,
    classes)
  optimizer = RMSprop(lr=cfg.getfloat('nn', 'learnrt'))
  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  model.fit(x_train,
            y_train,
            epochs=cfg.getint('nn', 'epochs'),
            batch_size=cfg.getint('nn', 'batch'),
            validation_split=0.0,
            verbose=1)


  # probability for each class; (test size, num of classes)
  distribution = model.predict(
    test_x,
    batch_size=cfg.getint('nn', 'batch'))

  # turn into an indicator matrix
  distribution[distribution < 0.5] = 0
  distribution[distribution >= 0.5] = 1

  f1 = f1_score(test_y, distribution, average='macro')
  precision = precision_score(test_y, distribution, average='macro')
  recall = recall_score(test_y, distribution, average='macro')
  print 'macro average p =', precision
  print 'macro average r =', recall
  print 'macro average f1 =', f1

if __name__ == "__main__":

  exclude = set(['GERD', 'Venous Insufficiency', 'CHF'])
  run_evaluation(exclude=exclude)
