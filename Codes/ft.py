#!/usr/bin/env python3

# reproducible results
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(1337)
rn.seed(27)
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
from keras import backend as k
s = tf.Session(graph=tf.get_default_graph())
k.set_session(s)

# the rest of imports
import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import configparser
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import keras as k
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.models import load_model
import dataset, word2vec

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

RESULTS_FILE = 'Model/results.txt'
MODEL_FILE = 'Model/model.h5'

from keras.callbacks import Callback
from sklearn.metrics import f1_score

class Metrics(Callback):

  def on_epoch_end(self, epoch, logs={}):
    """Compute f1 score"""

    # this is what we passed to fit()
    valid_x = self.validation_data[0]
    valid_y = self.validation_data[1]

    # probability for each class (test size, num of classes)
    distribution = self.model.predict(valid_x)

    # turn into an indicator matrix
    distribution[distribution < 0.5] = 0
    distribution[distribution >= 0.5] = 1

    p = precision_score(test_y, distribution, average='macro')
    r = recall_score(test_y, distribution, average='macro')
    f1 = f1_score(valid_y, distribution, average='macro')
    print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))

def print_config(cfg):
  """Print configuration settings"""

  print('train:', cfg.get('data', 'train'))
  if cfg.has_option('data', 'embed'):
    print('embeddings:', cfg.get('data', 'embed'))
  print('test_size', cfg.getfloat('args', 'test_size'))
  print('batch:', cfg.get('dan', 'batch'))
  print('epochs:', cfg.get('dan', 'epochs'))
  print('embdims:', cfg.get('dan', 'embdims'))
  print('hidden:', cfg.get('dan', 'hidden'))
  print('learnrt:', cfg.get('dan', 'learnrt'))

def get_model(cfg, init_vectors, num_of_features):
  """Model definition"""

  model = Sequential()
  model.add(Embedding(input_dim=num_of_features,
                      output_dim=cfg.getint('dan', 'embdims'),
                      input_length=maxlen,
                      trainable=True,
                      weights=init_vectors,
                      name='EL'))
  model.add(GlobalAveragePooling1D(name='AL'))

  model.add(Dense(cfg.getint('dan', 'hidden'), name='HL'))
  model.add(Activation(cfg.get('dan', 'activation')))

  model.add(Dropout(cfg.getfloat('dan', 'dropout')))

  model.add(Dense(classes))
  model.add(Activation('sigmoid'))

  return model

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  print_config(cfg)

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  code_file = os.path.join(base, cfg.get('data', 'codes'))

  dataset = dataset.DatasetProvider(
    train_dir,
    code_file,
    cfg.getint('args', 'min_token_freq'),
    cfg.getint('args', 'max_tokens_in_file'),
    cfg.getint('args', 'min_examples_per_code'))
  x, y = dataset.load()
  train_x, test_x, train_y, test_y = train_test_split(
    x,
    y,
    test_size=cfg.getfloat('args', 'test_size'))
  maxlen = max([len(seq) for seq in train_x])

  init_vectors = None
  if cfg.has_option('data', 'embed'):
    embed_file = os.path.join(base, cfg.get('data', 'embed'))
    w2v = word2vec.Model(embed_file)
    init_vectors = [w2v.select_vectors(dataset.token2int)]

  # turn x into numpy array among other things
  classes = len(dataset.code2int)
  train_x = pad_sequences(train_x, maxlen=maxlen)
  test_x = pad_sequences(test_x, maxlen=maxlen)
  train_y = np.array(train_y)
  test_y = np.array(test_y)

  print('train_x shape:', train_x.shape)
  print('train_y shape:', train_y.shape)
  print('test_x shape:', test_x.shape)
  print('test_y shape:', test_y.shape)
  print('number of features:', len(dataset.token2int))
  print('number of labels:', len(dataset.code2int))

  if cfg.has_option('dan', 'optimizer'):
    optimizer = cfg.get('dan', 'optimizer')
  else:
    optimizer = RMSprop(lr=cfg.getfloat('dan', 'learnrt'))

  model = get_model(cfg, init_vectors, len(dataset.token2int))
  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  model.fit(train_x,
            train_y,
            callbacks=[Metrics()] if test_x.shape[0]>0 else None,
            validation_data=(test_x, test_y) if test_x.shape[0]>0 else None,
            epochs=cfg.getint('dan', 'epochs'),
            batch_size=cfg.getint('dan', 'batch'),
            validation_split=0.0)

  model.save(MODEL_FILE)

  # do we need to evaluate?
  if cfg.getfloat('args', 'test_size') == 0:
    exit()

  # probability for each class; (test size, num of classes)
  distribution = model.predict(test_x, batch_size=cfg.getint('dan', 'batch'))

  # turn into an indicator matrix
  distribution[distribution < 0.5] = 0
  distribution[distribution >= 0.5] = 1

  f1 = f1_score(test_y, distribution, average='macro')
  precision = precision_score(test_y, distribution, average='macro')
  recall = recall_score(test_y, distribution, average='macro')
  print('macro average p =', precision)
  print('macro average r =', recall)
  print('macro average f1 =', f1)

  outf1 = open(RESULTS_FILE, 'w')
  int2code = dict((value, key) for key, value in list(dataset.code2int.items()))
  f1_scores = f1_score(test_y, distribution, average=None)
  outf1.write("%s|%s\n" % ('macro', f1))
  for index, f1 in enumerate(f1_scores):
    outf1.write("%s|%s\n" % (int2code[index], f1))
