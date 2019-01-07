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

# the rest of imports
import sys, random
sys.path.append('../Lib/')
sys.path.append('../Codes')
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
from keras.callbacks import Callback
from dataset_transfer import TransferDataset
from scipy.stats import uniform
from scipy.stats import randint
import dataset, word2vec, callback, rndsearch

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

RESULTS_FILE = 'Model/results.txt'
MODEL_FILE = 'Model/model.h5'

def make_param_space():
  """Hyperparameter space"""

  params = {};

  params['batch'] = (2, 4, 8, 16, 32, 64, 128, 256)
  params['hidden'] = (512, 1024, 2048, 4096, 8192, 16384)
  params['activation'] = ('relu', 'sigmoid', 'linear')
  params['log10lr'] = uniform(-4, 3) # 1e-4, ..., 1e-1
  params['dropout'] = uniform(0, 0.5)
  params['epochs'] = randint(1, 50)

  return params

def make_model(args):
  """Model for random search"""

  model = Sequential()
  model.add(Embedding(input_dim=args['num_features'],
                      output_dim=args['emb_dims'],
                      input_length=args['seq_len'],
                      weights=args['init_vectors'],
                      name='EL'))
  model.add(GlobalAveragePooling1D(name='AL'))

  model.add(Dense(args['hidden'], name='HL'))
  model.add(Activation(args['activation']))

  model.add(Dropout(args['dropout']))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=10**args['log10lr']),
    metrics=['accuracy'])

  return model

def main():
  """Driver function"""

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  # load x and y and split
  dataset = TransferDataset(
    os.path.join(base, cfg.get('data', 'train')),
    os.path.join(base, cfg.get('data', 'codes')),
    os.path.join(base, cfg.get('data', 'targets')),
    cfg.getint('args', 'min_token_freq'),
    cfg.getint('args', 'max_tokens_in_file'),
    cfg.getint('args', 'min_examples_per_code'))
  x, y = dataset.load()
  x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2)
  max_len = max([len(seq) for seq in x_train])

  # load pretrained embeddings
  init_vectors = None
  if cfg.has_option('data', 'embed'):
    embed_file = os.path.join(base, cfg.get('data', 'embed'))
    w2v = word2vec.Model(embed_file)
    init_vectors = [w2v.select_vectors(dataset.token2int)]

  # turn x into numpy array among other things
  x_train = pad_sequences(x_train, maxlen=max_len)
  x_val = pad_sequences(x_val, maxlen=max_len)
  y_train = np.array(y_train)
  y_val = np.array(y_val)

  fixed_args = {
    'num_features': len(dataset.token2int),
    'emb_dims': cfg.getint('search', 'emb_dim'),
    'seq_len': max_len,
    'init_vectors': init_vectors
  }
  param_space = make_param_space()

  results = rndsearch.run(
    make_model,
    fixed_args,
    param_space,
    x_train,
    y_train,
    x_val,
    y_val,
    cfg.getint('search', 'n'))

  # display configs sorted by f1
  print('\nconfigurations sorted by score:')
  sorted_by_value = sorted(results, key=results.get)
  for config in sorted_by_value:
    print('%s: %.3f' % (config, results[config]))

if __name__ == "__main__":

  main()
