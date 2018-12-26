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
import dataset, word2vec, callback

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

RESULTS_FILE = 'Model/results.txt'
MODEL_FILE = 'Model/model.h5'

class RandomSearch:
  """Random search for Keras"""

  def __init__(self):
    """Constructor"""

    self.params = {}; # hyperparameter space
    self.sample = {}; # sample from that space

    self.params['batch'] = (2, 4, 8, 16, 32, 64, 128, 256)
    self.params['hidden'] = (500, 1000, 5000, 10000)
    self.params['optimizer'] = ('rmsprop', 'adam', 'adamax', 'nadam')
    self.params['activation'] = ('relu', 'tanh', 'sigmoid', 'linear')
    self.params['dropout'] = (0, 0.25, 0.5, 0.75)
    self.params['epochs'] = range(0, 10)
    # self.params['embed'] = (True, False)
    # self.params['lr'] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

    self.sample_params()

  def sample_params(self):
    """Random training configuration"""

    for param, values in self.params.items():
      self.sample[param] = random.choice(values)
    print('sample:', self.sample)

  def define_model(self, init_vectors, num_features, max_len, emb_dims):
    """Model for random search"""

    model = Sequential()
    model.add(Embedding(input_dim=num_features,
                        output_dim=emb_dims,
                        input_length=max_len,
                        weights=init_vectors,
                        name='EL'))
    model.add(GlobalAveragePooling1D(name='AL'))

    model.add(Dense(self.sample['hidden'], name='HL'))
    model.add(Activation(self.sample['activation']))

    model.add(Dropout(self.sample['dropout']))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

  def report_results(self, val_y, predictions, average):
    """Report p, r, and f1"""

    p = precision_score(val_y, predictions, average=average)
    r = recall_score(val_y, predictions, average=average)
    f1 = f1_score(val_y, predictions, average=average)

    return p, r, f1

  def train_and_eval(self):
    """Driver function"""

    cfg = configparser.ConfigParser()
    cfg.read(sys.argv[1])
    base = os.environ['DATA_ROOT']

    dataset = TransferDataset(
      os.path.join(base, cfg.get('data', 'train')),
      os.path.join(base, cfg.get('data', 'codes')),
      os.path.join(base, cfg.get('data', 'targets')),
      cfg.getint('args', 'min_token_freq'),
      cfg.getint('args', 'max_tokens_in_file'),
      cfg.getint('args', 'min_examples_per_code'))
    x, y = dataset.load()
    train_x, val_x, train_y, val_y = train_test_split(
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
    train_x = pad_sequences(train_x, maxlen=maxlen)
    val_x = pad_sequences(val_x, maxlen=maxlen)
    train_y = np.array(train_y)
    val_y = np.array(val_y)

    model = self.define_model(
      init_vectors,
      len(dataset.token2int),
      maxlen,
      cfg.getint('dan', 'embdims'))
    model.compile(
      loss='binary_crossentropy',
      optimizer=self.sample['optimizer'],
      metrics=['accuracy'])
    model.fit(
      train_x,
      train_y,
      validation_data=(val_x, val_y) if val_x.shape[0]>0 else None,
      epochs=self.sample['epochs'],
      batch_size=self.sample['batch'])

    predictions = model.predict_classes(val_x)
    p, r, f1 = self.report_results(val_y, predictions, 'macro')
    print("[%s] p: %.3f - r: %.3f - f1: %.3f" % ('macro', p, r, f1))
    print

if __name__ == "__main__":

  rs = RandomSearch()
  rs.train_and_eval()
