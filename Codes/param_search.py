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

    pass

  def sample_params(self, params):
    """Random training configuration"""

    sample = {}
    for param, values in params.items():
      sample[param] = random.choice(values)

    print('sample:', sample)
    return sample

  def run(
    self,
    make_model,
    make_model_kwargs,
    param_space,
    x_train,
    y_train,
    x_val,
    y_val,
    n_iter):
    """Driver function"""

    for _ in range(n_iter):
      sample = self.sample_params(param_space)
      kwargs = sample.copy()
      kwargs.update(make_model_kwargs)
      model = make_model(kwargs)

      model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=sample['epochs'],
        batch_size=sample['batch'],
        verbose=0)

      predictions = model.predict_classes(x_val)
      f1 = f1_score(y_val, predictions, average='macro')
      print("macro f1: %.3f" % f1)

if __name__ == "__main__":

  rs = RandomSearch()
  rs.train_and_eval()
