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
import sys, random, gc
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def sample(params):
  """Sample a configuration from param space"""

  config = {}

  for param, value in params.items():
    if hasattr(value, 'rvs'):
      # this is a scipy.stats distribution
      config[param] = value.rvs()
    else:
      # this is a tuple
      config[param] = random.choice(value)

  return config

def run(
  make_model,      # function that returns a keras model
  make_model_args, # dict with make_model arguments
  param_space,     # dict with hyperparameter values
  x_train,         # training examples
  y_train,         # training labels
  x_val,           # validation examples
  y_val,           # validation labels
  n):              # number of iterations
  """Random search"""

  # configurations and their scores
  config2score = {}

  for i in range(n):

    # prevent OOM errors
    gc.collect()
    bke.clear_session()

    config = sample(param_space)
    args = config.copy()
    args.update(make_model_args)
    model = make_model(args)
    print('[%d] %s' % (i + 1, config))

    erstop = EarlyStopping(
      monitor='val_loss',
      min_delta=0,
      patience=2,
      restore_best_weights=True)

    model.fit(
      x_train,
      y_train,
      validation_data=(x_val, y_val),
      epochs=args['epochs'],
      batch_size=args['batch'],
      verbose=0,
      callbacks=[erstop])

    predictions = model.predict_classes(x_val)
    f1 = f1_score(y_val, predictions, average='macro')
    config2score[tuple(config.items())] = f1
    print('[%d] score: %.3f' % (i + 1, f1))

  return config2score

if __name__ == "__main__":

  pass
