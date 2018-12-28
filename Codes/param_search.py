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
from sklearn.metrics import f1_score

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def sample(params):
  """Sample a configuration from param space"""

  config = {}
  for param, values in params.items():
    config[param] = random.choice(values)

  print('sample:', config)
  return config

def run(
  make_model,      # function that returns a keras model
  make_model_args, # dict with make_model arguments
  param_space,     # dict with hyperparameter values
  x_train,         # training examples
  y_train,         # training labels
  x_val,           # validation examples
  y_val,           # validation labels
  n):              # number iterations
  """Random search"""

  # configurations and their scores
  config2score = {}

  for _ in range(n):

    config = sample(param_space)
    args = config.copy()
    args.update(make_model_args)
    model = make_model(args)

    model.fit(
      x_train,
      y_train,
      validation_data=(x_val, y_val),
      epochs=args['epochs'],
      batch_size=args['batch'],
      verbose=0)

    predictions = model.predict_classes(x_val)
    f1 = f1_score(y_val, predictions, average='macro')
    config2score[config] = f1
    print("macro f1: %.3f" % f1)

  # get config with best score
  print()
  print(config2score)
  return max(config2score, key=config2score.get)

if __name__ == "__main__":

  pass
