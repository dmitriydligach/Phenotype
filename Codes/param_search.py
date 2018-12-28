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
  """Random training configuration"""

  sample = {}
  for param, values in params.items():
    sample[param] = random.choice(values)

  print('sample:', sample)
  return sample

def run(
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
    sample = sample(param_space)
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

  pass
