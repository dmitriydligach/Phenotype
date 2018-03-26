#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../../Neural/Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os, random
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import keras as k
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.models import load_model
import dataset, word2vec
from random_search import RandomSearch

RESULTS_FILE = 'Model/results.txt'
MODEL_FILE = 'Model/model.h5'

class CodePredictionModel:

  def __init__(self):
    """Configuration parameters"""

    self.configs = {};

    self.configs['batch'] = (32, 64, 128, 256)
    self.configs['hidden'] = (100, 500, 1000, 3000)

  def get_random_config(self):
    """Random training configuration"""

    config = {};

    config['batch'] = random.choice(self.configs['batch'])
    config['hidden'] = random.choice(self.configs['hidden'])

    return config

  def get_model(self, init_vectors, vocab_size, input_length, output_units, config):
    """Model definition"""

    print 'vocab size:', vocab_size
    print 'input length:', input_length
    print 'classes:', output_units
    print 'configuration:', config

    cfg = ConfigParser.ConfigParser()
    cfg.read(sys.argv[1])

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=cfg.getint('dan', 'embdims'),
                        input_length=input_length,
                        trainable=True,
                        weights=init_vectors,
                        name='EL'))
    model.add(GlobalAveragePooling1D(name='AL'))

    model.add(Dense(config['hidden'], name='HL'))
    model.add(Activation('relu'))

    model.add(Dense(output_units))
    model.add(Activation('sigmoid'))

    return model

  def run_one_eval(self, train_x, train_y, valid_x, valid_y, epochs, config):
    """A single eval"""

    init_vectors = None
    vocab_size = train_x.max() + 1
    input_length = max([len(seq) for seq in x])
    output_units = train_y.shape[1]

    model = self.get_model(init_vectors,
                           vocab_size,
                           input_length,
                           output_units,
                           config)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=cfg.getfloat('dan', 'learnrt')),
                  metrics=['accuracy'])
    model.fit(train_x,
              train_y,
              epochs=epochs,
              batch_size=config['batch'],
              validation_split=0.0)

    # probability for each class; (test size, num of classes)
    distribution = model.predict(test_x, batch_size=cfg.getint('dan', 'batch'))

    # turn into an indicator matrix
    distribution[distribution < 0.5] = 0
    distribution[distribution >= 0.5] = 1

    f1 = f1_score(test_y, distribution, average='macro')
    print 'configuration:', config
    print 'epochs:', epochs
    print 'f1:', f1
    print
    
    return f1

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])

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

  maxlen = max([len(seq) for seq in x])
  x = pad_sequences(x, maxlen=maxlen)
  y = np.array(y)

  model = CodePredictionModel()
  search = RandomSearch(model, x, y)
  best_config = search.optimize()

  model = get_model(cfg, init_vectors, len(dataset.token2int))
