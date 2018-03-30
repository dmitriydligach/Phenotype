#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../../Neural/Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os, random
from sklearn.metrics import f1_score
import keras as k
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import dataset, word2vec
from random_search import RandomSearch

RESULTS_FILE = 'Model/results.txt'
MODEL_FILE = 'Model/model.h5'

class CnnCodePredictionModel:

  def __init__(self):
    """Configuration parameters"""

    self.configs = {};

    self.configs['batch'] = (32, 64, 128, 256, 512)
    self.configs['filters'] = (64, 128, 256, 1024, 2048, 4096)
    self.configs['filtlen'] = (2, 3, 4, 5)
    self.configs['dropout'] = (0, 0.25, 0.5)
    self.configs['hidden'] = (500, 1000, 5000, 10000)
    self.configs['optimizer'] = ('sgd', 'rmsprop', 'adagrad',
                                 'adadelta', 'adam', 'adamax', 'nadam')
    self.configs['activation'] = ('relu', 'tanh',
                                  'sigmoid', 'linear')
    self.configs['embed'] = (True, False)
    # self.configs['lr'] = (0.0001, 0.0005, 0.001, 0.005, 0.01)

  def get_random_config(self):
    """Random training configuration"""

    config = {};

    config['batch'] = random.choice(self.configs['batch'])
    config['filters'] = random.choice(self.configs['filters'])
    config['filtlen'] = random.choice(self.configs['filtlen'])
    config['dropout'] = random.choice(self.configs['dropout'])
    config['hidden'] = random.choice(self.configs['hidden'])
    config['optimizer'] = random.choice(self.configs['optimizer'])
    config['activation'] = random.choice(self.configs['activation'])
    config['embed'] = random.choice(self.configs['embed'])
    # config['lr'] = random.choice(self.configs['lr'])

    return config

  def get_model(self, init_vectors, vocab_size, input_length, output_units, config):
    """Model definition"""

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=300,
                        input_length=input_length,
                        trainable=True,
                        weights=init_vectors,
                        name='EL'))
    model.add(Conv1D(
      filters=config['filters'],
      kernel_size=config['filtlen'],
      activation='relu'))
    model.add(GlobalMaxPooling1D())

    model.add(Dropout(config['dropout']))
    model.add(Dense(config['hidden'], name='HL'))
    model.add(Activation(config['activation']))

    model.add(Dropout(config['dropout']))
    model.add(Dense(output_units))
    model.add(Activation('sigmoid'))

    return model

  def run_one_eval(self, train_x, train_y, valid_x, valid_y, epochs, config):
    """A single eval"""

    init_vectors = None
    if config['embed']:
      embed_file = os.path.join(base, cfg.get('data', 'embed'))
      w2v = word2vec.Model(embed_file)
      init_vectors = [w2v.select_vectors(dataset.token2int)]

    vocab_size = train_x.max() + 1
    input_length = max([len(seq) for seq in x])
    output_units = train_y.shape[1]

    model = self.get_model(init_vectors,
                           vocab_size,
                           input_length,
                           output_units,
                           config)
    model.compile(loss='binary_crossentropy',
                  optimizer=config['optimizer'],
                  metrics=['accuracy'])
    model.fit(train_x,
              train_y,
              epochs=epochs,
              batch_size=config['batch'],
              validation_split=0.0,
              verbose=0)

    # probability for each class; (test size, num of classes)
    distribution = model.predict(valid_x, batch_size=cfg.getint('dan', 'batch'))

    # turn into an indicator matrix
    distribution[distribution < 0.5] = 0
    distribution[distribution >= 0.5] = 1

    f1 = f1_score(valid_y, distribution, average='macro')
    print 'config: %s, epochs: %d, f1: %.3f' % (config, epochs, f1)

    return 1 - f1

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

  model = CnnCodePredictionModel()
  search = RandomSearch(model, x, y)
  best_config = search.optimize(max_iter=256)
  print 'best config:', best_config
