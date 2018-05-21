#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../../Neural/Lib/')
sys.dont_write_bytecode = True
import configparser, os, random
from sklearn.metrics import f1_score
import keras as k
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import dataset, word2vec
from random_search import RandomSearch

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

RESULTS_FILE = 'Model/results.txt'
MODEL_FILE = 'Model/model.h5'

class CnnCodePredictionModel:

  def __init__(self):
    """Configuration parameters"""

    self.configs = {};

    self.configs['batch'] = (8, 16, 32)
    self.configs['filters'] = (512, 1024, 2048)
    self.configs['filtlen'] = (4, 5, 6, 7, 8)
    self.configs['dropout'] = (0, 0.25, 0.5, 0.75)
    self.configs['hidden'] = (1000, 3000)
    self.configs['optimizer'] = ('adam', 'rmsprop', 'adagrad')
    self.configs['activation'] = ('relu', 'sigmoid', 'linear')
    self.configs['embed'] = (True, False)
    self.configs['layers'] = (0, 1, 2)

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
    config['layers'] = random.choice(self.configs['layers'])

    return config

  def get_model(self, init_vectors, vocab_size, input_length, output_units, config):
    """Model definition"""

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=500,
                        input_length=input_length,
                        trainable=True,
                        weights=init_vectors,
                        name='EL'))
    model.add(Conv1D(
      filters=config['filters'],
      kernel_size=config['filtlen'],
      activation='relu'))
    model.add(GlobalMaxPooling1D(name='MP'))

    for n in range(config['layers']):
      model.add(Dense(config['hidden'], name='HL%d' % (n + 1)))
      model.add(Activation(config['activation']))

    # dropout on the fully-connected layer
    model.add(Dropout(config['dropout']))

    model.add(Dense(output_units))
    model.add(Activation('sigmoid'))

    return model

  def run_one_eval(self, train_x, train_y, valid_x, valid_y, epochs, config):
    """A single eval"""

    print(config)

    init_vectors = None
    if config['embed']:
      embed_file = os.path.join(base, cfg.get('data', 'embed'))
      w2v = word2vec.Model(embed_file)
      init_vectors = [w2v.select_vectors(provider.token2int)]

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
    # print 'f1: %.3f, config: %s, epochs: %d' % (f1, config, epochs)
    print('f1: %.3f after %d epochs\n' % (f1, epochs))

    return 1 - f1

if __name__ == "__main__":

  # fyi this is a global variable now
  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  code_file = os.path.join(base, cfg.get('data', 'codes'))

  provider = dataset.DatasetProvider(
    train_dir,
    code_file,
    cfg.getint('args', 'min_token_freq'),
    cfg.getint('args', 'max_tokens_in_file'),
    cfg.getint('args', 'min_examples_per_code'),
    use_cuis=False)
  x, y = provider.load(tokens_as_set=False)

  maxlen = max([len(seq) for seq in x])
  x = pad_sequences(x, maxlen=maxlen)
  y = np.array(y)

  print('x shape:', x.shape)
  print('y shape:', y.shape)
  print('max seq len:', maxlen)
  print('vocab size:', x.max() + 1)
  print('number of features:', len(provider.token2int))
  print('number of labels:', len(provider.code2int))

  model = CnnCodePredictionModel()
  search = RandomSearch(model, x, y)
  best_config = search.optimize(max_iter=64)
  print('best config:', best_config)
