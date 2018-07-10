#!/usr/bin/env python3

import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.dont_write_bytecode = True
import configparser, pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model
from dataset import DatasetProvider
import i2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

FEATURE_LIST = 'Model/features.txt'
NGRAM_RANGE = (1, 1) # use unigrams for cuis
MIN_DF = 0

def grid_search(x, y):
  """Find best model"""

  param_grid = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
  lr = LinearSVC(class_weight='balanced')
  grid_search = GridSearchCV(
    lr,
    param_grid,
    scoring='f1_micro',
    cv=10,
    n_jobs=-1)
  grid_search.fit(x, y)

  return grid_search.best_estimator_

def run_evaluation(disease, judgement):
  """Use pre-trained patient representations"""

  print('disease:', disease)

  x_train_sparse, y_train_sparse, x_test_sparse, y_test_sparse = \
    get_sparse_data(disease, judgement)

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_data = os.path.join(base, cfg.get('data', 'train_data'))
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))
  test_data = os.path.join(base, cfg.get('data', 'test_data'))
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  # load pre-trained model
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(
    inputs=model.input,
    outputs=model.get_layer('HL').output)
  maxlen = model.get_layer(name='EL').get_config()['input_length']

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'))
  x_train_dense, y_train_dense = train_data_provider.load()
  x_train_dense = pad_sequences(x_train_dense, maxlen=maxlen)
  x_train_dense = interm_layer_model.predict(x_train_dense)

  # now load the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'))
  x_test_dense, y_test_dense = test_data_provider.load()
  x_test_dense = pad_sequences(x_test_dense, maxlen=maxlen)
  x_test_dense = interm_layer_model.predict(x_test_dense)

  if x_train_dense.shape[0] != x_train_sparse.shape[0]:
    print('\033[92m', 'mismatch!')
    print('\033[92m', x_train_dense.shape, x_train_sparse.shape)
    print('\033[0m')
  if x_test_dense.shape[0] != x_test_sparse.shape[0]:
    print('\033[92m', 'mismatch!')
    print('\033[92m', x_train_dense.shape, x_train_sparse.shape)
    print('\033[0m')

  print()
  print(type(x_train_dense), type(x_train_sparse))
  print(x_train_dense.shape, x_train_sparse.shape)
  x_train = np.concatenate((x_train_dense, x_train_sparse), axis=1)
  print('train shape:', x_train.shape)
  x_test = np.concatenate((x_test_dense, x_test_sparse), axis=1)
  print('test shape:', x_test.shape)
  print()

  if cfg.get('data', 'classif_param') == 'search':
    classifier = grid_search(x_train, y_train_sparse)
  else:
    classifier = LinearSVC(class_weight='balanced')
    classifier.fit(x_train, y_train_sparse)

  predictions = classifier.predict(x_test)
  p = precision_score(y_test_sparse, predictions, average='macro')
  r = recall_score(y_test_sparse, predictions, average='macro')
  f1 = f1_score(y_test_dense, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f\n" % (p, r, f1))

  return p, r, f1

def get_sparse_data(disease, judgement, use_svd=False):
  """Train on train set and evaluate on test set"""

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_data = os.path.join(base, cfg.get('data', 'train_data'))
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))
  test_data = os.path.join(base, cfg.get('data', 'test_data'))
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  # handle training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=True, # alphabet not really used
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))
  x_train, y_train = train_data_provider.load_raw()

  vectorizer = CountVectorizer(
    ngram_range=NGRAM_RANGE,
    stop_words='english',
    min_df=MIN_DF,
    vocabulary=None,
    binary=False)
  train_count_matrix = vectorizer.fit_transform(x_train)

  tf = TfidfTransformer()
  train_tfidf_matrix = tf.fit_transform(train_count_matrix)

  # now handle the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True, # alphabet not really used
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))
  x_test, y_test = test_data_provider.load_raw()

  test_count_matrix = vectorizer.transform(x_test)
  test_tfidf_matrix = tf.transform(test_count_matrix)

  if use_svd:
    # reduce sparse vector to 300 dimensions
    svd = TruncatedSVD(n_components=300)
    train_tfidf_matrix = svd.fit_transform(train_tfidf_matrix)
    test_tfidf_matrix = svd.transform(test_tfidf_matrix)

  return train_tfidf_matrix.toarray(), y_train, test_tfidf_matrix.toarray(), y_test

def run_evaluation_all_diseases():
  """Evaluate classifier performance for all 16 comorbidities"""

  exclude = set()

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  judgement = cfg.get('data', 'judgement')
  evaluation = cfg.get('data', 'evaluation')
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  ps = []; rs = []; f1s = []
  for disease in i2b2.get_disease_names(test_annot, exclude):
    p, r, f1 = run_evaluation(disease, judgement)
    ps.append(p)
    rs.append(r)
    f1s.append(f1)

  print('average p = %.3f' % np.mean(ps))
  print('average r = %.3f' % np.mean(rs))
  print('average f1 = %.3f' % np.mean(f1s))

if __name__ == "__main__":

  run_evaluation_all_diseases()
