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
from sklearn.feature_extraction.text import TfidfVectorizer
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

def run_evaluation_dense(cfg, disease, judgement):
  """Use pre-trained patient representations"""

  print('disease:', disease)
  x_train, y_train, x_test, y_test = data_dense(cfg, disease, judgement)

  if cfg.get('data', 'classif_param') == 'search':
    classifier = grid_search(x_train, y_train)
  else:
    classifier = LinearSVC(class_weight='balanced')
    classifier.fit(x_train, y_train)

  predictions = classifier.predict(x_test)
  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f\n" % (p, r, f1))

  return p, r, f1

def run_evaluation_sparse(cfg, disease, judgement, use_svd=False):
  """Train on train set and evaluate on test set"""

  print('disease:', disease)
  x_train, y_train, x_test, y_test = data_sparse(cfg, disease, judgement)

  if use_svd:
    # reduce sparse vector to 300 dimensions
    svd = TruncatedSVD(n_components=300)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)

  if cfg.get('data', 'classif_param') == 'search':
    classifier = grid_search(x_train, y_train)
  else:
    classifier = LinearSVC(class_weight='balanced')
    classifier.fit(x_train, y_train)

  predictions = classifier.predict(x_test)

  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print('unique labels in train:', len(set(y_train)))
  print('p = %.3f' % p)
  print('r = %.3f' % r)
  print('f1 = %.3f\n' % f1)

  return p, r, f1

def run_evaluation_hybrid(cfg, disease, judgement):
  """Concatenated dense and sparse vectors and eval"""

  print('disease:', disease)
  x_train_dense, y_train, x_test_dense, y_test = data_dense(cfg, disease, judgement)
  x_train_sparse, y_train, x_test_sparse, y_test = data_sparse(cfg, disease, judgement)

  if x_train_dense.shape[0] != x_train_sparse.shape[0]:
    print('mismatch in train!')
  if x_test_dense.shape[0] != x_test_sparse.shape[0]:
    print('mismatch in test!')

  x_train = np.concatenate((x_train_dense, x_train_sparse), axis=1)
  print('train shape:', x_train.shape)
  x_test = np.concatenate((x_test_dense, x_test_sparse), axis=1)
  print('test shape:', x_test.shape)

  if cfg.get('data', 'classif_param') == 'search':
    classifier = grid_search(x_train, y_train)
  else:
    classifier = LinearSVC(class_weight='balanced')
    classifier.fit(x_train, y_train)

  predictions = classifier.predict(x_test)
  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f\n" % (p, r, f1))

  return p, r, f1

def data_dense(cfg, disease, judgement):
  """Data to feed into code prediction model"""

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

  # determine whether to treat input tokens as a sequence or set
  if cfg.get('data', 'model_type') == 'dan':
    use_cuis = True
    tokens_as_set = True
  else:
    use_cuis = False
    tokens_as_set = False

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'),
    use_cuis=use_cuis)
  x_train, y_train = train_data_provider.load(tokens_as_set=tokens_as_set)

  classes = len(set(y_train))
  print('unique labels in train:', classes)
  x_train = pad_sequences(x_train, maxlen=maxlen)

  # make training vectors for target task
  print('original x_train shape:', x_train.shape)
  x_train = interm_layer_model.predict(x_train)
  print('new x_train shape:', x_train.shape)

  # now load the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'),
    use_cuis=use_cuis)
  x_test, y_test = test_data_provider.load(tokens_as_set=tokens_as_set)
  x_test = pad_sequences(x_test, maxlen=maxlen)

  # make test vectors for target task
  print('original x_test shape:', x_test.shape)
  x_test = interm_layer_model.predict(x_test)
  print('new x_test shape:', x_test.shape)

  return x_train, y_train, x_test, y_test

def data_sparse(cfg, disease, judgement, use_svd=False):
  """Bag-of-cuis data for sparse evaluation"""

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
    judgement)
  x_train, y_train = train_data_provider.load_raw()
  print('train examples:', len(x_train))

  vectorizer = TfidfVectorizer(
    ngram_range=NGRAM_RANGE,
    stop_words='english',
    min_df=MIN_DF,
    vocabulary=None,
    binary=False)
  x_train = vectorizer.fit_transform(x_train)

  # now handle the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement)
  x_test, y_test = test_data_provider.load_raw()
  print('test examples:', len(x_test))
  x_test = vectorizer.transform(x_test)

  return x_train.toarray(), y_train, x_test.toarray(), y_test

def run_evaluation_svd(disease, judgement):
  """Train on train set and evaluate on test set"""

  print('disease:', disease)
  print('judgement:', judgement)

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
    judgement)
  x_train, y_train = train_data_provider.load_raw()
  print('train examples:', len(x_train))

  # load tfidf vectorizer model and transform xs into it
  vectorizer = pickle.load(open('../Svd/Model/tfidf.p', 'rb'))
  train_tfidf_matrix = vectorizer.transform(x_train)

  # now handle the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement)
  x_test, y_test = test_data_provider.load_raw()
  print('test examples:', len(x_test))
  test_tfidf_matrix = vectorizer.transform(x_test)

  # load svd model and map train/test to low dimensions
  print('input shape:', train_tfidf_matrix.shape)
  svd = pickle.load(open('../Svd/Model/svd.p', 'rb'))
  train_tfidf_matrix = svd.transform(train_tfidf_matrix)
  test_tfidf_matrix = svd.transform(test_tfidf_matrix)
  print('output shape:', train_tfidf_matrix.shape)

  classifier = LinearSVC(class_weight='balanced')
  classifier.fit(train_tfidf_matrix, y_train)
  predictions = classifier.predict(test_tfidf_matrix)

  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print('unique labels in train:', len(set(y_train)))
  print('p = %.3f' % p)
  print('r = %.3f' % r)
  print('f1 = %.3f\n' % f1)

  print('%.3f & %.3f & %.3f\n' % (p, r, f1))

  return p, r, f1

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
    if evaluation == 'sparse':
      # use bag-of-word vectors
      p, r, f1 = run_evaluation_sparse(cfg, disease, judgement)
    elif evaluation == 'svd':
      # use low dimensional vectors obtained via svd
      p, r, f1 = run_evaluation_svd(cfg, disease, judgement)
    elif evaluation == 'dense':
      # use learned patient vectors
      p, r, f1 = run_evaluation_dense(cfg, disease, judgement)
    elif evaluation == 'hybrid':
      # use combined dense and sparse vectors
      p, r, f1 = run_evaluation_hybrid(cfg, disease, judgement)
    ps.append(p)
    rs.append(r)
    f1s.append(f1)

  print('average p = %.3f' % np.mean(ps))
  print('average r = %.3f' % np.mean(rs))
  print('average f1 = %.3f' % np.mean(f1s))

if __name__ == "__main__":

  run_evaluation_all_diseases()
