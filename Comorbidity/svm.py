#!/usr/bin/env python

import numpy
numpy.random.seed(0)

import sys
sys.dont_write_bytecode = True
import ConfigParser, os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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
NUM_FOLDS = 5
NGRAM_RANGE = (1, 1) # use unigrams for cuis
MIN_DF = 0

def run_cross_validation(disease, judgement):
  """Run n-fold CV on training set"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_data = os.path.join(base, cfg.get('data', 'train_data'))
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))

  dataset = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement)
  x, y = dataset.load_raw()

  # fit builds alphabet; transform extracts counts
  vectorizer = CountVectorizer(
    ngram_range=NGRAM_RANGE,
    stop_words='english',
    min_df=MIN_DF,
    vocabulary=None,
    binary=False)
  count_matrix = vectorizer.fit_transform(x)

  # print features to file for debugging
  feature_file = open(FEATURE_LIST, 'w')
  for feature in vectorizer.get_feature_names():
    feature_file.write(feature + '\n')

  # transform raw counts to tf-idf
  tf = TfidfTransformer()
  tfidf_matrix = tf.fit_transform(count_matrix)

  classifier = LinearSVC(class_weight='balanced', C=10)
  cv_scores = cross_val_score(
    classifier,
    tfidf_matrix,
    y,
    scoring='f1_macro',
    cv=NUM_FOLDS)

  print 'average f1:', numpy.mean(cv_scores)
  print 'standard devitation:', numpy.std(cv_scores)

def run_evaluation(disease, judgement='intuitive'):
  """Train on train set and evaluate on test set"""

  cfg = ConfigParser.ConfigParser()
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
    use_pickled_alphabet=False,
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
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))
  x_test, y_test = test_data_provider.load_raw()

  test_count_matrix = vectorizer.transform(x_test)
  test_tfidf_matrix = tf.transform(test_count_matrix)

  classifier = LinearSVC(class_weight='balanced', C=1)
  classifier.fit(train_tfidf_matrix, y_train)
  predictions = classifier.predict(test_tfidf_matrix)

  f1 = f1_score(y_test, predictions, average='macro')
  print '%s: f1 = %.3f\n' % (disease, f1)

  return f1

def run_evaluation_learned_rep(disease, judgement='intuitive'):
  """Use pre-trained patient representations"""

  cfg = ConfigParser.ConfigParser()
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

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'))
  x_train, y_train = train_data_provider.load()

  classes = len(train_data_provider.label2int)
  maxlen = cfg.getint('data', 'maxlen')
  x_train = pad_sequences(x_train, maxlen=maxlen)

  # make training vectors for target task
  print 'original x_train shape:', x_train.shape
  x_train = interm_layer_model.predict(x_train)
  print 'new x_train shape:', x_train.shape

  # now load the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'))
  x_test, y_test = test_data_provider.load()
  x_test = pad_sequences(x_test, maxlen=maxlen)

  # make test vectors for target task
  print 'original x_test shape:', x_test.shape
  x_test = interm_layer_model.predict(x_test)
  print 'new x_test shape:', x_test.shape

  classifier = LinearSVC(class_weight='balanced', C=1)
  model = classifier.fit(x_train, y_train)
  predicted = classifier.predict(x_test)
  f1 = f1_score(y_test, predicted, average='macro')
  print '%s: f1 = %.3f\n' % (disease, f1)

  return f1

def run_evaluation_all_diseases(judgement):
  """Evaluate classifier performance for all 16 comorbidities"""

  exclude = set(['GERD', 'Venous Insufficiency', 'CHF'])

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))

  f1s = []
  for disease in i2b2.get_disease_names(train_annot, exclude):
    # f1 = run_evaluation(disease, judgement)
    f1 = run_evaluation_learned_rep(disease, judgement)
    f1s.append(f1)

  print 'average f1 =', numpy.mean(f1s)

if __name__ == "__main__":

  run_evaluation_all_diseases('intuitive')
