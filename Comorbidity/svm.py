#!/usr/bin/env python
import numpy, sys, ConfigParser, os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from dataset import DatasetProvider

DISEASE = 'Asthma'
JUDGEMENT = 'textual'
FEATURE_LIST = './features.txt'
NUM_FOLDS = 5
NGRAM_RANGE = (1, 1) # use unigrams for cuis
MIN_DF = 0

def run_cross_validation():
  """Run n-fold CV and return average accuracy"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'path'))

  dataset = DatasetProvider(data_dir, DISEASE, JUDGEMENT)
  x, y = dataset.load_raw()
  print x

  # raw occurences
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

  # tf-idf
  tf = TfidfTransformer()
  tfidf_matrix = tf.fit_transform(count_matrix)

  classifier = LinearSVC(class_weight='balanced', C=1)
  cv_scores = cross_val_score(
    classifier,
    tfidf_matrix,
    y,
    scoring='f1_macro',
    cv=NUM_FOLDS)

  print 'fold f1s:', cv_scores
  print 'average f1:', numpy.mean(cv_scores)
  print 'standard devitation:', numpy.std(cv_scores)

if __name__ == "__main__":

  run_cross_validation()
