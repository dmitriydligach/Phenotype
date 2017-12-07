#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import utils
import numpy, pickle
import ConfigParser, os, nltk, pandas
import glob, string, collections, operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

class TrainSVD:
  """Mimic data loader"""

  def __init__(self,
               corpus_path,
               max_tokens_in_file):
    """Load documents as strings"""

    self.samples = []
    self.corpus_path = corpus_path
    self.max_tokens_in_file = max_tokens_in_file

  def load(self):
    """Load documents as strings"""

    for f in os.listdir(self.corpus_path):
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)
      if len(file_feat_list) < self.max_tokens_in_file:
        self.samples.append(' '.join(file_feat_list))

  def reduce(self):
    """Turn into a lower dimensional matrix"""

    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=0)
    train_count_matrix = vectorizer.fit_transform(self.samples)
    print 'input dimensions:', train_count_matrix.shape
    pickle.dump(vectorizer, open('cv.p', 'wb'))

    tf = TfidfTransformer()
    train_tfidf_matrix = tf.fit_transform(train_count_matrix)
    pickle.dump(tf, open('tf.p', 'wb'))

    svd = TruncatedSVD(n_components=1000)
    svd.fit(train_tfidf_matrix)
    pickle.dump(svd, open('svd.p', 'wb'))

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  path = 'MimicIII/Patients/Cuis/'
  data_dir = os.path.join(base, path)

  dataset = TrainSVD(data_dir, 10000)
  dataset.load()
  dataset.reduce()
