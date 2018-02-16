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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TrainSVD:
  """Use SVD to train a dimensionality reduction model"""

  def __init__(self,
               corpus_path,
               max_tokens_in_file):
    """Load documents as strings"""

    self.samples = []

    for f in os.listdir(corpus_path):
      file_path = os.path.join(corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)
      if len(file_feat_list) < max_tokens_in_file:
        self.samples.append(' '.join(file_feat_list))

  def train(self):
    """Train SVD"""

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(self.samples)
    pickle.dump(vectorizer, open('Model/tfidf.p', 'wb'))

    svd = TruncatedSVD(n_components=300)
    svd.fit(tfidf_matrix)
    pickle.dump(svd, open('Model/svd.p', 'wb'))

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  path = 'MimicIII/Patients/Cuis/'
  data_dir = os.path.join(base, path)

  dataset = TrainSVD(data_dir, 10000)
  dataset.train()
