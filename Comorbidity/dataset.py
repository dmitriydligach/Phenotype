#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import utils, i2b2
import numpy, pickle
import ConfigParser, os, nltk, pandas
import glob, string, collections, operator

ALPHABET_FILE = 'alphabet.txt'

class DatasetProvider:
  """Comorboditiy data loader"""

  def __init__(self,
               corpus_path,
               annot_xml,
               disease,
               judgement,
               min_token_freq=0):
    """Index words by frequency in a file"""

    self.corpus_path = corpus_path
    self.annot_xml = annot_xml
    self.min_token_freq = min_token_freq
    self.disease = disease
    self.judgement = judgement

    self.label2int = {'N':0, 'Y':1, 'Q':2, 'U':3}
    self.token2int = {}
    self.make_token_alphabet()

  def make_token_alphabet(self):
    """Map tokens (CUIs) to integers"""

    # count tokens in the entire corpus
    token_counts = collections.Counter()

    for f in os.listdir(self.corpus_path):
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)
      token_counts.update(file_feat_list)

    # now make alphabet (high freq tokens first)
    index = 1
    self.token2int['oov_word'] = 0
    outfile = open(ALPHABET_FILE, 'w')
    for token, count in token_counts.most_common():
      if count > self.min_token_freq:
        outfile.write('%s|%s\n' % (token, count))
        self.token2int[token] = index
        index = index + 1

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices for keras"""

    labels = []   # int labels
    examples = [] # int sequence represents each example

    # document id -> label mapping
    doc2label = i2b2.parse_standoff(
      self.annot_xml,
      self.disease,
      self.judgement)

    # load examples and labels
    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)

      example = []
      # TODO: use unique tokens or not?
      for token in set(file_feat_list):
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]

      # no labels for some documents for some reason
      if doc_id in doc2label:
        string_label = doc2label[doc_id]
        int_label = self.label2int[string_label]
        labels.append(int_label)
        examples.append(example)
      else:
        print 'missing label for doc:', doc_id

    return examples, labels

  def load_raw(self):
    """Load for sklearn training"""

    labels = []   # string labels
    examples = [] # examples as strings

    # document id -> label mapping
    doc2label = i2b2.parse_standoff(
      self.annot_xml,
      self.disease,
      self.judgement)

    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)

      # no labels for some documents for some reason
      if doc_id in doc2label:
        string_label = doc2label[doc_id]
        labels.append(string_label)
        examples.append(' '.join(file_feat_list))
      else:
        print 'missing label for doc:', doc_id

    return examples, labels

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'path'))
  annot_xml = os.path.join(base, cfg.get('data', 'annot'))

  dataset = DatasetProvider(data_dir, annot_xml, 'Asthma', 'textual')
  x, y = dataset.load_raw()
  print x
  print y
