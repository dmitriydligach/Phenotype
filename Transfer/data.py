#!/usr/bin/env python3

import configparser, sys, os, shutil
sys.dont_write_bytecode = True
sys.path.append('../Codes')
from dataset import DatasetProvider

MODEL_DIR = 'Model/'
ALPHABET_FILE = 'Model/alphabet.txt'
ALPHABET_PICKLE = 'Model/alphabet.p'
CODE_FREQ_FILE = 'Model/codes.txt'
DIAG_ICD9_FILE = 'DIAGNOSES_ICD.csv'

class TransferDataset(DatasetProvider):
  """Make data for transfer learning"""

  def __init__(self,
               corpus_path,
               code_dir,
               target_code_path,
               min_token_freq,
               max_tokens_in_file,
               min_examples_per_code,
               collapse_codes,
               use_cuis=True):
    """Constructor. Allows to specify ICD codes."""

    self.corpus_path = corpus_path
    self.code_dir = code_dir
    self.target_code_path = target_code_path
    self.min_token_freq = min_token_freq
    self.max_tokens_in_file = max_tokens_in_file
    self.min_examples_per_code = min_examples_per_code
    self.code_characters = 3 if collapse_codes else None
    self.use_cuis = use_cuis

    self.token2int = {}  # words indexed by frequency
    self.subj2codes = {} # subj_id to set of icd9 codes

    # remove old model directory and make a fresh one
    if os.path.isdir(MODEL_DIR):
      print('removing old model directory...')
      shutil.rmtree(MODEL_DIR)
    print('making alphabet and saving it in file...')
    os.mkdir(MODEL_DIR)
    self.make_and_write_token_alphabet()

    # 3051 is actually tobacco use disorder
    # so don't need to truncate at 3 chars?
    print('mapping codes...')
    code_file = os.path.join(self.code_dir, DIAG_ICD9_FILE)
    self.index_codes(
      code_file,
      'HADM_ID',
      'ICD9_CODE',
      'diag',
      self.code_characters)

  def load(self,
           maxlen=float('inf'),
           tokens_as_set=True):
    """Make x and y"""

    labels = [] # does this example have one of predefined codes?
    examples = [] # int sequence represents each example

    # read target codes
    target_code_categories = set([])
    for line in open(self.target_code_path):
      short_code = 'diag_%s' % line.strip()[0:self.code_characters]
      target_code_categories.add(short_code)

    for file in os.listdir(self.corpus_path):
      file_ngram_list = None
      if self.use_cuis == True:
        file_ngram_list = self.read_cuis(file)
      else:
        file_ngram_list = self.read_tokens(file)
      if file_ngram_list == None:
        continue # file too long

      # determine the label for this subj_id
      subj_id = file.split('.')[0]
      if subj_id not in self.subj2codes:
        continue # subject was present once with no code
      if len(self.subj2codes[subj_id]) == 0:
        continue # shouldn't happen

      icd9_categories = set(self.subj2codes[subj_id])
      overlap = target_code_categories.intersection(icd9_categories)
      if len(overlap) > 0:
        labels.append(1) # this subj has a target code
      else:
        labels.append(0) # no target code for this subj

      # represent this example as a list of ints
      example = []

      if tokens_as_set:
        file_ngram_list = set(file_ngram_list)

      for token in file_ngram_list:
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]

      examples.append(example)

    return examples, labels

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  code_file = os.path.join(base, cfg.get('data', 'codes'))
  targ_file = os.path.join(base, cfg.get('data', 'targets'))

  dataset = TransferDataset(
    train_dir,
    code_file,
    targ_file,
    cfg.getint('args', 'min_token_freq'),
    cfg.getint('args', 'max_tokens_in_file'),
    cfg.getint('args', 'min_examples_per_code'),
    cfg.getboolean('args', 'collapse_codes'))
  print('chars:', dataset.code_characters)    
  x, y = dataset.load()

  print("positive:", sum(y))
  print("negative:", len(y) - sum(y))
