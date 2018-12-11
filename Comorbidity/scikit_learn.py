#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential

class FixedKerasClassifier(KerasClassifier):
  """My version that fixes an issue in predict method"""

  def fit(self, x, y, sample_weight=None, **kwargs):
    """Fixed version of fit method"""

    y = np.array(y)

    if len(y.shape) == 2 and y.shape[1] > 1:
      self.classes_ = np.arange(y.shape[1])
    elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:

      # line below is a problem when y doesn't contain all labels
      # for a particular split since the missing label can be predicted
      # in my case, when there are 3 outputs, sometimes the third label
      # is not in the training data since it's rare in i2b2 data

      # self.classes_ = np.unique(y)

      # ugly fix for now
      self.classes_ = np.array([0, 1, 2])

      y = np.searchsorted(self.classes_, y)
    else:
      raise ValueError('Invalid shape for y: ' + str(y.shape))
    self.n_classes_ = len(self.classes_)
    if sample_weight is not None:
      kwargs['sample_weight'] = sample_weight

    return super(KerasClassifier, self).fit(x, y, **kwargs)

if __name__ == "__main__":

  pass
