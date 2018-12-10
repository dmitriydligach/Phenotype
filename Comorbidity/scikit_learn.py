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
    print('fit / y.shape', y.shape)

    if len(y.shape) == 2 and y.shape[1] > 1:
      self.classes_ = np.arange(y.shape[1])
      print("fit / inside if self.classes_:", self.classes_)
    elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
      self.classes_ = np.unique(y)
      print("fit / inside ELIF self.classes_:", self.classes_)
      y = np.searchsorted(self.classes_, y)
    else:
      raise ValueError('Invalid shape for y: ' + str(y.shape))
    self.n_classes_ = len(self.classes_)
    if sample_weight is not None:
      kwargs['sample_weight'] = sample_weight

    return super(KerasClassifier, self).fit(x, y, **kwargs)

  def predict(self, x, **kwargs):
    """Fix version of predict method"""

    kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)

    proba = self.model.predict(x, **kwargs)
    print("proba shape:", proba.shape)
    # print("self classes_:", self.classes_)

    if proba.shape[-1] > 1:
        classes = proba.argmax(axis=-1)
        print('predict / inside if classes:', classes[:10])
    else:
        classes = (proba > 0.5).astype('int32')
        print('predict / inside else classes:', classes[:10])

    return self.classes_[classes]

if __name__ == "__main__":

  pass
