#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential

class FixedKerasClassifier(KerasClassifier):
  """My version that fixes an issue in predict method"""

  def predict(self, x, **kwargs):
    """An updated version"""

    kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)

    proba = self.model.predict(x, **kwargs)
    print("proba shape:", proba.shape)
    print("self classes_:", self.classes_)

    if proba.shape[-1] > 1:
        classes = proba.argmax(axis=-1)
        print('inside if classes:', classes[:10])
    else:
        classes = (proba > 0.5).astype('int32')
        print('inside else classes:', classes[:10])

    return self.classes_[classes]

if __name__ == "__main__":

  pass
