#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets, svm

def digits():
    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target

    svm_classifier = svm.SVC()
    svm_classifier.fit(X, Y)

    for i in xrange(12):
        x,y = X[i],Y[i]
        prediction = svm_classifier.predict(x)
        pl.subplot(3,4,i+1)
        color = pl.cm.gray if prediction == y else pl.cm.Reds_r
        pl.imshow(digits.images[i], cmap=color)
        pl.title('Predicted=%i vs. Actual=%i' % (prediction,y))

    pl.show()

if __name__ == '__main__':
    digits()
