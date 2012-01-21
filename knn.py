#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets

class KNN:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, x, k=5, classes=None):
        """ make prediction based on k neighbors """
        # calculate the euclidean distance between features (x) and each example in the training set (X)
        similarities = (np.linalg.norm(difference) for difference in (x - self.X))
        # sort list by similarity and and index them so we can reference the rows output
        indexed = [[index,similarity] for index,similarity in enumerate(similarities)]
        indexed.sort(key=lambda x: x[1])
        
        if classes:
            # probability for each class using the k nearest neighbors
            results = np.zeros(len(classes))
            for i in range(k):
                index = classes.index(self.Y[indexed[i][0]])
                results[index] += 1.0
            return results / k

        else:
            # mean of all the k nearest neighbors
            results = 0.0
            for i in range(k):
                results += self.Y[indexed[i][0]]
            return results / k


def digits():
    """ use k nearest neighbors to predict what a digit looks like """
    # use the digits dataset from the sklearn library
    # the images are 8x8 bitmaps of handwritten digits {0,9}
    # when 'unrolled' each image becomes a 1x64 matrix
    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target

    classes = list(set(Y))

    # use the first 200 examples as our training set
    Xtrain = X[0:200]
    Ytrain = Y[0:200]
    knn = KNN(Xtrain,Ytrain)

    
    # see how well we can predict the 12 unseen digits
    for n in xrange(12):
        i = 300 + n
        x,y = X[i],Y[i]
        results = knn.predict(x,k=5,classes=classes)
        prediction = results.argmax()
        pl.subplot(3,4,n+1)
        color = pl.cm.gray if prediction == y else pl.cm.Reds_r
        pl.imshow(digits.images[i], cmap=color)
        pl.title('Predicted=%i vs. Actual=%i' % (prediction,y))

    pl.show()

if __name__ == '__main__':
    digits()
