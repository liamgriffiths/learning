#!/usr/bin/python

import numpy as np
from knn import KNN
from ann import ANN
from sklearn import datasets, cross_validation, metrics

class Evaluator:
    """ object to keep track of results and print out metrics """
    def __init__(self):
        self.Ytrue = []
        self.Ypredict = []

    def append_results(self, Ytrue, Ypredict):
        self.Ytrue.append(Ytrue)
        self.Ypredict.append(Ypredict)

    def avg_f1_score(self):
        scores = [metrics.f1_score(self.Ytrue[i],self.Ypredict[i]) for i in range(len(self.Ytrue))]
        return np.mean(scores)

    def print_reports(self):
        for i in range(len(self.Ytrue)):
            print metrics.classification_report(Ytrue,Ypredict)


def crossvalidate(function, X, Y, k=10):
    """ cross validate by splitting data into k pieces """
    m = len(Y)
    kfold = cross_validation.KFold(m, k=k) 
    evaluator = Evaluator()
    for train_index, test_index in kfold:
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        Ypredict = function(Xtrain,Ytrain,Xtest,Ytest)
        evaluator.append_results(Ytest,Ypredict)

    return evaluator
    
def compare_functions():
    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target
    classes = list(set(Y))

    # compare the KNN and ANN using

    def knncv(Xtrain, Ytrain, Xtest, Ytest):
        knn = KNN(Xtrain,Ytrain)
        
        m = len(Ytest)
        Ypredict = np.zeros(m)
        for i in xrange(m):
            x,y = Xtest[i],Ytest[i]
            results = knn.predict(x,k=4,classes=classes)
            prediction = results.argmax()
            Ypredict[i] = prediction
        return Ypredict

    evaluator = crossvalidate(knncv,X,Y,k=3)
    print "KNN avg f1 scores", evaluator.avg_f1_score()

    def anncv(Xtrain, Ytrain, Xtest, Ytest):
        inputsize = Xtrain.shape[1]
        hiddensize = 12
        outputsize = len(classes)
        ann = ANN(inputsize,hiddensize,outputsize)

        mtrain = len(Ytrain)
        # for this example, im only training with the first 12 examples
        for n in xrange(400):
            for i in xrange(mtrain):
                x,y = Xtrain[i],Ytrain[i]
                target = np.zeros(len(classes))
                target[classes.index(y)] = 1
                ann.train(x,target,alpha=0.1,momentum=0.2)
        
        mtest = len(Ytest)
        Ypredict = np.zeros(mtest)
        for i in xrange(mtest):
            x,y = Xtest[i],Ytest[i]
            results = ann.predict(x)
            prediction = results.argmax()
            Ypredict[i] = prediction
        return Ypredict

    evaluator = crossvalidate(anncv,X,Y,k=3)
    print "ANN avg f1 scores", evaluator.avg_f1_score()

if __name__ == '__main__':
    compare_functions()
    

