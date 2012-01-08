#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets

class ANN:
    def __init__(self, inputsize, hiddensize, outputsize):
        """ setup the network """
        # add an additional node to the input and hidden layers as bias nodes
        inputsize += 1
        hiddensize += 1

        # thetas hold the weights for connections between nodes 
        # theta1 matrix holds weights for connections between input and hidden nodes
        # theta2 matrix for hidden to output nodes
        # initially set the weights randomly
        np.random.seed(123)
        self.theta1 = np.random.normal(0,0.5,[inputsize,hiddensize])
        self.theta2 = np.random.normal(0,0.5,[hiddensize,outputsize])

        # change matricies to hold onto the last change in weight for the thetas
        # this is used for incorporating momentum into the weight updates in backprop
        self.theta1change = np.zeros([inputsize,hiddensize])
        self.theta2change = np.zeros([hiddensize,outputsize])

    def feedforward(self, x):
        """ push feature vector x through the network, return each layers output """
        # set inputlayer output to x plus 1.0 bias node
        inputlayer = np.append(1.0,np.tanh(x))

        # calculate hidden layer vector output, set bias node to 1.0
        z2 = np.dot(inputlayer,self.theta1)
        hiddenlayer = np.tanh(z2)
        hiddenlayer[0] = 1.0

        # calculate output layer vector
        z3 = np.dot(hiddenlayer,self.theta2)
        outputlayer = np.tanh(z3)

        return inputlayer,hiddenlayer,outputlayer

    def backpropagation(self, targets, inputlayer, hiddenlayer, outputlayer, alpha, momentum):
        """ utilize backprop to update theta1 and theta2 weights """
        # alpha is the learning rate, or how much to update theta per training
        # momentum is the what we add to the theta change to prevent getting stuck in a local minimum

        # dtanh returns the derivative of the tanh function
        dtanh = lambda y: 1.0 - y ** 2

        # calculate the errors between the expected result and the result of the output and hidden layers
        # delta matricies determine how much and in what direction to "correct" weights 
        outputerrors = targets - outputlayer
        outputdeltas = dtanh(outputlayer) * outputerrors

        hiddenerrors = np.dot(outputdeltas,self.theta2.T)
        hiddendeltas = dtanh(hiddenlayer) * hiddenerrors

        # for each theta:
        # use the deltas to calculate the change gradient
        # update the weights for thetas to correct the errors
        change = np.array(np.matrix(hiddenlayer).T * np.matrix(outputdeltas))
        self.theta2 = self.theta2 + (alpha * change) + (momentum * self.theta2change)
        self.theta2change = change

        change = np.array(np.matrix(inputlayer).T * np.matrix(hiddendeltas))
        self.theta1 = self.theta1 + (alpha * change) + (momentum * self.theta1change)
        self.theta1change = change

    def predict(self, x):
        """ given feature vector x return the learned outputlayer """
        inputlayer,hiddenlayer,outputlayer = self.feedforward(x)
        return outputlayer
        
    def train(self, x, y, alpha=0.5, momentum=0.3):
        """ train a single example (x) with expected output (y) """
        # the learning rate (alpha) and the momentum values may need to be adjusted so they are not too high/low

        # first get the outputs of all the layers from pushing x through the network
        inputlayer,hiddenlayer,outputlayer = self.feedforward(x)
        # then use backprop to adjust the weights so the network's output is closer to y
        self.backpropagation(y,inputlayer,hiddenlayer,outputlayer,alpha,momentum)


def digits():
    """ teach the neural network what digits look like """
    # use the digits dataset from the sklearn library
    # the images are 8x8 bitmaps of handwritten digits {0,9}
    # when 'unrolled' each image becomes a 1x64 matrix
    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target

    classes = list(set(Y))

    # use each pixel as an input node
    # using 12 hidden nodes/neurons
    # output layer contains 10 nodes, one for each digit
    inputsize = X.shape[1]
    hiddensize = 12
    outputsize = len(classes)
    ann = ANN(inputsize,hiddensize,outputsize)

    # for this example, im only training with the first 12 examples
    for n in xrange(400):
        for i in xrange(12):
            x,y = X[i],Y[i]
            target = np.zeros(len(classes))
            target[classes.index(y)] = 1
            ann.train(x,target,alpha=0.1,momentum=0.2)

    # see how well the trained examples were learned
    for i in xrange(12):
        x,y = X[i],Y[i]
        results = ann.predict(x)
        prediction = results.argmax()
        pl.subplot(3,4,i+1)
        color = pl.cm.gray if prediction == y else pl.cm.Reds_r
        pl.imshow(digits.images[i], cmap=color)
        pl.title('Predicted=%i vs. Actual=%i' % (prediction,y))

    pl.show()

if __name__ == '__main__':
    digits()
