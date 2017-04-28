#!/usr/bin/env python
# encoding: utf-8

import sys
from sklearn import svm, grid_search

def accuracy(Y, label):
    correct = 0
    for y, l in zip(Y, label):
        if y == l:
            correct+=1
    return correct/float(len(Y))


if __name__=='__main__':
    domain = sys.argv[1]

