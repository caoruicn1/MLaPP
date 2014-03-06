#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from pmtk3.data_tool import load_data
from pmtk3.toolbox.supervised_model import naive_bayes
from pmtk3.stats import mutual_info


def main():
    data = load_data('XwindowsDocData')
    Xtrain = data['xtrain']
    ytrain = np.ravel(data['ytrain'])
    vocab = [o[0] for o in data['vocab']]

    # train a model
    model = naive_bayes.fit(Xtrain, ytrain)
    for idx, theta in enumerate(model.theta):
        plt.figure(idx)
        plt.plot(theta)
        plt.title('p(x_j=1|y={})'.format(idx+1))

    #plt.show()

    # top n words
    n = 5
    for idx, theta in enumerate(model.theta):
        ndx = theta.argsort()[::-1]
        sorted_prob = theta[ndx]
        print 'top {} words for class {}'.format(n, idx+1)
        for w in range(0, n):
            print '{:d} {:6.4f} {}'.format(w+1, sorted_prob[w], vocab[ndx[w]][0])
        print

    # compute mutual information between words and class labels
    mi = mutual_info.class_features_binary(Xtrain, ytrain)
    ndx = mi.argsort(axis=0)[::-1]
    sorted_mi = mi[ndx]
    print 'top {} words sorted by mutual information'.format(n)
    for w in range(0, n):
        print '{} {:6.4f} {}'.format(w + 1, sorted_mi[w], vocab[ndx[w]][0])
    print 


if __name__ == '__main__':
    main()
