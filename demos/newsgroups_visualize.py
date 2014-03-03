#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from pmtk3.data_tool import load_data


def main():
    data = load_data('20news_w100')
    X = data['documents']
    y = np.array(data['newsgroups'])[0]
    classlabels = [o[0] for o in data['groupnames'][0]]
    wordlist = np.array([o[0] for o in data['wordlist'][0]])

    # pick the first 1000 by word count
    nwords = np.array(X.sum(axis=0))[0]
    ndx = nwords.argsort()[::-1]
    ndx = ndx[:1000]
    XX = X[:,ndx]
    yy = y[ndx]

    # sort by label
    ndx = yy.argsort()
    yy = yy[ndx]
    XX = XX[:,ndx]

    # plot
    xx = np.transpose(XX.todense())
    plt.imshow(xx, aspect='auto', interpolation='none')

    for nrow in np.cumsum(np.bincount(yy)[1:]):
        plt.axhline(nrow, lw=3)

    plt.xlim(0, 99)
    plt.ylim(999, 0)
    plt.xlabel('words')
    plt.ylabel('documents')
    plt.show()

    # randomly sample bag of words for each class
    i = 0
    for idx, j in enumerate(np.cumsum(np.bincount(yy)[1:])):
        k = np.random.random_integers(i, j)
        m = np.array(XX[:,k].toarray().flatten(), dtype=bool)

        print classlabels[idx]
        print ' '.join(wordlist[m])
        print

        i = j


if __name__ == '__main__':
    main()
