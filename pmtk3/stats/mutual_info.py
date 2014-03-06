#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import numpy as np
from pmtk3.toolbox.supervised_model import naive_bayes


def class_features_binary(X, y):
    model = naive_bayes.fit(X, y)
    py = model.class_prior
    pxy = model.theta
    C, D = pxy.shape

    # px(j) = p(x=j) = sum_c p(xj=1|y=c) p(y=c)
    px = (py[:, np.newaxis] * pxy).sum(axis=0)

    mi = np.zeros(D)
    for c in range(0, C):
        mi += py[c] * (pxy[c,:] * np.log2(pxy[c,:] / px)
                       + (1. - pxy[c,:]) * np.log2((1. - pxy[c,:]) / (1. - px)))

    return mi
