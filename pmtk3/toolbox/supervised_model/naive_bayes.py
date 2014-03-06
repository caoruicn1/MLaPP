#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import numpy as np
from pmtk3.stats.normalize import normalize


def fit(Xtrain, ytrain, pseudo_count=1):
    """Fit a naive Bayes classifer using MAP/ML estimation.

    This implemtation assumes binary features (i.e., bernoulli
    distribution).

    Parameters
    ----------
    Xtrain : array_like
        2d array (i, j) of 0 or 1, for bit j in case i
    ytrain : array_like
        1d array of class in {1, ..., C}
    pseudo_count : int, optional
        Strength of symmetric beta prior for the features, for
        computing posterior mean.  Use 0 for MLE.  Defaults to 1.

    Returns
    -------
    A Model object with the following attributes:

    model.theta : array_like
        2d probability array (c, j) of bit j turning on in class c.
    model.class_prior : array_like
        p(y = c)
    """
    C = len(np.bincount(ytrain)[1:])
    Ntrain, D = Xtrain.shape

    theta = np.zeros((C, D))
    Nclass = np.zeros(C)
    
    for c in range(1, C+1):
        ndx = ytrain == c
        Xtr = Xtrain[ndx,:]
        Non = (Xtr > 0).sum(axis=0)
        Noff = (Xtr == 0).sum(axis=0)
        theta[c-1,:] = (Non + pseudo_count) / (Non + Noff + 2.*pseudo_count)
        Nclass[c-1] = np.sum(ndx)

    class Model(object):
        pass

    model = Model()
    model.class_prior = normalize(Nclass)
    model.theta = theta

    return model


def predict(model, Xtest, nargout=1):
    # only implementing "vectorized" case
    eps = np.spacing(1)
    Ntest = Xtest.shape[0]
    theta = model.theta
    C = theta.shape[0]
    log_prior = np.log(model.class_prior + eps)
    log_theta = np.log(theta + eps)
    log_theta_not = np.log(1. - theta + eps)
    Xtest = Xtest.toarray().astype(int)
    XtestNot = np.logical_not(Xtest).astype(int)

    log_posterior = np.zeros((Ntest, C))
    for c in range(0, C):
        L1 = log_theta[c,:] * Xtest
        L0 = log_theta_not[c,:] * XtestNot
        log_posterior[:,c] = (L1 + L0).sum(axis=1) + log_prior[c]

    yhat = log_posterior.argmax(axis=1)

    if nargout == 1:
        return yhat
    if nargout == 2:
        raise NotImplemented
        #py = np.exp(normalize_logspace(log_posterior))
        return yhat, py
