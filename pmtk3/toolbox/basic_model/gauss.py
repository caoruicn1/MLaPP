#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import numpy as np


def gauss_sample(mu, Sigma):
    A = np.linalg.cholesky(Sigma)
    Z = np.random.normal(size=mu.shape)
    S = mu + A.T * Z
    return S.H
