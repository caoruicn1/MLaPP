#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import numpy as np


def zero_one_loss_fn(yhat, ytest):
    L = (yhat[:] != ytest)
    return L
