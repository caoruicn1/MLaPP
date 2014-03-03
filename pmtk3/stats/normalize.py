#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-


def normalize(X, axis=None):
    """Normalize an array (i.e., sum to one).

    If axis is not given, normalize the whole array to one.  If axis
    is given, normalize each column (0)/row (1) sums to one.
    """
    if axis is None:
        z = X.sum()
        normed = X / z
    else:
        # only considering 2d for now...
        z = X.sum(axis=axis)
        if axis == 0:
            normed = X / z
        elif axis == 1:
            normed = X / z[:, np.newaxis]
        else:
            raise RuntimeError('only 2d array allowed')

    return normed, z
