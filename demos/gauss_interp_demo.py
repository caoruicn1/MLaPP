#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""Interpolate some data using a joint Gaussian.

Based on p140 of "Introduction to Bayesian scientific computation"
by Calvetti and Somersalo."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags
from pmtk3.data_tool import load_data
from pmtk3.toolbox.decision_theory import zero_one_loss_fn
from pmtk3.stats import mutual_info
from pmtk3.toolbox.basic_model import gauss


def main():
    # set seed

    D = 150
    Nobs = 10
    ss = np.linspace(0, 1, D)
    perm = np.random.permutation(np.arange(1, D-1))
    # make sure end points are from observation
    obsNdx = [0]
    obsNdx = np.append(obsNdx, sorted(perm[:Nobs-2]))
    obsNdx = np.append(obsNdx, [D-1])
    hidNdx = sorted(perm[Nobs-2:])
    xobs = np.random.normal(size=Nobs)

    # make a (D-2) * D tridiagonal matrix
    ones = np.ones(D)
    L = 0.5 * spdiags([ones*-1, ones*2, ones*-1], [0, 1, 2], D-2, D)

    # prior precision lambda -- it only affects the variance, not the
    # mean, so we pick a value that results in a pretty plot.
    for idx, (lamb, name) in enumerate([(30., '30'), (5., '0p1')]):
        LL = L.todense() * lamb

        L1 = LL[:, hidNdx]
        L2 = LL[:, obsNdx]
        lam11 = L1.conj().T * L1
        lam12 = L1.conj().T * L2

        post_dist = {}
        post_dist['Sigma'] = np.linalg.inv(lam11)
        post_dist['mu'] = (-(np.linalg.inv(lam11) * lam12) * xobs.reshape((Nobs, 1))).A

        # plot
        ## plt.figure(idx)
        ## plt.plot(ss[hidNdx], post_dist['mu'], 'bo', lw=2)
        ## plt.plot(ss[obsNdx], xobs, 'ro', ms=12)
        ## plt.title('{}={}'.format('lambda', name))

        # marginal posterior probabilty mass standard dev as gray band
        xbar = np.zeros(D)
        xbar[hidNdx] = post_dist['mu']
        xbar[obsNdx] = xobs

        sigma = np.zeros(D)
        sigma[hidNdx] = np.sqrt(np.diagonal(post_dist['Sigma']))
        sigma[obsNdx] = 0

        mu = xbar
        s2 = sigma**2
        f = np.append(mu + 2*np.sqrt(s2), (mu - 2*np.sqrt(s2))[::-1])
        sss = np.append(ss, ss[::-1])

        plt.figure(idx+10)
        plt.fill(sss, f, 'gray')
        plt.plot(ss[obsNdx], xobs, 'kx', ms=14, lw=3)
        plt.plot(ss, xbar, 'k-', lw=2)
        plt.title('{}={}'.format('\lambda', name))
        plt.ylim(-5, 5)

        # plot samples from posterior predictive
        for i in range(3):
            fs = np.zeros(D)
            fs[hidNdx] = gauss.gauss_sample(post_dist['mu'], post_dist['Sigma'])
            fs[obsNdx] = xobs
            plt.plot(ss, fs, 'k-', lw=1)

    plt.show()
          

if __name__ == '__main__':
    main()
