#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import os
import zipfile
import requests
from scipy.io import loadmat


def load_data(dataset):
    datadir = './downloaded_data'
    matfile = ('{}/{}/{}.mat'.format(datadir, dataset, dataset))
    destdir = os.path.dirname(matfile)
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    repo = 'github.com/okomestudio/pmtkdata'
    url = ('https://{repo}/raw/master/{dataset}/{dataset}.zip'
           .format(repo=repo, dataset=dataset))

    trial = 0
    while not os.path.exists(matfile):
        try:
            zf = zipfile.ZipFile(StringIO(requests.get(url).content), 'r')
            with open(matfile, 'w') as f:
                f.write(zf.read('{dataset}/{dataset}.mat'
                                .format(dataset=dataset)))
        except:
            os.remove(path)
            raise

        trial += 1
        if trial > 3:
            raise RuntimeError('dataset {} cannot be downloaded'
                               .format(dataset))

    return loadmat(matfile)
