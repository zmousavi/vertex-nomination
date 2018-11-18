#!/usr/bin/env python

# gmm.py
# Copyright (c) 2017. All rights reserved.
# Adopted from JHU 

import numpy as np
from sklearn.mixture import GaussianMixture

def gaussian_clustering(X, max_clusters = 2, min_clusters = 1, acorn=1234):
    """
    Inputs
        X - n x d feature matrix; it is assumed that the d features are ordered
        max_clusters - The maximum number of clusters
        min_clusters - The minumum number of clusters
    Outputs
        Predicted class labels that maximize BIC
    """
    np.random.seed(acorn)   

    n, d = X.shape

    max_clusters = int(round(max_clusters))
    min_clusters = int(round(min_clusters))

    if max_clusters < d:
        X = X[:, :max_clusters].copy()

    cov_types = ['spherical']

    clf = GaussianMixture(n_components = min_clusters, covariance_type = 'spherical', n_init=1, max_iter = 200, init_params = 'kmeans')
    clf.fit(X)
    BIC_max = -clf.bic(X)
    cluster_likelihood_max = min_clusters
    cov_type_likelihood_max = "spherical"

    for i in range(min_clusters, max_clusters + 1):
        for k in cov_types:
            clf = GaussianMixture(n_components=i, 
                                covariance_type=k, n_init=1,  max_iter = 200, init_params = 'kmeans')

            clf.fit(X)

            current_bic = -clf.bic(X)

            if current_bic > BIC_max:
                BIC_max = current_bic
                cluster_likelihood_max = i
                cov_type_likelihood_max = k

    clf = GaussianMixture(n_components = cluster_likelihood_max,
                    covariance_type = cov_type_likelihood_max)
    clf.fit(X)

    predictions = clf.predict(X)
    predictions = np.array([int(i) for i in predictions])

    return predictions, BIC_max