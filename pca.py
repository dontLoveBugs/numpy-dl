#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-13 20:20
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : pca.py
"""

import numpy as np

class PCA(object):

    def __init__(self, n_component, solver="pca"):
        self.n_component=n_component
        assert solver in ["pca", "svd"], "only support pca and svd solver."
        self.solver=solver

    def fit(self, data):
        """
        :param data: numpy.array, NXD
        """
        if self.solver=="pca":
            d_mean = data - np.mean(data, axis=0, keepdims=True) # meke mean=0, like Batch norm.
            cov=np.dot(d_mean.T, d_mean) / d_mean.shape[0] # covariance matrix
            eigen_value, eigen_vector=np.linalg.eig(cov)
            eigen_index = np.argsort(eigen_value)[::-1]
            self._components = eigen_vector[:, eigen_index][::, 0:self.n_component]
            self._component_values = eigen_value[eigen_index][0:self.n_component]
        else:
            d_mean=data-np.mean(data, axis=0, keepdims=True)
            u, sigma, v=np.linalg.svd(d_mean, full_matrices=False)
            print("u:", u.shape, "sigma:", sigma.shape, "v:", v.shape)
            eigen_value=sigma**2 / (d_mean.shape[0]-1)
            eigen_vector=v.T
            eigen_index = np.argsort(eigen_value)[::-1]
            self._components = eigen_vector[:, eigen_index][::, 0:self.n_component]
            self._component_values = eigen_value[eigen_index][0:self.n_component]


    def transform(self, data):
        pca_data = np.dot(data, self._components)
        return pca_data