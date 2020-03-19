#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-19 16:56
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : kmeans.py
"""

from utils import distance
import numpy as np


class Kmeans(object):

    def __init__(self, k):
        self.K = k
        self.centroids = None

    def _randCentroids(self, data):
        """
        :param data: numpy.array (n, d)
        """
        n, d = data.shape
        choices = np.linspace(0, n - 1, num=n, dtype=np.int)
        centroids_index = np.random.choice(choices, size=self.K)
        self.centroids = np.copy(data[centroids_index])

    def fit(self, data):
        n, d = data.shape
        self._randCentroids(data)

        converged = False
        cluster_labels = np.zeros(n)

        while not converged:
            old_centroids = np.copy(self.centroids)
            # update cluster labels
            for i in range(n):
                cluster_labels[i] = np.argmin(distance(data[i], self.centroids), axis=0)

            # update centorids
            for i in range(self.K):
                index = np.nonzero(cluster_labels == i)
                self.centroids[i] = np.mean(data[index], axis=0)
            converged = np.sum(np.abs(self.centroids - old_centroids))

    def predict(self, data):
        n, d = data.shape
        labels = np.zeros(n)
        for i in range(n):
            labels = np.argmin(distance(data[i], self.centroids), axis=0)
        return labels
