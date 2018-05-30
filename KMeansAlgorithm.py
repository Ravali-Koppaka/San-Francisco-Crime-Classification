import math
import random

import numpy as np
from numpy import linalg as la


def same_centroids(centroids, new_centroids):
    if len(centroids) != len(new_centroids):
        return False

    n = len(centroids)
    count = 0
    for i in range(n):
        value = centroids[i]
        for index in range(n):
            new_value = new_centroids[index]
            count1 = 0
            for j in range(len(value)):
                if value[j] == new_value[j]:
                    count1 += 1
            if count1 == len(value):
                count += 1

    return count == n


class KMeans:
    k = None
    random_state = None
    centroid_values = {}
    cluster_values = {}

    def __init__(self, k=1, random_state=1):
        self.k = k
        self.random_state = random_state

    def fit(self, x):
        n = len(x)

        points = x
        points_id = {}
        for i in range(n):
            points_id.setdefault(i, x[i])

        random.seed(self.random_state)

        centroids_choice = random.sample(range(0, n), self.k)
        centroids = []
        for i in range(len(centroids_choice)):
            centroids.append(points_id.get(centroids_choice[i]))

        new_centroids = [np.zeros(len(x[0])) for i in range(self.k)]
        # 2-D binary (0-1) matrix which determines if a point belongs to a cluster
        clusters = np.zeros((n, self.k))
        epoch = 1
        while not same_centroids(centroids, new_centroids):
            # initially new_centroids is empty
            if epoch != 1:
                centroids = new_centroids

            # initialize 2-D matrix to all zeroes for every new round of clustering
            clusters = np.zeros((n, self.k))

            for i in range(n):
                dist = math.inf
                assigned_centroid = 0
                for j in range(self.k):
                    dist_val = la.norm(centroids[j] - points[i])
                    if dist_val < dist:
                        dist = dist_val
                        assigned_centroid = j
                clusters[i, assigned_centroid] = 1

            for i in range(self.k):
                cluster_points = np.zeros(len(x[0]))
                no_of_points = 0
                for j in range(n):
                    if clusters[j, i] == 1:
                        cluster_points += points_id.get(j)
                        no_of_points += 1
                new_centroids[i] = cluster_points/no_of_points
            epoch += 1

        for i in range(self.k):
            points_list = []
            for j in range(n):
                if clusters[j, i] == 1:
                    points_list.append(points_id.get(j))
            self.cluster_values.setdefault(i, points_list)
            self.centroid_values.setdefault(i, new_centroids[i])
