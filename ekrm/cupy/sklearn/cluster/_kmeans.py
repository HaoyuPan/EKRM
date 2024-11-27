import functools
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
# noinspection PyProtectedMember
import sklearn.cluster._kmeans
from cupy import asnumpy as _, array as _arr
# noinspection PyProtectedMember
from sklearn.cluster._kmeans import KMeans as _KMeans

KMeansHistory = namedtuple('KMeansHistory', ['labels', 'inertia', 'centers', 'max_iter'])


@dataclass
class KMeansHistory:
    labels: np.ndarray
    inertia: float
    centers: np.ndarray
    max_iter: int


class KMeans(_KMeans):
    history: list[KMeansHistory]

    # noinspection PyUnresolvedReferences,PyProtectedMember
    @contextmanager
    def log_history(self):
        def wrapper(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
                result = func(*args, **kwargs)
                labels, inertia, centers, max_iter = result
                history = KMeansHistory(labels=labels, inertia=inertia, centers=centers, max_iter=max_iter)
                self.history.append(history)
                return result

            return inner

        self.history = []
        previous = sklearn.cluster._kmeans._kmeans_single_elkan, sklearn.cluster._kmeans._kmeans_single_lloyd
        new = wrapper(previous[0]), wrapper(previous[1])
        sklearn.cluster._kmeans._kmeans_single_elkan, sklearn.cluster._kmeans._kmeans_single_lloyd = new
        yield
        sklearn.cluster._kmeans._kmeans_single_elkan, sklearn.cluster._kmeans._kmeans_single_lloyd = previous

    def fit(self, X, y=None, sample_weight=None):
        X = _(X)
        y = y if y is None else _(y)
        sample_weight = sample_weight if sample_weight is None else _(sample_weight)
        with self.log_history():
            return super().fit(X, y, sample_weight)

    def predict(self, X, sample_weight=None):
        X = _(X)
        sample_weight = sample_weight if sample_weight is None else _(sample_weight)
        return _arr(super().predict(X, sample_weight))
