# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# ruff: noqa: E501

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from datumaro.components.annotation import FeatureVector
from datumaro.components.dataset import IDataset
from datumaro.components.launcher import LauncherWithModelInterpreter
from datumaro.util import take_by
from datumaro.util.deprecation import deprecated

if TYPE_CHECKING:
    from scipy import linalg, stats

    from datumaro.plugins.openvino_plugin import shift_launcher
else:
    from datumaro.util.import_util import lazy_import

    linalg = lazy_import("scipy.linalg")
    stats = lazy_import("scipy.stats")
    shift_launcher = lazy_import("datumaro.plugins.openvino_plugin.shift_launcher")


@deprecated(deprecated_version="1.11", removed_version="1.12")
class RunningStats1D:
    def __init__(self):
        self.running_mean = None
        self.running_sq_mean = None
        self.num: int = 0

    def add(self, feats: List[FeatureVector]) -> None:
        arr = np.stack([feat.vector for feat in feats], axis=0)
        assert arr.ndim == 2

        batch_size, _ = arr.shape
        mean = arr.mean(0)
        arr = np.expand_dims(arr, axis=-1)  # B x D x 1
        sq_mean = np.mean(np.matmul(arr, np.transpose(arr, axes=(0, 2, 1))), axis=0)  # D x D

        self.num += batch_size

        if self.running_mean is not None:
            self.running_mean = self.running_mean + batch_size / float(self.num) * (
                mean - self.running_mean
            )
        else:
            self.running_mean = mean

        if self.running_sq_mean is not None:
            self.running_sq_mean = self.running_sq_mean + batch_size / float(self.num) * (
                sq_mean - self.running_sq_mean
            )
        else:
            self.running_sq_mean = sq_mean

    @property
    def mean(self) -> np.ndarray:
        return self.running_mean

    @property
    def cov(self) -> np.ndarray:
        mean = np.expand_dims(self.running_mean, axis=-1)  # D x 1
        return self.running_sq_mean - np.matmul(mean, mean.transpose())


@deprecated(deprecated_version="1.11", removed_version="1.12")
class FeatureAccumulator:
    def __init__(self, model: LauncherWithModelInterpreter):
        self.model = model
        self._batch_size = 1

    def get_activation_stats(self, dataset: IDataset) -> RunningStats1D:
        running_stats = RunningStats1D()

        for batch in take_by(dataset, self._batch_size):
            outputs = self.model.launch(batch)[0]
            features = [outputs[-1]]  # extracted feature vector of googlenet-v4
            running_stats.add(features)

        return running_stats


@deprecated(deprecated_version="1.11", removed_version="1.12")
class FeatureAccumulatorByLabel(FeatureAccumulator):
    def __init__(self, model):
        super().__init__(model)

    def get_activation_stats(self, dataset: IDataset) -> Dict[int, RunningStats1D]:
        running_stats = defaultdict(RunningStats1D)

        for batch in take_by(dataset, self._batch_size):
            inputs, targets = [], []
            for item in batch:
                for ann in item.annotations:
                    inputs.append(np.atleast_3d(item.media.data))
                    targets.append(ann.label)

            outputs = self.model.launch(batch)[0]
            features = [outputs[-1]]  # extracted feature vector of googlenet-v4

            for target in targets:
                running_stats[target].add(features)

        return running_stats


@deprecated(deprecated_version="1.11", removed_version="1.12")
class ShiftAnalyzer:
    def __init__(self) -> None:
        """
        Searcher for Datumaro dataitems

        Parameters
        ----------
        dataset:
            Datumaro dataset to search similar dataitem.
        topk:
            Number of images.
        """
        self._model = shift_launcher.ShiftLauncher(
            model_name="googlenet-v4-tf",
            output_layers="InceptionV4/Logits/PreLogitsFlatten/flatten_1/Reshape:0",
        )

    def compute_covariate_shift(self, sources: List[IDataset], method: Optional[str] = "fid"):
        assert (
            len(sources) == 2
        ), "Shift analyzer should get two datasets to compute shifts between them."

        if method == "fid":
            _feat_aggregator = FeatureAccumulator(model=self._model)

            src_stats = _feat_aggregator.get_activation_stats(sources[0])
            tgt_stats = _feat_aggregator.get_activation_stats(sources[1])

            src_mu, src_sigma = src_stats.mean, src_stats.cov
            tgt_mu, tgt_sigma = tgt_stats.mean, tgt_stats.cov

            return self._frechet_distance(src_mu, src_sigma, tgt_mu, tgt_sigma, atol=1e-3)

        elif method == "emd":
            _feat_aggregator = FeatureAccumulatorByLabel(model=self._model)

            src_stats = _feat_aggregator.get_activation_stats(sources[0])
            tgt_stats = _feat_aggregator.get_activation_stats(sources[1])

            w_s = np.array([stats.num for stats in src_stats.values()])
            w_t = np.array([stats.num for stats in tgt_stats.values()])

            f_s = np.stack([stats.mean for stats in src_stats.values()], axis=0)
            f_t = np.stack([stats.mean for stats in tgt_stats.values()], axis=0)

            # earth_mover_distance returns the similarity score in [0, 1].
            # We return the dissimilarity score by 1 - similarity score.
            return 1.0 - self._earth_mover_distance(w_s, f_s, w_t, f_t, gamma=0.01)

    def compute_label_shift(self, sources: List[IDataset]):
        assert (
            len(sources) == 2
        ), "Shift analyzer should get two datasets to compute shifts between them."

        labels = defaultdict(list)
        for idx, source in enumerate(sources):
            for item in source:
                for ann in item.annotations:
                    labels[idx].append(ann.label)

        _, _, pv = stats.anderson_ksamp([labels[0], labels[1]])

        return 1 - pv

    def _frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
        atol: float = 1e-3,
    ):
        """
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        We borrowed the implementation of [1]_ (Apache 2.0 license).
        Our implementation forces 64-bit floating-type calculations to avoid numerical instability.
        Parameters
        ----------
        mu1
            Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
        mu2
            The sample mean over activations, precalculated on an representative data set.
        sigma1
            The covariance matrix over activations for generated samples.
        sigma2
            The covariance matrix over activations, precalculated on an representative data set.
        eps
            Epsilone term to the diagonal part of sigma covariance matrix.
        atol
            Threshold value to check whether the covariance matrix is real valued.
            If any imagenary diagonal part of the covariance matrix is greather than `atol`,
            raise `ValueError`.
        Returns
        -------
        Distance
            Frechet distance
        References
        ----------
        .. [1] https://github.com/mseitzer/pytorch-fid/blob/3d604a25516746c3a4a5548c8610e99010b2c819/src/pytorch_fid/fid_score.py#L150
        """
        mu1 = np.atleast_1d(mu1).astype(np.float64)
        mu2 = np.atleast_1d(mu2).astype(np.float64)

        sigma1 = np.atleast_2d(sigma1).astype(np.float64)
        sigma2 = np.atleast_2d(sigma2).astype(np.float64)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths."
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions."

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates."
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=atol):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def _earth_mover_distance(
        self,
        w_s: np.ndarray,
        f_s: np.ndarray,
        w_t: np.ndarray,
        f_t: np.ndarray,
        gamma: float,
    ) -> float:
        w_1 = np.zeros((len(w_s) + len(w_t),), np.float64)
        w_2 = np.zeros((len(w_s) + len(w_t),), np.float64)
        w_1[: len(w_s)] = w_s / np.sum(w_s)
        w_2[len(w_s) :] = w_t / np.sum(w_t)

        f_concat = np.concatenate([f_s, f_t], axis=0)
        distances = np.linalg.norm(f_concat[:, None] - f_concat[None, :], axis=2).astype(np.float64)

        import pyemd

        emd = pyemd.emd(w_1, w_2, distances)
        return np.exp(-gamma * emd).item()
