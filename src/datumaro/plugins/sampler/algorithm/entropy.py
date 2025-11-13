# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import math
import re
from typing import TYPE_CHECKING

from .algorithm import InferenceResultAnalyzer

if TYPE_CHECKING:
    import pandas as pd
else:
    from datumaro.util.import_util import lazy_import

    pd = lazy_import("pandas")


class SampleEntropy(InferenceResultAnalyzer):
    """
    Entropy is a class that inherits an Sampler,
    calculates an uncertainty score based on an entropy,
    and get samples based on that score.
    """

    def __init__(self, data, inference):
        """
        Constructor function

        Args:

            data: Receive the data format in pd.DataFrame format.
                    ImageID is an essential element for data.
            inference:
                Receive the inference format in the form of pd.DataFrame.
                ImageID and ClassProbability are essential for inferences.
        """
        super().__init__(data, inference)

        # check the existence of "ImageID" in data & inference
        if "ImageID" not in data:
            raise Exception("Invalid Data, ImageID not found in data")
        if "ImageID" not in inference:
            raise Exception("Invalid Data, ImageID not found in inference")

        # check the existence of "ClassProbability" in inference
        self.num_classes = 0
        for head in list(inference):
            if re.match(r"ClassProbability\d+", head):
                self.num_classes += 1

        if self.num_classes == 0:
            raise Exception("Invalid data, Inference do not have ClassProbability values")

        # rank: The inference DataFrame, sorted according to the score.
        self.rank = self._rank_images().sort_values(by="rank")

    def get_sample(self, method: str, k: int, n: int = 3) -> pd.DataFrame:
        """
        A function that extracts sample data and returns it.

        Args:
            method:
                - 'topk' - It extracts the k sample data with the
                  highest uncertainty.
                - 'lowk' - It extracts the k sample data with the
                  lowest uncertainty.
                - 'randomk' - Extract and return random k sample data.
            k: number of sample data
            n: Parameters to be used in the randtopk method, Variable to first
                extract data of multiple n of k.
        Returns:
            Extracted sample data : pd.DataFrame
        """
        temp_rank = self.rank

        # 1. k value check
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"Invalid value {k}. k must have an integer greater than zero.")

        # 2. Select a sample according to the method
        if k <= len(temp_rank):
            if method == self.sampling_method.topk.name:
                temp_rank = temp_rank[:k]
            elif method == self.sampling_method.lowk.name:
                temp_rank = temp_rank[-k:]
            elif method == self.sampling_method.randk.name:
                return self.data.sample(n=k).reset_index(drop=True)
            elif method in {self.sampling_method.mixk.name, self.sampling_method.randtopk.name}:
                return self._get_sample_mixed(method=method, k=k, n=n)
            else:
                raise ValueError(f"Unknown sampling method '{method}'")
        else:
            log.warning("The number of samples is greater than the size of the selected subset.")

        columns = list(self.data.columns)
        merged_df = pd.merge(temp_rank, self.data, how="inner", on=["ImageID"])
        return merged_df[columns].reset_index(drop=True)

    def _get_sample_mixed(self, method: str, k: int, n: int = 3) -> pd.DataFrame:
        """
        A function that extracts sample data and returns it.

        Args:
            method:
                - 'mixk': Return top-k and low-k halves based on uncertainty.
                - 'randomtopk': Randomly extract n*k and return k
                  with high uncertainty.
            k: number of sample data
            n: Number to extract n * k from total data according to n,
                and top-k from it
        Returns:
            Extracted sample data : pd.DataFrame
        """
        temp_rank = self.rank

        # Select a sample according to the method
        if k <= len(temp_rank):
            if method == self.sampling_method.mixk.name:
                if k % 2 == 0:
                    temp_rank = pd.concat([temp_rank[: k // 2], temp_rank[-(k // 2) :]])
                else:
                    temp_rank = pd.concat([temp_rank[: (k // 2) + 1], temp_rank[-(k // 2) :]])
            elif method == self.sampling_method.randtopk.name:
                if n * k <= len(temp_rank):
                    temp_rank = temp_rank.sample(n=n * k).sort_values(by="rank")
                else:
                    log.warning(msg="n * k exceeds the length of the inference")
                temp_rank = temp_rank[:k]

        columns = list(self.data.columns)
        merged_df = pd.merge(temp_rank, self.data, how="inner", on=["ImageID"])
        return merged_df[columns].reset_index(drop=True)

    def _rank_images(self) -> pd.DataFrame:
        """
        A internal function that ranks the inference data based on uncertainty.
        Returns:
            inference data sorted by uncertainty. pd.DataFrame
        """
        # 1. Load Inference
        inference, res = None, None
        if self.inference is not None:
            inference = pd.DataFrame(self.inference)
        else:
            raise Exception("Invalid Data, Failed to load inference result")

        # 2. If the reference data frame does not contain an uncertify score, calculate it
        if "Uncertainty" not in inference:
            inference = self._calculate_uncertainty_from_classprob(inference)

        # 3. Check that Uncertainty values are in place.
        na_df = inference.isna().sum()
        if "Uncertainty" in na_df and na_df["Uncertainty"] > 0:
            raise Exception("Some inference results do not have Uncertainty values")

        # 4. Ranked based on Uncertainty score
        res = inference[["ImageID", "Uncertainty"]].groupby("ImageID").mean()
        res["rank"] = res["Uncertainty"].rank(ascending=False, method="first")
        return res.reset_index()

    def _calculate_uncertainty_from_classprob(self, inference: pd.DataFrame) -> pd.DataFrame:
        """
        A function that calculates uncertainty based on entropy through
        ClassProbability values.
        Args:
            inference: Inference data where uncertainty has not been calculated
        Returns:
            inference data with uncertainty variable
        """

        # Calculate Entropy (Uncertainty Score)
        uncertainty = []
        for i in range(len(inference)):
            entropy = 0
            for j in range(self.num_classes):
                p = inference.loc[i][f"ClassProbability{j + 1}"]
                if p < 0 or p > 1:
                    raise Exception("Invalid data, Math domain Error! p is between 0 and 1")
                entropy -= p * math.log(p + 1e-14, math.e)

            uncertainty.append(entropy)

        inference["Uncertainty"] = uncertainty

        return inference
