# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from enum import Enum, auto

import cv2
import numpy as np
from scipy.linalg import orth

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME
from datumaro.components.transformer import Transform
from datumaro.util import parse_str_enum_value


class Algorithm(Enum):
    gradient = auto()
    # other algorithms will be added


class OverSamplingMethod(Enum):
    random = auto()
    similarity = auto()


class UnderSamplingMethod(Enum):
    uniform = auto()
    inverse = auto()


class NDR(Transform, CliPlugin):
    """
    Removes near-duplicated images in subset|n
    |n
    Remove duplicated images from a dataset. Keep at most `-k/--num_cut`
    resulting images.|n
    |n
    Available oversampling policies (the `-e` parameter):|n
    |s|s- `random` - sample from removed data randomly|n
    |s|s- `similarity` - sample from removed data with ascending similarity score|n
    |n
    Available undersampling policies (the `-u` parameter):|n
    |s|s- `uniform` - sample data with uniform distribution|n
    |s|s- `inverse` - sample data with reciprocal of the number of number of
    items with the same similarity|n
    |n
    Example: apply NDR, return no more than 100 images|n

    .. code-block::

    |s|s%(prog)s|n
    |s|s|s|s--working_subset train|n
    |s|s|s|s--algorithm gradient|n
    |s|s|s|s--num_cut 100|n
    |s|s|s|s--over_sample random|n
    |s|s|s|s--under_sample uniform
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-w",
            "--working_subset",
            default=None,
            help="Name of the subset to operate (default: %(default)s)",
        )
        parser.add_argument(
            "-d",
            "--duplicated_subset",
            default="duplicated",
            help="Name of the subset for the removed data " "after NDR runs (default: %(default)s)",
        )
        parser.add_argument(
            "-a",
            "--algorithm",
            default=Algorithm.gradient.name,
            choices=[algo.name for algo in Algorithm],
            help="Name of the algorithm to use (default: %(default)s)",
        )
        parser.add_argument(
            "-k", "--num_cut", default=None, type=int, help="Maximum output dataset size"
        )
        parser.add_argument(
            "-e",
            "--over_sample",
            default=OverSamplingMethod.random.name,
            choices=[method.name for method in OverSamplingMethod],
            help="The policy to use when num_cut is bigger "
            "than result length (default: %(default)s)",
        )
        parser.add_argument(
            "-u",
            "--under_sample",
            default=UnderSamplingMethod.uniform.name,
            choices=[method.name for method in UnderSamplingMethod],
            help="The policy to use when num_cut is smaller "
            "than result length (default: %(default)s)",
        )
        parser.add_argument("-s", "--seed", type=int, help="Random seed")
        return parser

    def __init__(
        self,
        extractor,
        working_subset,
        duplicated_subset="duplicated",
        algorithm=None,
        num_cut=None,
        over_sample=None,
        under_sample=None,
        seed=None,
        **kwargs,
    ):
        """
        Near-duplicated image removal

        Arguments
        ---------
        working_subset: str
            name of the subset to operate
            if None, use DEFAULT_SUBSET_NAME
        duplicated_subset: str
            name of the subset for the removed data after NDR runs
        algorithm: str
            name of the algorithm to use
            "gradient" only for now.
        num_cut: int
            number of outputs you want.
            the algorithm will cut whole dataset to this amount
            if None, return result without any modification
        over_sample: "random" or "similarity"
            specify the strategy when num_cut > length of the result after removal
            if random, sample from removed data randomly
            if similarity, select from removed data with ascending
            order of similarity
        under_sample: "uniform" or "inverse"
            specify the strategy when num_cut < length of the result after removal
            if uniform, sample data with uniform distribution
            if inverse, sample data with reciprocal of the number
            of data which have same hash key

        Algorithm Specific for gradient
            block_shape: tuple, (h, w)
                for the robustness, this function will operate on blocks
                mean and variance will be calculated on this block
            hash_dim: int
                dimension(or bit) of the hash function
            sim_threshold: float
                the threshold value for saving hash-collided samples.
                larger value means more generous, i.e., saving more samples
        Return
        ---------------
        None, other subsets combined with the result
        """
        super().__init__(extractor)

        if not working_subset:
            working_subset = DEFAULT_SUBSET_NAME
        if working_subset not in extractor.subsets():
            raise ValueError("Invalid working_subset name")
        self.working_subset = working_subset

        # parameter validation before main runs
        if working_subset == duplicated_subset:
            raise ValueError("working_subset == duplicated_subset")

        algorithm = parse_str_enum_value(
            algorithm,
            Algorithm,
            default=Algorithm.gradient,
            unknown_member_error="Unknown algorithm '{value}'.",
        )
        over_sample = parse_str_enum_value(
            over_sample,
            OverSamplingMethod,
            default=OverSamplingMethod.random,
            unknown_member_error="Unknown oversampling method '{value}'.",
        )
        under_sample = parse_str_enum_value(
            under_sample,
            UnderSamplingMethod,
            default=UnderSamplingMethod.uniform,
            unknown_member_error="Unknown undersampling method '{value}'.",
        )

        if seed:
            self.seed = seed
        else:
            self.seed = None
        self.working_subset = working_subset
        self.duplicated_subset = duplicated_subset
        self.algorithm = algorithm

        self.num_cut = num_cut
        self.over_sample = over_sample
        self.under_sample = under_sample

        self.algorithm_specific = kwargs
        self._initialized = False

    def _remove(self):
        if self.seed:
            np.random.seed(self.seed)

        working_subset = self._extractor.get_subset(self.working_subset)
        having_image = []
        all_imgs = []
        for item in working_subset:
            if item.media.has_data:
                having_image.append(item)
                img = item.media.data
                # Not handle empty image, as utils/image.py if check empty
                if len(img.shape) == 2:
                    img = np.stack((img,) * 3, axis=-1)
                elif len(img.shape) == 3:
                    if img.shape[2] == 1:
                        img = np.stack((img[:, :, 0],) * 3, axis=-1)
                    elif img.shape[2] == 4:
                        img = img[..., :3]
                    elif img.shape[2] == 3:
                        pass
                    else:
                        raise ValueError(
                            "Item %s: invalid image shape: "
                            "unexpected number of channels (%s)" % (item.id, img.shape[2])
                        )
                else:
                    raise ValueError(
                        "Item %s: invalid image shape: "
                        "unexpected number of dimensions (%s)" % (item.id, len(img.shape))
                    )

                if self.algorithm == Algorithm.gradient:
                    # Calculate gradient
                    img = self._cgrad_feature(img)
                else:
                    raise NotImplementedError()
                all_imgs.append(img)
            else:
                log.debug("Skipping item %s: no image data available", item.id)

        if self.num_cut and self.num_cut > len(all_imgs):
            raise ValueError("The number of images is smaller than the cut you want")

        if self.algorithm == Algorithm.gradient:
            all_key, fidx, kept_index, key_counter, removed_index_with_sim = self._gradient_based(
                all_imgs, **self.algorithm_specific
            )
        else:
            raise NotImplementedError()

        kept_index = self._keep_cut(
            self.num_cut,
            all_key,
            fidx,
            kept_index,
            key_counter,
            removed_index_with_sim,
            self.over_sample,
            self.under_sample,
        )
        self.kept_item_id = set(having_image[ii].id for ii in kept_index)

    def _gradient_based(self, all_imgs, block_shape=(4, 4), hash_dim=32, sim_threshold=0.5):
        if len(block_shape) != 2:
            raise ValueError("Invalid block_shape")
        if block_shape[0] <= 0 or block_shape[1] <= 0:
            raise ValueError("block_shape should be positive")
        if sim_threshold <= 0:
            raise ValueError("sim_threshold should be large than 0")
        if hash_dim > 3 * block_shape[0] * block_shape[1]:
            raise ValueError("hash_dim should be smaller than feature shape")
        if hash_dim <= 0:
            raise ValueError("hash_dim should be positive")

        # Compute hash keys from all the features
        all_clr = np.reshape(np.array(all_imgs), (len(all_imgs), -1))
        all_key = self._project(all_clr, hash_dim)

        # Remove duplication using hash
        clr_dict = {}
        key_counter = {}
        kept_index = []
        removed_index_with_similarity = dict()

        fidx = np.random.permutation(np.arange(len(all_imgs)))
        for ii in fidx:
            key = all_key[ii]
            clr = all_clr[ii]
            if key not in clr_dict:
                clr_dict[key] = [clr]
                key_counter[key] = 1
                kept_index.append(ii)
                continue

            # Hash collision: compare dot-product based feature similarity
            # the value for maximizing the gap
            # between duplicated and non-duplicated
            large_exponent = 50
            max_sim = np.max(np.dot(clr_dict[key], clr) ** large_exponent)

            # Keep if not a duplicated one
            if max_sim < sim_threshold:
                clr_dict[key].append(clr)
                key_counter[key] += 1
                kept_index.append(ii)
            else:
                removed_index_with_similarity[ii] = max_sim
        return all_key, fidx, kept_index, key_counter, removed_index_with_similarity

    def _keep_cut(
        self,
        num_cut,
        all_key,
        fidx,
        kept_index,
        key_counter,
        removed_index_with_similarity,
        over_sample,
        under_sample,
    ):
        if num_cut and num_cut > len(kept_index):
            if over_sample == OverSamplingMethod.random:
                selected_index = np.random.choice(
                    list(set(fidx) - set(kept_index)), size=num_cut - len(kept_index), replace=False
                )
            elif over_sample == OverSamplingMethod.similarity:
                removed_index_with_similarity = [
                    [key, value] for key, value in removed_index_with_similarity.items()
                ]
                removed_index_with_similarity.sort(key=lambda x: x[1])
                selected_index = [
                    index for index, _ in removed_index_with_similarity[: num_cut - len(kept_index)]
                ]
            kept_index.extend(selected_index)
        elif num_cut and num_cut < len(kept_index):
            if under_sample == UnderSamplingMethod.uniform:
                prob = None
            elif under_sample == UnderSamplingMethod.inverse:
                # if inverse - probability with inverse
                # of the collision(number of same hash key)
                # [x1, x2, y1, y2, y3, y4, z1, z2, z3]. x, y and z for hash key
                # i.e. there are 4 elements which have hash key y.
                # then the occurrence will be [2, 4, 3] and reverse of them
                # will be [1/2, 1/4, 1/3]
                # Normalizing them by dividing with sum, we get [6/13, 3/13, 4/13]
                # Then the key x will be sampled with probability 6/13
                # and each point, x1 and x2, will share same prob. 3/13
                key_with_reverse_occur = {key: 1 / key_counter[key] for key in key_counter}
                reverse_occur_sum = sum(key_with_reverse_occur.values())
                key_normalized_reverse_occur = {
                    key: reverse_occur / reverse_occur_sum
                    for key, reverse_occur in key_with_reverse_occur.items()
                }
                prob = [
                    key_normalized_reverse_occur[all_key[ii]] / key_counter[all_key[ii]]
                    for ii in kept_index
                ]
            kept_index = np.random.choice(kept_index, size=num_cut, replace=False, p=prob)

        return kept_index

    @staticmethod
    def _cgrad_feature(img, out_wh=(8, 8)):
        if img.dtype == "uint8":
            img = img.astype(float) / 255.0
        else:
            img = img.astype(float)

        r_img = cv2.resize(img, out_wh, interpolation=cv2.INTER_AREA)
        r2 = cv2.resize(img**2, out_wh, interpolation=cv2.INTER_AREA)

        r2 -= r_img**2
        r2 = np.sqrt(np.maximum(r2, 0))

        # mean and variance feature, zero padding for gradient computation
        rr = np.pad(np.concatenate([r_img, r2], axis=-1), ((1, 1), (1, 1), (0, 0)))

        # compute gradients along x- and y-axes
        rx = rr[1:-1, :-2, :] - rr[1:-1, 2:, :]
        ry = rr[:-2, 1:-1, :] - rr[2:, 1:-1, :]

        # concat and l2 normalize
        res = np.concatenate([rx, ry], axis=-1)
        res = res / np.sqrt(np.sum(res**2))
        return res

    @staticmethod
    def _project(feat, hash_dim=32):
        """
        Project feat to hash_dim space and create hexadecimal string key

        Arguments
        ------------
        feat : ndarray
            feature to project and hash
        hash_dim : int
            specified dimension of the hashed output
        feature dimension should larger than hash_dim
        """
        proj = None
        ndim = feat.shape[-1]
        feat = np.reshape(feat, (-1, ndim))
        # random projection matrix would become unstable, so reject such cases
        assert ndim >= hash_dim, "{} is smaller than hash_dim({})".format(ndim, hash_dim)

        # compute the random projection matrix
        for _ in range(100):
            # try to get an orthonormal projection matrix
            proj = orth(np.random.uniform(-1, 1, (ndim, ndim)))[:, :hash_dim]
            if proj.shape[1] == hash_dim:
                break
        if proj is None:
            # if failed to get an orthonormal one, just use a random one instead
            proj = np.random.uniform(-1, 1, (ndim, ndim))
            proj /= np.sqrt(np.sum(proj**2, axis=1, keepdims=True))

        # simple binarization
        # compute dot product between each feature and each projection basis,
        # then use its sign for the binarization
        feat_binary = np.dot(feat, proj) >= 0

        # generate hash key strings
        # assign hex string from each consecutive 16 bits and concatenate
        _all_key = np.packbits(feat_binary, axis=-1)
        _all_key = np.array(
            list(map(lambda row: "".join(["{:02x}".format(r) for r in row]), _all_key))
        )
        if len(_all_key) == 1:
            return _all_key[0]
        else:
            return _all_key

    def _check_subset(self, item):
        if item.subset:
            if item.subset == self.working_subset:
                if item.id in self.kept_item_id:
                    return item.subset
                else:
                    return self.duplicated_subset
            else:
                return item.subset
        else:
            return DEFAULT_SUBSET_NAME

    def __iter__(self):
        if not self._initialized:
            self._remove()
            self._initialized = True
        for item in self._extractor:
            yield self.wrap_item(item, subset=self._check_subset(item))
