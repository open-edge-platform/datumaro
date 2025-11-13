# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Union

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import IDataset
from datumaro.components.transformer import Transform
from datumaro.util import parse_str_enum_value

from .algorithm.algorithm import Algorithm, SamplingMethod

if TYPE_CHECKING:
    import pandas as pd
else:
    from datumaro.util.import_util import lazy_import

    pd = lazy_import("pandas")


class RelevancySampler(Transform, CliPlugin):
    r"""
    Sampler that analyzes model inference results on the dataset |n
    and picks the best sample for training.|n
    |n
    Creates a dataset from the `-k/--count` hardest items for a model.
    The whole dataset or a single subset will be split into the `sampled`
    and `unsampled` subsets based on the model confidence.
    The dataset **must** contain model confidence
    values in the `scores` attributes of annotations.|n
    |n
    There are five methods of sampling (the `-m/--method` option):|n
    |s|s- `topk` - Return the k items with the highest uncertainty data|n
    |s|s- `lowk` - Return the k items with the lowest uncertainty data|n
    |s|s- `randk` - Return random k items|n
    |s|s- `mixk` - Return a half using topk, and the other half using lowk method|n
    |s|s- `randtopk` - Select 3*k items randomly, and return the topk among them|n
    |n
    Notes:|n
    |s|s- Each image's inference result must contain the probability for|n
    |s|s|s|sall classes.|n
    |s|s- Requesting a sample larger than the number of all images will|n
    |s|s|s|sreturn all images.|n
    |n
    Example: select the most relevant data subset of 20 images |n
    |s|sbased on model certainty, put the result into 'sample' subset |n
    |s|sand put all the rest into 'unsampled' subset, use 'train' subset |n
    |s|sas input. |n

    .. code-block::

    |s|s%(prog)s \ |n
    |s|s|s|s--algorithm entropy \ |n
    |s|s|s|s--subset_name train \ |n
    |s|s|s|s--sample_name sample \ |n
    |s|s|s|s--unsampled_name unsampled \ |n
    |s|s|s|s--sampling_method topk -k 20
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("-k", "--count", type=int, required=True, help="Number of items to sample")
        parser.add_argument(
            "-a",
            "--algorithm",
            default=Algorithm.entropy.name,
            choices=[t.name for t in Algorithm],
            help="Sampling algorithm (one of {}; default: %(default)s)".format(", ".join(t.name for t in Algorithm)),
        )
        parser.add_argument(
            "-i",
            "--input_subset",
            default=None,
            help="Subset name to select sample from (default: %(default)s)",
        )
        parser.add_argument(
            "-o",
            "--sampled_subset",
            default="sample",
            help="Subset name to put sampled data to (default: %(default)s)",
        )
        parser.add_argument(
            "-u",
            "--unsampled_subset",
            default="unsampled",
            help="Subset name to put the rest data to (default: %(default)s)",
        )
        parser.add_argument(
            "-m",
            "--sampling_method",
            default=SamplingMethod.topk.name,
            choices=[t.name for t in SamplingMethod],
            help="Sampling method (one of {}; default: %(default)s)".format(", ".join(t.name for t in SamplingMethod)),
        )
        parser.add_argument("-d", "--output_file", help="A .csv file path to dump sampling results")
        return parser

    def __init__(
        self,
        extractor: IDataset,
        count: int,
        *,
        algorithm: Union[str, Algorithm],
        sampling_method: Union[str, SamplingMethod],
        input_subset: Optional[str] = None,
        sampled_subset: str = "sample",
        unsampled_subset: str = "unsampled",
        output_file: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        extractor
        algorithm
            Specifying the algorithm to calculate the uncertainty
            for sample selection. default: 'entropy'
        subset_name
            The name of the subset to which you want to select a sample.
        sample_name
            Subset name of the selected sample, default: 'sample'
        sampling_method
            Method of sampling, 'topk' or 'lowk' or 'randk'
        count
            Number of samples extracted
        output_file
            A path to .csv file for sampling results
        """
        super().__init__(extractor)

        self.input_subset = input_subset
        self.sampled_subset = sampled_subset
        self.unsampled_subset = unsampled_subset
        self.algorithm = parse_str_enum_value(algorithm, Algorithm).name
        self.sampling_method = parse_str_enum_value(sampling_method, SamplingMethod).name
        self.count = count
        self.output_file = output_file

        # Use the --output_file option to save the sample list as a csv file
        if output_file and not output_file.endswith(".csv"):
            raise ValueError("The output file must have the '.csv' extension")

    @staticmethod
    def _load_inference_from_subset(extractor, subset_name):
        # 1. Get Dataset from subset name
        if subset_name in extractor.subsets():
            subset = extractor.get_subset(subset_name)
        else:
            raise Exception(f"Unknown subset '{subset_name}'")

        data_df = defaultdict(list)
        infer_df = defaultdict(list)

        # 2. Fill the data_df and infer_df to fit the sampler algorithm
        # input format.
        for item in subset:
            data_df["ImageID"].append(item.id)

            if not item.media or item.media.size is None:
                raise Exception(f"Item {item.id} does not have image info")

            width, height = item.media.size
            data_df["Width"].append(width)
            data_df["Height"].append(height)
            data_df["ImagePath"].append(getattr(item.media, "path", None))

            if not item.annotations:
                raise Exception(f"Item {item.id} does not have annotations")

            for annotation in item.annotations:
                if "scores" not in annotation.attributes:
                    raise Exception(f"Item {item.id} - an annotation does not have 'scores' attribute")
                probs = annotation.attributes["scores"]

                infer_df["ImageID"].append(item.id)

                for prob_idx, prob in enumerate(probs):
                    infer_df[f"ClassProbability{prob_idx + 1}"].append(prob)

        data_df = pd.DataFrame(data_df)
        infer_df = pd.DataFrame(infer_df)

        return data_df, infer_df

    @staticmethod
    def _calculate_uncertainty(algorithm, data, inference):
        # Checking and creating algorithms
        if algorithm == Algorithm.entropy.name:
            from .algorithm.entropy import SampleEntropy

            # Data delivery, uncertainty score calculations also proceed.
            sampler = SampleEntropy(data, inference)
        else:
            raise Exception(f"Unknown algorithm '{algorithm}', available algorithms: {[a.name for a in Algorithm]}")
        return sampler

    def _get_sample_subset(self, image):
        if image.subset == self.input_subset:
            # 1. Returns the sample subset if the id belongs to samples.
            if image.id in self.sample_id:
                return self.sampled_subset
            return self.unsampled_subset
        # 2. Returns the existing subset name if it is not a sample
        return image.subset

    def __iter__(self):
        # Import data into a subset name and convert it
        # to a format that will be used in the sampler algorithm with
        # the inference result.
        data_df, infer_df = self._load_inference_from_subset(self._extractor, self.input_subset)

        sampler = self._calculate_uncertainty(self.algorithm, data_df, infer_df)
        self.result = sampler.get_sample(method=self.sampling_method, k=self.count)

        if self.output_file is not None:
            self.result.to_csv(self.output_file, index=False)

        self.sample_id = self.result["ImageID"].to_list()

        # Transform properties for each data
        for item in self._extractor:
            # After checking whether each item belongs to a sample,
            # rename the subset
            yield self.wrap_item(item, subset=self._get_sample_subset(item))
