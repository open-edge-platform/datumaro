import polars as pl
from typing_extensions import Annotated

from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import score_field


def test_score_field_scalar_roundtrip():
    class PredSample(Sample):
        confidence: Annotated[float, score_field()]

    ds = Dataset(PredSample)

    s = PredSample(confidence=0.87)
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, PredSample)
    assert abs(out.confidence - 0.87) < 1e-6


def test_confidence_field_list_roundtrip():
    class PredSample(Sample):
        confidences: Annotated[list[float], score_field(is_list=True)]

    ds = Dataset(PredSample)

    vals = [0.1, 0.5, 0.9]
    s = PredSample(confidences=vals)
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, PredSample)
    assert len(out.confidences) == len(vals)
    for a, b in zip(out.confidences, vals):
        assert abs(a - b) < 1e-6


def test_confidence_field_custom_dtype():
    class PredSample(Sample):
        confidence: Annotated[float, score_field(dtype=pl.Float64)]

    ds = Dataset(PredSample)

    s = PredSample(confidence=0.123456789)
    ds.append(s)

    out = ds[0]
    assert isinstance(out.confidence, float)
    assert abs(out.confidence - 0.123456789) < 1e-12
