import numpy as np
import polars as pl
import pytest

from datumaro.experimental.dataset import AttributeInfo
from datumaro.experimental.fields import (
    TileInfo,
    bbox_field,
    image_field,
    image_info_field,
    instance_mask_field,
    label_field,
    mask_field,
    polygon_field,
    tile_field,
)
from datumaro.experimental.schema import Schema
from datumaro.experimental.tiling.tiler_registry import (
    AttributeSpec,
    TilingConfig,
    _apply_tiling,
    _calculate_tiles,
    _create_tiling_plan,
)


@pytest.fixture
def sample_schema():
    return Schema(
        attributes={
            "image": AttributeInfo(type=np.ndarray, annotation=image_field(pl.UInt8())),
            "image_info": AttributeInfo(type=dict, annotation=image_info_field()),
        }
    )


@pytest.fixture
def sample_df():
    # Create sample image data
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    image2 = np.ones((200, 150, 3), dtype=np.uint8)

    # Flatten images and get shapes
    flat_image1 = image1.reshape(-1)
    flat_image2 = image2.reshape(-1)
    shape1 = list(image1.shape)
    shape2 = list(image2.shape)

    # Create sample image info
    info1 = {"height": 100, "width": 100}
    info2 = {"height": 200, "width": 150}

    # Create DataFrame with explicit schema
    return pl.DataFrame(
        {
            "image": [flat_image1, flat_image2],
            "image_shape": [shape1, shape2],
            "image_info": [info1, info2],
        },
        schema={
            "image": pl.List(pl.UInt8),
            "image_shape": pl.List(pl.Int32),
            "image_info": pl.Struct([pl.Field("height", pl.Int32), pl.Field("width", pl.Int32)]),
        },
    )


def test_calculate_tiles(sample_df):
    config = TilingConfig(tile_width=50, tile_height=50)

    schema = Schema(
        attributes={
            "image_info": AttributeInfo(type=dict, annotation=image_info_field()),
            "tile": AttributeInfo(type=dict, annotation=tile_field()),
        }
    )

    # Create specs from schema
    image_info_spec = AttributeSpec(
        name="image_info", field=schema.attributes["image_info"].annotation
    )
    tile_info_spec = AttributeSpec(name="tile", field=schema.attributes["tile"].annotation)

    # Calculate tiles
    tiles_df = _calculate_tiles(sample_df, config, image_info_spec, tile_info_spec)

    # Check that we have the right number of tiles
    # First image (100x100) should have 4 tiles (2x2 grid)
    # Second image (200x150) should have 12 tiles (4x3 grid)
    assert len(tiles_df) == 16

    # Verify tile parameters
    tiles = [TileInfo(**row["tile"]) for row in tiles_df.to_dicts()]

    # Check first image tiles
    first_image_tiles = tiles[:4]
    for tile in first_image_tiles:
        assert tile.width <= 50
        assert tile.height <= 50
        assert tile.x % 50 == 0
        assert tile.y % 50 == 0

    # Check second image tiles
    second_image_tiles = tiles[4:]
    for tile in second_image_tiles:
        assert tile.width <= 50
        assert tile.height <= 50
        assert tile.x % 50 == 0
        assert tile.y % 50 == 0


def test_apply_tiling(sample_df, sample_schema):
    config = TilingConfig(tile_width=50, tile_height=50)

    # Apply tiling
    plan = _create_tiling_plan(sample_schema, config)
    result_df, result_fields = _apply_tiling(sample_df, None, plan, ["image", "image_info"])

    # Check the generated fields
    assert result_fields == {"image", "image_info"}

    # Check DataFrame
    assert "tile" in result_df.columns
    assert "image_info" in result_df.columns
    assert "image" in result_df.columns
    assert len(result_df) == 16  # Same as test_calculate_tiles

    # Check image info field values
    for row in result_df.iter_rows(named=True):
        tile_info = row["image_info"]
        # Each tile should have dimensions <= config size
        assert tile_info["width"] <= config.tile_width
        assert tile_info["height"] <= config.tile_height
        # Should track original image
        assert "source_sample_idx" in tile_info


def test_tiling_with_overlap(sample_df, sample_schema):
    config = TilingConfig(tile_width=50, tile_height=50, overlap_x=10, overlap_y=10)

    # Apply tiling
    plan = _create_tiling_plan(sample_schema, config)
    result_df, _ = _apply_tiling(sample_df, None, plan, [])

    # Convert tile data to TileInfo objects
    tiles = [TileInfo(**row["tile"]) for row in result_df.to_dicts()]

    # Check overlap between adjacent tiles
    for i in range(len(tiles) - 1):
        tile = tiles[i]
        next_tile = tiles[i + 1]

        # If tiles are from the same row
        if tile.source_sample_idx == next_tile.source_sample_idx and tile.y == next_tile.y:
            overlap = (tile.x + tile.width) - next_tile.x
            assert overlap == config.overlap_x

        # If tiles are from the same column
        elif tile.source_sample_idx == next_tile.source_sample_idx and tile.x == next_tile.x:
            overlap = (tile.y + tile.height) - next_tile.y
            assert overlap == config.overlap_y


def test_invalid_schema():
    # Create schema without image info
    invalid_schema = Schema(
        attributes={
            "image": AttributeInfo(type=np.ndarray, annotation=image_field(pl.UInt8())),
        }
    )

    df = pl.DataFrame(
        {"image": [np.zeros((100, 100, 3)).flatten()], "image_shape": [(100, 100, 3)]}
    )
    config = TilingConfig(tile_width=50, tile_height=50)

    # Should raise error due to missing ImageInfoField
    with pytest.raises(ValueError, match="Schema must contain an ImageInfoField"):
        _create_tiling_plan(invalid_schema, config)


def test_mask_tiling():
    # Create schema with both types of masks
    schema = Schema(
        attributes={
            "image_info": AttributeInfo(type=dict, annotation=image_info_field()),
            "segmentation": AttributeInfo(type=np.ndarray, annotation=mask_field()),
            "instances": AttributeInfo(type=list, annotation=instance_mask_field()),
        }
    )

    # Create sample data
    # Semantic segmentation: 100x100 with two classes (0 and 1)
    semantic_mask = np.zeros((100, 100), dtype=np.int32)
    semantic_mask[25:75, 25:75] = 1  # Class 1 square in middle

    # Instance segmentation: 2 instances x 100 x 100
    instances = np.zeros((2, 100, 100), dtype=bool)
    instances[0, 20:70, 20:70] = True  # Large instance in middle
    instances[1, 80:90, 80:90] = True  # Small instance in corner

    df = pl.DataFrame(
        {
            "image_info": [
                {"width": 100, "height": 100},
            ],
            "segmentation": semantic_mask.reshape(1, -1),
            "segmentation_shape": [semantic_mask.shape],
            "instances": instances.reshape(1, -1),
            "instances_shape": [instances.shape],
        }
    )

    # Configure tiling to create 50x50 tiles
    config = TilingConfig(tile_width=50, tile_height=50)

    # Apply tiling
    plan = _create_tiling_plan(schema, config)
    result_df, _ = _apply_tiling(df, None, plan, ["segmentation", "instances"])

    # There should be 4 tiles (2x2 grid)
    assert len(result_df) == 4

    # Check first tile (top-left)

    # Check semantic segmentation tile
    assert result_df[0, "segmentation"].shape == (50 * 50,)
    assert tuple(result_df[0, "segmentation_shape"]) == (50, 50)
    assert np.any(result_df[0, "segmentation"].to_numpy() == 1)  # Should contain class 1

    # Check instance segmentation tile
    assert result_df[0, "instances"].shape == (2 * 50 * 50,)
    assert tuple(result_df[0, "instances_shape"]) == (2, 50, 50)
    instances = result_df[0, "instances"].reshape((2, 50, 50)).to_numpy()
    assert np.any(instances[0])
    assert not np.any(instances[1])

    # Check last tile (bottom-right)

    # Check semantic segmentation tile
    assert result_df[3, "segmentation"].shape == (50 * 50,)
    assert tuple(result_df[3, "segmentation_shape"]) == (50, 50)
    assert np.any(result_df[3, "segmentation"].to_numpy() == 1)  # Should contain class 1

    # Check instance segmentation tile
    assert result_df[3, "instances"].shape == (2 * 50 * 50,)
    assert tuple(result_df[3, "instances_shape"]) == (2, 50, 50)
    instances = result_df[3, "instances"].reshape((2, 50, 50)).to_numpy()
    assert np.any(instances[0])  # First instance not present
    assert np.any(instances[1])  # Second instance present


def test_bbox_tiling():
    # Create schema with bounding boxes
    schema = Schema(
        attributes={
            "image_info": AttributeInfo(type=dict, annotation=image_info_field()),
            "bboxes": AttributeInfo(type=list, annotation=bbox_field(dtype=pl.Float64)),
        }
    )

    # Create sample data with bounding boxes
    df = pl.DataFrame(
        {
            "image_info": [
                {"width": 100, "height": 100},
            ],
            "bboxes": [
                [
                    # Box that spans tiles
                    [20, 20, 80, 80],
                    # Box entirely in top-right tile
                    [60, 10, 90, 40],
                    # Box outside any tile (at overlap)
                    [48, 48, 52, 52],
                ]
            ],
        }
    )

    # Configure tiling to create 50x50 tiles
    config = TilingConfig(tile_width=50, tile_height=50)

    # Apply tiling
    plan = _create_tiling_plan(schema, config, threshold_drop_ann=0.0)
    result_df, _ = _apply_tiling(df, None, plan, ["bboxes"])

    # There should be 4 tiles (2x2 grid)
    assert len(result_df) == 4

    # Check first tile (top-left, 0,0,50,50)
    assert len(result_df[0, "bboxes"]) == 2  # All original boxes except one
    assert (result_df[0, "bboxes"][0] == [20, 20, 50, 50]).all()
    assert (result_df[0, "bboxes"][1] == [48, 48, 50, 50]).all()

    # Check second tile (top-right, 50,0,100,50)
    assert len(result_df[1, "bboxes"]) == 3  # All original boxes
    assert (result_df[1, "bboxes"][0] == [0, 20, 30, 50]).all()  # Clipped and offset
    assert (result_df[1, "bboxes"][1] == [10, 10, 40, 40]).all()  # Offset from tile origin
    assert (result_df[1, "bboxes"][2] == [0, 48, 2, 50]).all()  # Clipped and offset


def test_polygon_and_label_tiling():
    # Create schema with both polygon and label fields
    schema = Schema(
        attributes={
            "image_info": AttributeInfo(type=dict, annotation=image_info_field()),
            "polygons": AttributeInfo(type=list, annotation=polygon_field(dtype=pl.Float64)),
            "labels": AttributeInfo(type=list, annotation=label_field(is_list=True)),
        }
    )

    # Create sample data with image info, polygons and labels
    df = pl.DataFrame(
        {
            "image_info": [
                {"width": 100, "height": 100},
            ],
            "polygons": [
                [
                    # Two polygons: one that will intersect tile, one that won't
                    [[25, 25], [75, 25], [75, 75], [25, 75], [25, 25]],  # Square in middle
                    [[80, 80], [90, 80], [90, 90], [80, 90], [80, 80]],  # Square in corner
                ]
            ],
            "labels": [["cat", "dog"]],  # One label per polygon
        }
    )

    # Configure tiling to create 50x50 tiles
    config = TilingConfig(tile_width=50, tile_height=50)

    # Apply tiling
    plan = _create_tiling_plan(schema, config, threshold_drop_ann=0.1)
    result_df, result_schema = _apply_tiling(df, None, plan, ["polygons", "labels"])

    # There should be 4 tiles (2x2 grid)
    assert len(result_df) == 4

    # Check first tile (top-left, should contain part of first polygon)
    assert len(result_df[0, "polygons"]) == 1  # One entry for the original polygon
    assert len(result_df[0, "labels"]) == 1  # One entry for the original label
    assert result_df[0, "labels"][0] == "cat"  # First label kept

    # Check last tile (bottom-right, should contain second polygon)
    assert len(result_df[3, "polygons"]) == 2
    assert len(result_df[3, "labels"]) == 2
    assert result_df[3, "labels"][0] == "cat"  # First label kept
    assert result_df[3, "labels"][1] == "dog"  # Second label kept
