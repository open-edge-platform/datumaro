# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
import polars as pl

from datumaro.v2.categories import LabelCategories
from datumaro.v2.converters.base import Converter, list_eval_ref
from datumaro.v2.converters.registry import converter
from datumaro.v2.fields.annotations import BBoxField, LabelField, PolygonField, RotatedBBoxField
from datumaro.v2.fields.images import ImageField
from datumaro.v2.schema import AttributeSpec


@converter
class LabelIndexConverter(Converter):
    """
    Converter for updating label indices when label categories change.

    This converter handles remapping label values in LabelField when the order
    of labels has changed between input and output categories. It only applies
    when both input and output categories are LabelCategories and have different
    label orders but the same set of labels.
    """

    input_labels: AttributeSpec[LabelField]
    output_labels: AttributeSpec[LabelField]

    def filter_output_spec(self) -> bool:
        """
        Check if this converter is applicable based on input/output categories.

        Returns True only if:
        1. Both input and output have LabelCategories
        2. The categories have the same set of labels but different order
        3. The field types are the same (no field conversion needed)
        """
        # Check that both specs have categories
        if self.input_labels.categories is None or self.output_labels.categories is None:
            return False

        # Check that both are LabelCategories
        if not isinstance(self.input_labels.categories, LabelCategories) or not isinstance(
            self.output_labels.categories, LabelCategories
        ):
            return False

        input_cats = self.input_labels.categories
        output_cats = self.output_labels.categories

        # Check that the sets of labels are the same but order might differ
        input_labels_set = set(input_cats.labels)
        output_labels_set = set(output_cats.labels)

        if input_labels_set != output_labels_set:
            return False

        # Only apply if the order actually differs
        if input_cats.labels == output_cats.labels:
            return False

        index_mapping = {}
        for old_idx, label in enumerate(input_cats.labels):
            semantic = None
            for sem, lbl in input_cats.label_semantics.items():
                if lbl == label:
                    semantic = sem
                    break
            if semantic is not None:
                new_label = output_cats.label_semantics[semantic]
                new_idx, _ = output_cats.find(new_label)
            else:
                new_idx, _ = output_cats.find(label)
            if new_idx is not None:
                index_mapping[old_idx] = new_idx
        self._index_mapping = index_mapping
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert label indices based on category mapping.

        If label semantics are defined, use them to match input and output labels for mapping.
        Otherwise, fall back to label name.
        """
        # Use the precomputed index mapping from filter_output_spec
        index_mapping = getattr(self, "_index_mapping", None)
        if index_mapping is None:
            raise RuntimeError("index_mapping not computed. Call filter_output_spec first.")

        input_col = self.input_labels.name
        output_col = self.output_labels.name
        field = self.input_labels.field

        if field.multi_label or field.is_list:
            mapping_expr = pl.col(input_col).list.eval(
                pl.element().replace(list(index_mapping.keys()), list(index_mapping.values()), default=None)
            )
        else:
            mapping_expr = pl.col(input_col).replace(
                list(index_mapping.keys()), list(index_mapping.values()), default=None
            )

        return df.with_columns(mapping_expr.alias(output_col))


@converter
class BBoxCoordinateConverter(Converter):
    """
    Convert bounding box coordinates between normalized and absolute formats.

    This converter handles transformations between normalized coordinates
    (range [0,1]) and absolute pixel coordinates using image dimensions.
    """

    input_bbox: AttributeSpec[BBoxField]
    input_image: AttributeSpec[ImageField]
    output_bbox: AttributeSpec[BBoxField]

    def filter_output_spec(self) -> bool:
        """
        Check if bbox normalization conversion is needed and configure output.

        Returns:
            True if conversion is needed (normalization status differs), False otherwise
        """
        input_normalized = self.input_bbox.field.normalize
        output_normalized = self.output_bbox.field.normalize

        # Determine the target normalization from output specification
        target_normalized = output_normalized

        # Configure output specification with correct normalization
        self.output_bbox = AttributeSpec(
            name=self.output_bbox.name,
            field=BBoxField(
                semantic=self.input_bbox.field.semantic,
                dtype=self.input_bbox.field.dtype,
                format=self.input_bbox.field.format,
                normalize=target_normalized,
            ),
        )

        # Apply converter only if normalization status needs to change
        return input_normalized != target_normalized

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert bbox coordinates between normalized and absolute formats.

        Uses image dimensions to transform coordinates. For normalized to absolute:
        multiplies by image dimensions. For absolute to normalized: divides by
        image dimensions.

        Args:
            df: Input DataFrame containing bbox and image data

        Returns:
            DataFrame with converted bounding box coordinates
        """
        input_bbox_name = self.input_bbox.name
        input_image_name = self.input_image.name
        output_bbox_name = self.output_bbox.name

        # Get the image shape column name
        image_shape_name = f"{input_image_name}_shape"

        input_normalized = self.input_bbox.field.normalize

        # Create temporary column names for dimensions
        temp_width_col = f"{input_image_name}_width"
        temp_height_col = f"{input_image_name}_height"

        # Coordinate order for width/height mapping: [height, width, height, width]
        coordinates_order = [1, 0, 1, 0]

        def op(x: pl.Expr, y: pl.Expr) -> pl.Expr:
            """Choose operation based on conversion direction."""
            # FIXME: x.cast(pl.Float64) is a workaround for Polars bug
            # https://github.com/pola-rs/polars/issues/23924
            xy = x * y if input_normalized else x.cast(pl.Float64) / y
            return xy.cast(self.output_bbox.field.dtype)

        # Extract width and height from image shape
        df_with_temp = df.with_columns(
            [
                pl.col(image_shape_name).list.get(1).alias(temp_width_col),  # width
                pl.col(image_shape_name).list.get(0).alias(temp_height_col),  # height
            ]
        )

        # Apply coordinate transformation
        result_df = df_with_temp.with_columns(
            list_eval_ref(
                input_bbox_name,
                image_shape_name,
                lambda element, ref: pl.concat_arr(
                    op(element.arr.get(0), ref.list.get(coordinates_order[0])),  # x1
                    op(element.arr.get(1), ref.list.get(coordinates_order[1])),  # y1
                    op(element.arr.get(2), ref.list.get(coordinates_order[2])),  # x2
                    op(element.arr.get(3), ref.list.get(coordinates_order[3])),  # y2
                ),
            ).alias(output_bbox_name)
        )

        # Clean up temporary columns
        return result_df.drop([temp_width_col, temp_height_col])


@converter
class BBoxDtypeConverter(Converter):
    """
    Convert BBox field between different data types.
    """

    input_bbox: AttributeSpec[BBoxField]
    output_bbox: AttributeSpec[BBoxField]

    def filter_output_spec(self) -> bool:
        """Configure output bbox specification with target dtype."""
        self.output_bbox = AttributeSpec(
            name=self.output_bbox.name,
            field=BBoxField(
                semantic=self.input_bbox.field.semantic,
                dtype=self.output_bbox.field.dtype,  # Use target dtype
                format=self.input_bbox.field.format,
                normalize=self.input_bbox.field.normalize,
            ),
        )

        # Apply converter only if dtypes are different
        return self.input_bbox.field.dtype != self.output_bbox.field.dtype

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert bbox data to target dtype."""
        input_col = self.input_bbox.name
        output_col = self.output_bbox.name

        # Cast bbox values to target dtype by converting each element
        return df.with_columns(
            pl.col(input_col)
            .list.eval(pl.element().arr.to_list().list.eval(pl.element().cast(self.output_bbox.field.dtype)))
            .alias(output_col)
        )


@converter
class LabelDtypeConverter(Converter):
    """
    Convert Label field between different data types.
    """

    input_label: AttributeSpec[LabelField]
    output_label: AttributeSpec[LabelField]

    def filter_output_spec(self) -> bool:
        """Configure output label specification with target dtype."""
        self.output_label = AttributeSpec(
            name=self.output_label.name,
            field=LabelField(
                semantic=self.input_label.field.semantic,
                dtype=self.output_label.field.dtype,  # Use target dtype
                multi_label=self.input_label.field.multi_label,
                is_list=self.input_label.field.is_list,
            ),
        )

        # Apply converter only if dtypes are different
        return self.input_label.field.dtype != self.output_label.field.dtype

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert label data to target dtype."""
        input_col = self.input_label.name
        output_col = self.output_label.name

        if self.input_label.field.is_list:
            # Handle list of labels
            return df.with_columns(
                pl.col(input_col).list.eval(pl.element().cast(self.output_label.field.dtype)).alias(output_col)
            )
        # Handle single label
        return df.with_columns(pl.col(input_col).cast(self.output_label.field.dtype).alias(output_col))


@converter
class PolygonToBBoxConverter(Converter):
    """
    Converts polygon annotations to bounding boxes.

    Extracts the bounding box coordinates that enclose each polygon.
    """

    input_polygon: AttributeSpec[PolygonField]
    output_bbox: AttributeSpec[BBoxField]

    def filter_output_spec(self) -> bool:
        """Configure output specification for bounding box format."""
        # Configure output for bbox format
        self.output_bbox = AttributeSpec(
            name=self.output_bbox.name,
            field=BBoxField(
                semantic=self.input_polygon.field.semantic,
                dtype=self.input_polygon.field.dtype,
                format=self.output_bbox.field.format,
                normalize=self.input_polygon.field.normalize,  # Inherit normalization from polygon
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract bounding boxes from polygon coordinates.

        Args:
            df: DataFrame with polygon coordinates

        Returns:
            DataFrame with bounding box data in output column
        """
        input_column_name = self.input_polygon.name
        output_column_name = self.output_bbox.name

        df = df.with_columns(
            pl.col(input_column_name)
            .list.eval(
                pl.concat_arr(
                    [
                        pl.element().list.eval(pl.element().arr.get(0)).list.min(),
                        pl.element().list.eval(pl.element().arr.get(1)).list.min(),
                        pl.element().list.eval(pl.element().arr.get(0)).list.max(),
                        pl.element().list.eval(pl.element().arr.get(1)).list.max(),
                    ]
                )
            )
            .alias(output_column_name)
        )

        # Format according to output bbox format
        if self.output_bbox.field.format == "x1y1x2y2":
            # Already in this format
            pass
        elif self.output_bbox.field.format == "xywh":
            df = df.with_columns(
                pl.col(output_column_name).list.eval(
                    pl.concat_arr(
                        [
                            pl.element().arr.get(0),
                            pl.element().arr.get(1),
                            pl.element().arr.get(2) - pl.element().arr.get(0),
                            pl.element().arr.get(3) - pl.element().arr.get(1),
                        ]
                    )
                )
            )
        else:
            raise NotImplementedError(
                f"This conversion is not yet implemented for the format {self.output_bbox.field.format}."
            )

        return df


@converter
class RotatedBBoxToPolygonConverter(Converter):
    """
    Converts rotated bounding boxes to polygon coordinates.

    Transforms rotated bounding box parameters (cx, cy, w, h, r) into
    polygon corner points by rotating the rectangle corners around the center.
    """

    input_rotated_bbox: AttributeSpec[RotatedBBoxField]
    output_polygon: AttributeSpec[PolygonField]

    def filter_output_spec(self) -> bool:
        """Configure output specification for polygon format."""
        # Configure output for polygon format
        self.output_polygon = AttributeSpec(
            name=self.output_polygon.name,
            field=PolygonField(
                semantic=self.input_rotated_bbox.field.semantic,
                dtype=self.input_rotated_bbox.field.dtype,
                format=self.output_polygon.field.format,
                normalize=self.input_rotated_bbox.field.normalize,  # Inherit normalization
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert rotated bounding boxes to polygon corner points.

        Args:
            df: DataFrame with rotated bounding box coordinates

        Returns:
            DataFrame with polygon data in output column
        """
        input_column_name = self.input_rotated_bbox.name
        output_column_name = self.output_polygon.name

        cx = pl.element().arr.get(0)
        cy = pl.element().arr.get(1)
        w = pl.element().arr.get(2)
        h = pl.element().arr.get(3)
        r = pl.element().arr.get(4)

        def rotate_corner(expr: pl.Expr):
            px = expr.arr.get(0)
            py = expr.arr.get(1)
            cos_theta = r.cos()
            sin_theta = r.sin()
            return pl.concat_arr(cos_theta * px - sin_theta * py + cx, sin_theta * px + cos_theta * py + cy)

        return df.with_columns(
            pl.col(input_column_name)
            .list.eval(
                pl.concat_list(
                    rotate_corner(pl.concat_arr(-w / 2, -h / 2)),
                    rotate_corner(pl.concat_arr(w / 2, -h / 2)),
                    rotate_corner(pl.concat_arr(w / 2, h / 2)),
                    rotate_corner(pl.concat_arr(-w / 2, h / 2)),
                )
            )
            .alias(output_column_name)
        )
