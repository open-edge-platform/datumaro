from datumaro.experimental import LazyImage

# Datumaro Experimental Module

The `datumaro.experimental` module provides a modern, type-safe framework for working with machine learning datasets. It offers a declarative approach to defining data schemas, automatic type conversion between different representations, and efficient storage using Polars DataFrames.

## Table of Contents

- [Key Concepts](#key-concepts)
- [Defining Samples](#defining-samples)
  - [Basic Sample Definition](#basic-sample-definition)
  - [Available Field Types](#available-field-types)
  - [Semantic Tags](#semantic-tags)
  - [Schema Inference](#schema-inference)
- [Working with Datasets](#working-with-datasets)        import types
        from typing import Union, get_args, get_origin
  - [Creating Datasets](#creating-datasets)
  - [Adding Samples](#adding-samples)
  - [Accessing Samples](#accessing-samples)
  - [Modifying Datasets](#modifying-datasets)
  - [Iterating and Slicing](#iterating-and-slicing)
  - [Filtering by Subset](#filtering-by-subset)
  - [Merging Datasets](#merging-datasets)
- [Schema Conversion](#schema-conversion)
  - [Automatic Conversion](#automatic-conversion)
  - [How Converters Work](#how-converters-work)
  - [Registering Custom Converters](#registering-custom-converters)
  - [Built-in Converters](#built-in-converters)
- [Categories](#categories)
  - [Label Categories](#label-categories)
  - [Mask Categories](#mask-categories)
  - [Hierarchical Labels](#hierarchical-labels)
- [Import and Export](#import-and-export)
  - [Native Format (Parquet)](#native-format-parquet)
  - [Data Formats (COCO, etc.)](#data-formats-coco-etc)
- [Transforms](#transforms)
  - [Lazy Evaluation](#lazy-evaluation)
  - [Custom Transforms](#custom-transforms)
- [Tiling](#tiling)
- [Filtering](#filtering)
- [Media Handling](#media-handling)
  - [Lazy Image Loading](#lazy-image-loading)
  - [Image Caching](#image-caching)
- [Type Registry](#type-registry)
- [Best Practices](#best-practices)

---

## Key Concepts

The experimental module is built around these core concepts:

1. **Sample**: A dataclass-like definition of a single data point in your dataset
2. **Dataset**: A typed container holding multiple samples with a consistent schema
3. **Schema**: A formal description of the dataset structure derived from Sample definitions
4. **Field**: Type descriptors that define how data is stored and converted
5. **Converter**: Transformation logic between different field representations
6. **Transform**: Lazy operations applied during sample retrieval

---

## Defining Samples

### Basic Sample Definition

Samples are defined as classes that inherit from `Sample`. Each attribute is annotated with both a Python type and a Field descriptor:

```python
from typing import Any
import numpy as np
import polars as pl
from datumaro.experimental import Sample
from datumaro.experimental.fields import (
    ImageInfo,
    bbox_field,
    image_field,
    image_info_field,
    label_field,
)

class DetectionSample(Sample):
    """A sample for object detection tasks."""
    
    # Image data as a numpy array
    image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
    
    # Bounding boxes in x1y1x2y2 format
    bboxes: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), format="x1y1x2y2")
    
    # Class labels for each bounding box
    labels: np.ndarray[Any, Any] = label_field(dtype=pl.Int32(), is_list=True)
    
    # Image metadata
    image_info: ImageInfo = image_info_field()
```

You can also use the `Annotated` syntax for field definitions:

```python
from typing import Annotated

class DetectionSample(Sample):
    image: Annotated[np.ndarray, image_field(dtype=pl.UInt8(), format="RGB")]
    bboxes: Annotated[np.ndarray, bbox_field(dtype=pl.Float32())]
```

### Available Field Types

The module provides a rich set of field types for common ML data:

#### Image Fields

| Field | Description | Factory Function |
|-------|-------------|------------------|
| `ImageField` | In-memory image tensor | `image_field(dtype, format, channels_first)` |
| `ImagePathField` | Path to image file on disk | `image_path_field()` |
| `ImageCallableField` | Lazy-loaded image via callable | `image_callable_field()` |
| `ImageBytesField` | Raw image bytes | `image_bytes_field()` |
| `TensorField` | Generic n-dimensional tensor | `tensor_field(dtype, semantic)` |
| `ImageInfoField` | Image metadata (width, height) | `image_info_field(semantic)` |

#### Annotation Fields

| Field | Description | Factory Function |
|-------|-------------|------------------|
| `BBoxField` | Bounding boxes | `bbox_field(dtype, format, normalize)` |
| `RotatedBBoxField` | Rotated bounding boxes | `rotated_bbox_field(dtype, format)` |
| `PolygonField` | Polygon annotations | `polygon_field(dtype)` |
| `LabelField` | Class labels | `label_field(dtype, is_list)` |
| `KeypointsField` | Keypoint annotations | `keypoints_field(dtype)` |

#### Mask Fields

| Field | Description | Factory Function |
|-------|-------------|------------------|
| `MaskField` | Semantic segmentation mask | `mask_field(dtype)` |
| `MaskCallableField` | Lazy-loaded semantic mask | `mask_callable_field()` |
| `InstanceMaskField` | Instance segmentation mask | `instance_mask_field(dtype)` |
| `InstanceMaskCallableField` | Lazy-loaded instance mask | `instance_mask_callable_field()` |

#### Other Fields

| Field | Description | Factory Function |
|-------|-------------|------------------|
| `SubsetField` | Dataset subset (train/val/test) | `subset_field()` |
| `TileField` | Tile metadata for tiling operations | `tile_field()` |
| `NumericField` | Numeric values | `numeric_field(dtype, is_list)` |

### Semantic Tags

When you have multiple fields of the same type (e.g., stereo images), use semantic tags to distinguish them:

```python
class StereoSample(Sample):
    # Left camera image
    left_image: np.ndarray = image_field(
        dtype=pl.UInt8(), 
        format="RGB", 
        semantic="left"
    )
    
    # Right camera image
    right_image: np.ndarray = image_field(
        dtype=pl.UInt8(), 
        format="RGB", 
        semantic="right"
    )
    
    # Metadata for each camera
    left_info: ImageInfo = image_info_field(semantic="left")
    right_info: ImageInfo = image_info_field(semantic="right")
```

The schema enforces that only one field of each (type, semantic) combination can exist.

### Schema Inference

The schema is automatically inferred from the Sample class definition:

```python
# Get the schema from a Sample class
schema = DetectionSample.infer_schema()

# Schema contains AttributeInfo for each field
for name, attr_info in schema.attributes.items():
    print(f"{name}: type={attr_info.type}, field={attr_info.field}")
```

You can also create schemas dynamically:

```python
from datumaro.experimental.schema import Schema, AttributeInfo

schema = Schema(
    attributes={
        "image": AttributeInfo(
            type=np.ndarray,
            field=image_field(dtype=pl.UInt8(), format="RGB"),
        ),
        "bboxes": AttributeInfo(
            type=np.ndarray,
            field=bbox_field(dtype=pl.Float32()),
        ),
    }
)
```

---

## Working with Datasets

### Creating Datasets

Create a dataset from a Sample class:

```python
from datumaro.experimental import Dataset

# Create an empty dataset with the DetectionSample schema
dataset = Dataset(DetectionSample)

# Or create from a Schema object
dataset = Dataset(schema)
```

You can also provide initial categories:

```python
from datumaro.experimental.categories import LabelCategories

categories = {"labels": LabelCategories(labels=("cat", "dog", "bird"))}
dataset = Dataset(DetectionSample, categories=categories)
```

### Adding Samples

Add samples to the dataset using `append()`:

```python
sample = DetectionSample(
    image=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    bboxes=np.array([[10, 20, 100, 150], [50, 60, 200, 250]], dtype=np.float32),
    labels=np.array([0, 1], dtype=np.int32),
    image_info=ImageInfo(width=224, height=224),
)

dataset.append(sample)
```

### Accessing Samples

Retrieve samples by index:

```python
# Get a single sample
sample = dataset[0]

# Access sample attributes
print(sample.image.shape)
print(sample.bboxes)
print(sample.labels)
```

### Modifying Datasets

Update or delete samples:

```python
# Update a sample at index
dataset[0] = new_sample

# Delete a sample
del dataset[0]
```

> **Note**: Transformed datasets (after calling `convert_to_schema()` or `transform()`) are immutable.

### Iterating and Slicing

```python
# Get dataset length
print(f"Dataset has {len(dataset)} samples")

# Iterate over all samples
for sample in dataset:
    process(sample)

# Create a slice of the dataset
subset = dataset.slice(offset=10, length=50)
```

### Filtering by Subset

Datasets with a `SubsetField` can be filtered by subset:

```python
from datumaro.experimental.fields import Subset, subset_field

class TrainableSample(Sample):
    image: np.ndarray = image_field(dtype=pl.UInt8())
    subset: Subset = subset_field()

dataset = Dataset(TrainableSample)

# Add samples with different subsets
dataset.append(TrainableSample(image=..., subset=Subset.TRAINING))
dataset.append(TrainableSample(image=..., subset=Subset.VALIDATION))
dataset.append(TrainableSample(image=..., subset=Subset.TEST))

# Filter by subset
train_dataset = dataset.filter_by_subset(Subset.TRAINING)
val_dataset = dataset.filter_by_subset(Subset.VALIDATION)
```

### Merging Datasets

Append one dataset to another:

```python
dataset1 = Dataset(MySample)
dataset2 = Dataset(MySample)

# Add samples to both...

# Merge dataset2 into dataset1 (in-place)
dataset1.append_dataset(dataset2)
```

If the schemas differ, automatic conversion is attempted.

---

## Schema Conversion

### Automatic Conversion

One of the most powerful features is automatic schema conversion. The system uses an A* search algorithm to find the optimal conversion path:

```python
class SourceSample(Sample):
    image: np.ndarray = image_field(dtype=pl.UInt8(), format="RGB")
    bboxes: np.ndarray = bbox_field(dtype=pl.Float32(), format="x1y1x2y2")

class TargetSample(Sample):
    image: np.ndarray = image_field(dtype=pl.Float32(), format="BGR")
    bboxes: np.ndarray = bbox_field(dtype=pl.Float32(), format="xywh")

# Create source dataset
source_dataset = Dataset(SourceSample)
source_dataset.append(...)

# Convert to target schema automatically
target_dataset = source_dataset.convert_to_schema(TargetSample)

# The conversion handles:
# - RGB → BGR color conversion
# - UInt8 → Float32 dtype conversion
# - x1y1x2y2 → xywh bbox format conversion
```

You can also convert a single sample:

```python
from datumaro.experimental.dataset import convert_sample_to_schema

converted_sample = convert_sample_to_schema(
    sample=source_sample,
    source_schema=SourceSample.infer_schema(),
    target_dtype_or_schema=TargetSample,
)
```

### How Converters Work

Converters are classes that transform data between field representations:

1. **Input/Output Specifications**: Each converter declares what field types it consumes and produces
2. **Filter Method**: Determines if the converter applies to specific field configurations
3. **Convert Method**: Performs the actual data transformation on a Polars DataFrame

```python
from datumaro.experimental.converters import Converter, converter
from datumaro.experimental.schema import AttributeSpec

@converter  # Registers the converter automatically
class MyConverter(Converter):
    # Declare input and output field types
    input_image: AttributeSpec[ImageField]
    output_image: AttributeSpec[ImageField]
    
    def filter_output_spec(self) -> bool:
        """Return True if this converter can handle the input/output combination."""
        return (
            self.input_image.field.format == "RGB" and
            self.output_image.field.format == "BGR"
        )
    
    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """Perform the conversion."""
        # Swap R and B channels
        # ... implementation
        return df
```

### Registering Custom Converters

Use the `@converter` decorator to register your converter:

```python
from datumaro.experimental import converter, Converter

@converter
class CustomConverter(Converter):
    # ... implementation
```

Or register manually:

```python
from datumaro.experimental import ConverterRegistry

ConverterRegistry.add_converter(CustomConverter)
```

### Built-in Converters

The module includes many pre-registered converters:

| Converter | Description |
|-----------|-------------|
| `RedBlueColorConverter` | RGB ↔ BGR color format conversion |
| `UInt8ToFloat32Converter` | Image dtype conversion |
| `BBoxCoordinateConverter` | Bbox format conversion (xywh, x1y1x2y2, etc.) |
| `ImagePathToImageConverter` | Load image from path |
| `ImageCallableToImageConverter` | Execute lazy image loader |
| `ImageBytesToImageConverter` | Decode image from bytes |
| `PolygonToBBoxConverter` | Generate bboxes from polygons |
| `PolygonToMaskConverter` | Rasterize polygons to masks |
| `PolygonToInstanceMaskConverter` | Rasterize to instance masks |
| `RotatedBBoxToPolygonConverter` | Convert rotated bbox to polygon |
| `MaskCallableToMaskConverter` | Execute lazy mask loader |
| `LabelIndexConverter` | Convert label names to indices |

---

## Categories

Categories provide metadata about label spaces for classification, detection, and segmentation tasks.

### Label Categories

```python
from datumaro.experimental.categories import LabelCategories, GroupType

# Simple label list
labels = LabelCategories(labels=("cat", "dog", "bird"))

# With group type (exclusive = single label per item)
labels = LabelCategories(
    labels=("cat", "dog", "bird"),
    group_type=GroupType.EXCLUSIVE,
)

# Access labels
print(labels[0])  # "cat"
print(len(labels))  # 3
print("dog" in labels)  # True

# Find label index
idx, name = labels.find("dog")  # (1, "dog")
```

### Mask Categories

```python
from datumaro.experimental.categories import MaskCategories

# Define colors for each class
mask_categories = MaskCategories(
    colormap={
        0: (0, 0, 0),      # background - black
        1: (255, 0, 0),    # class 1 - red
        2: (0, 255, 0),    # class 2 - green
    }
)
```

### Hierarchical Labels

For datasets with parent-child label relationships:

```python
from datumaro.experimental.categories import HierarchicalLabelCategories

hierarchy = HierarchicalLabelCategories(
    labels=("animal", "cat", "dog", "vehicle", "car", "truck"),
    parent_map={
        "cat": "animal",
        "dog": "animal", 
        "car": "vehicle",
        "truck": "vehicle",
    }
)

# Get children of a parent
children = hierarchy.get_children("animal")  # ["cat", "dog"]
```

### Applying Categories to Datasets

```python
categories = {
    "labels": LabelCategories(labels=("cat", "dog", "bird")),
}

dataset = Dataset(DetectionSample, categories=categories)

# Or add to schema
schema = schema.with_categories(categories)
```

---

## Import and Export

### Native Format (Parquet)

Export datasets to a portable format with full fidelity:

```python
from datumaro.experimental import export_dataset, import_dataset

# Export to directory
export_dataset(dataset, "/path/to/output", export_images=True)

# Export as ZIP archive
export_dataset(dataset, "/path/to/output.zip", export_images=True, as_zip=True)

# Import dataset
loaded_dataset = import_dataset("/path/to/output")

# Import with specific Sample type (for type hints)
loaded_dataset = import_dataset("/path/to/output", dtype=DetectionSample)
```

The export format includes:
- `data.parquet`: DataFrame with all serializable data
- `metadata.json`: Schema, categories, and version info
- `images/`: Exported images (if `export_images=True`)

### Data Formats (COCO, etc.)

Load and save datasets in common formats:

```python
from datumaro.experimental.data_formats.base import DataFormat, load_dataset, save_dataset

# Load COCO dataset
dataset = load_dataset("/path/to/coco", DataFormat.COCO)

# Save as COCO format
save_dataset(dataset, "/path/to/output", DataFormat.COCO)
```

#### COCO Format

The COCO loader/saver handles the standard COCO 2017 directory structure:

```
coco_root/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   ├── person_keypoints_val2017.json
│   ├── captions_train2017.json
│   └── captions_val2017.json
├── train2017/
│   └── *.jpg
└── val2017/
    └── *.jpg
```

```python
from datumaro.experimental.data_formats.coco.io import load_coco_dataset, save_coco_dataset
from datumaro.experimental.data_formats.coco.sample import CocoSample

# Load COCO dataset
dataset = load_coco_dataset("/path/to/coco", version="2017")

# Work with CocoSample attributes
for sample in dataset:
    print(sample.image)       # Image path
    print(sample.bboxes)      # Bounding boxes
    print(sample.labels)      # Class labels
    print(sample.polygons)    # Polygon annotations
    print(sample.keypoints)   # Keypoint annotations
    print(sample.captions)    # Image captions
    print(sample.subset)      # train/val/test

# Save modified dataset
save_coco_dataset(dataset, "/path/to/output")
```

---

## Transforms

Transforms enable lazy, chainable operations on datasets without materializing intermediate results.

### Lazy Evaluation

Many operations (like schema conversion) create transform chains that are only evaluated when samples are accessed:

```python
# This creates a transform chain, not immediate conversion
converted = dataset.convert_to_schema(TargetSample)

# The conversion happens here, when accessing data
sample = converted[0]  # Lazy evaluation triggered
```

### Custom Transforms

Create custom transforms by implementing the `Transform` class:

```python
from datumaro.experimental.transform import Transform

class MyTransform(Transform):
    def __init__(self, parent: Transform, schema: Schema):
        super().__init__(schema)
        self._parent = parent
    
    def apply(self, fields: Sequence[str]) -> pl.DataFrame:
        """Apply transformation to the specified fields."""
        df = self._parent.apply(fields)
        # ... modify df
        return df
    
    def get_lazy_attributes(self) -> set[str]:
        """Return attributes that should be evaluated lazily."""
        return self._parent.get_lazy_attributes()
    
    def slice(self, offset: int, length: int | None = None) -> Transform:
        """Create a slice of this transform."""
        # ... implementation
    
    def __len__(self) -> int:
        return len(self._parent)
```

Use transforms via the dataset's `transform()` method:

```python
new_dataset = dataset.transform(
    lambda parent: MyTransform(parent, schema),
    dtype=TargetSample,
)
```

---

## Tiling

The tiling system allows splitting large images into smaller tiles for processing:

```python
from datumaro.experimental.tiling.tiler_registry import TilerRegistry
from datumaro.experimental.fields import TileInfo, tile_field

class TiledSample(Sample):
    image: np.ndarray = image_field(dtype=pl.UInt8())
    tile: TileInfo = tile_field()
```

Tilers are registered per field type and handle how each field should be tiled:
- `ImageTiler`: Extracts image regions
- `BBoxTiler`: Clips/filters bounding boxes to tile boundaries
- `MaskTiler`: Extracts mask regions
- `PolygonTiler`: Clips polygons to tile boundaries

---

## Filtering

The filtering system removes samples or annotations based on criteria:

```python
from datumaro.experimental.filtering import FilterRegistry, Filter

# Filters are registered per field type
@FilterRegistry.register(BBoxField)
class EmptyBBoxFilter(Filter):
    def filter(self, df: pl.DataFrame) -> pl.Expr:
        """Return boolean expression for rows to keep."""
        return pl.col(self.field_spec.name).list.len() > 0
```

---

## Media Handling

### Lazy Image Loading

For datasets with images on disk, use lazy loading to avoid memory issues:

```python
from datumaro.experimental.media import LazyImage

class LazySample(Sample):
    image: LazyImage = image_callable_field()

# The image is loaded only when accessed
sample = LazySample(image="path/to/image.jpg")
image_array = sample.image.data  # Loads from disk here
```

### Image Caching

The module includes a global LRU cache for loaded images with byte-size limiting (default 256 MB). The cache is managed through the `ImageCache` class:

```python
from datumaro.experimental import ImageCache

# Set cache to 512 MB
ImageCache.set_size(512 * 1024 * 1024)

# Check current cache usage
print(f"Cached images: {ImageCache.length()}")
print(f"Current size: {ImageCache.get_size()} bytes")
print(f"Max size: {ImageCache.get_max_size()} bytes")

# Get detailed cache info (count, current bytes, max bytes)
info = ImageCache.info()
print(f"Cache: {info['count']} images, {info['current_size'] / 1024 / 1024:.1f} MB used of {info['max_size'] / 1024 / 1024:.1f} MB")

# Clear all cached images
ImageCache.clear()
```

---

## Type Registry

Register custom type converters for Polars serialization:

```python
from datumaro.experimental import register_numpy_converter, register_from_polars_converter

# Register how to convert a custom type to numpy for Polars storage
@register_numpy_converter(MyCustomType)
def custom_to_numpy(value, dtype):
    return np.array(value.data)

# Register how to convert from Polars back to your type
@register_from_polars_converter(MyCustomType)
def numpy_to_custom(data, target_type):
    return MyCustomType(data)
```

---

## Best Practices

### 1. Define Sample Classes for Your Task

Create specific Sample classes that match your task requirements:

```python
class SegmentationSample(Sample):
    image: np.ndarray = image_field(dtype=pl.UInt8(), format="RGB")
    mask: np.ndarray = mask_field(dtype=pl.UInt8())
    image_info: ImageInfo = image_info_field()
```

### 2. Use Lazy Loading for Large Datasets

For datasets with many large images, use path or callable fields:

```python
class EfficientSample(Sample):
    image: LazyImage = image_path_field()  # Store path, load on demand
    image_info: ImageInfo = image_info_field()
```

### 3. Leverage Automatic Conversion

Don't manually convert between formats - let the converter system handle it:

```python
# Instead of manual conversion
# for sample in source_dataset:
#     converted = manually_convert(sample)
#     target_dataset.append(converted)

# Use automatic conversion
target_dataset = source_dataset.convert_to_schema(TargetSample)
```

### 4. Use Categories for Label Management

Always define categories for datasets with labels, these will be used for validation when adding samples to the dataset.

```python
categories = {"labels": LabelCategories(labels=get_class_names())}
dataset = Dataset(MySample, categories=categories)
```

### 5. Validate Samples

The Sample class automatically validates types. For custom validation:

```python
class ValidatedSample(Sample):
    def __post_init__(self):
        super().__post_init__()
        if self.bboxes is not None and len(self.bboxes) != len(self.labels):
            raise ValueError("Number of bboxes must match number of labels")
```

### 6. Manage Memory with Cache Settings

For large datasets, configure the image cache appropriately:

```python
from datumaro.experimental import ImageCache

# For memory-constrained environments (e.g., 64 MB)
ImageCache.set_size(64 * 1024 * 1024)

# For fast iteration over the same images (e.g., 1 GB)
ImageCache.set_size(1024 * 1024 * 1024)

# Disable caching entirely
ImageCache.set_size(0)
```

---

## Examples

### Complete Detection Workflow

```python
import numpy as np
import polars as pl
from datumaro.experimental import Dataset, Sample, export_dataset
from datumaro.experimental.fields import (
    ImageInfo,
    Subset,
    bbox_field,
    image_field,
    image_info_field,
    label_field,
    subset_field,
)
from datumaro.experimental.categories import LabelCategories

# Define sample schema
class DetectionSample(Sample):
    image: np.ndarray = image_field(dtype=pl.UInt8(), format="RGB")
    bboxes: np.ndarray = bbox_field(dtype=pl.Float32(), format="x1y1x2y2")
    labels: np.ndarray = label_field(dtype=pl.Int32(), is_list=True)
    image_info: ImageInfo = image_info_field()
    subset: Subset = subset_field()

# Create dataset with categories
categories = {"labels": LabelCategories(labels=("person", "car", "bicycle"))}
dataset = Dataset(DetectionSample, categories=categories)

# Add training samples
for img_path in training_images:
    img = load_image(img_path)
    boxes, labels = detect_objects(img)
    
    dataset.append(DetectionSample(
        image=img,
        bboxes=boxes,
        labels=labels,
        image_info=ImageInfo(width=img.shape[1], height=img.shape[0]),
        subset=Subset.TRAINING,
    ))

# Export dataset
export_dataset(dataset, "my_dataset.zip", as_zip=True)
```

### Converting Between Formats

```python
from datumaro.experimental.data_formats.base import DataFormat, load_dataset, save_dataset

# Load COCO dataset
coco_dataset = load_dataset("/data/coco", DataFormat.COCO)

# Convert to your custom schema
class MyDetectionSample(Sample):
    image: np.ndarray = image_field(dtype=pl.Float32(), format="BGR")
    bboxes: np.ndarray = bbox_field(dtype=pl.Float32(), format="xywh", normalize=True)
    labels: np.ndarray = label_field(dtype=pl.Int32(), is_list=True)

my_dataset = coco_dataset.convert_to_schema(MyDetectionSample)

# Process samples
for sample in my_dataset:
    # Image is now Float32 BGR, bboxes are normalized xywh
    model_input = preprocess(sample.image)
    predictions = model(model_input)
```

---

## API Reference

For detailed API documentation, see the docstrings in each module:

- `datumaro.experimental.dataset` - Dataset and Sample classes
- `datumaro.experimental.schema` - Schema and AttributeInfo
- `datumaro.experimental.fields` - Field type definitions
- `datumaro.experimental.converters` - Converter system
- `datumaro.experimental.categories` - Category definitions
- `datumaro.experimental.export_import` - Import/export functions
- `datumaro.experimental.transform` - Transform base classes
- `datumaro.experimental.media` - Media handling utilities
