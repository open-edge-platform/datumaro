# LabelMe
## Format specification
[LabelMe](http://labelme.csail.mit.edu/Release3.0/) is an open-source annotation tool provided by MIT, which is commonly used for annotating images and creating ground truth for various computer vision tasks such as object detection and segmentation.
It allows users to draw bounding boxes, polygons, and scribbles on images to label objects and regions of interest.
You can install LabelMe on your local as following [Github instructions](https://github.com/wkentaro/labelme).

Supported annotation types:
- `Bbox`
- `Polygon`
- `Mask`

## Import a LabelMe dataset
A Datumaro project with a LabelMe source can be created in the following way:

``` bash
datum project create
datum project import --format label_me <path/to/dataset>
```

## Export a dataset with LabelMe format
Datumaro helps to export a dataset with LabelMe format through below:

```bash
datum project create
datum project add -f <any-other-dataset-format> <path/to/dataset/>
datum project export -f label_me -o <output/dir> -- --save-media
```
or
```bash
datum convert -if <any-other-dataset-format> -i <path/to/dataset> \
              -f label_me -o <output/dir> -- --save-media
```

Or, using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', '<any-other-dataset-format>')
dataset.export('save_dir', 'label_me', save_media=True)
```

> This can help you to import any data into LabelMe annotation tool for modifying or adding more annotations to the dataset.

## Directory structure
<!--lint disable fenced-code-flag-->
```
└─ labelme/
   ├── Images                           # Image directory
   │    ├── subset1                     # Subset directory
   │    │    ├── img1.jpg               # Image file
   │    │    ├── img2.jpg               # Image file
   │    │    └── ...
   │    ├── subset2                     # Subset directory
   │    │    ├── img1.jpg               # Image file
   │    │    └── ...
   │    └── ...
   ├── Annotations                      # Label directory
   │    ├── subset1                     # Subset directory
   │    │    ├── img1.xml               # Annotation file
   │    │    ├── img2.xml               # Annotation file
   │    │    └── ...
   │    ├── subset2                     # Subset directory
   │    │    ├── img1.xml               # Annotation file
   │    │    └── ...
   │    └── ...
   ├── Masks                            # Mask directory
   │    ├── subset1                     # Subset directory
   │    │    ├── img1_mask_0.png        # Mask file
   │    │    ├── img1_mask_1.png        # Mask file
   │    │    ├── img2_mask_0.png        # Mask file
   │    │    └── ...
   │    ├── subset2                     # Subset directory
   │    │    ├── img1_mask_0.png        # Mask file
   │    │    └── ...
   │    └── ...
   └──Scribbles                         # Scribble directory
        ├── subset1                     # Subset directory
        │    ├── img1_scribble_0.png    # Scribble file
        │    ├── img1_scribble_1.png    # Scribble file
        │    ├── img2_scribble_0.png    # Scribble file
        │    └── ...
        ├── subset2                     # Subset directory
        │    ├── img1_scribble_0.png    # Scribble file
        │    └── ...
        └── ...
```

## Annotation XML example
<!--lint disable fenced-code-flag-->
```
<annotation>
    <filename>example_image.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>cat</name>
        <polygon>
            <pt>
                <x>100</x>
                <y>150</y>
            </pt>
            <!-- Additional points defining the polygon -->
        </polygon>
    </object>
    <!-- Additional objects and annotations -->
</annotation>
```
