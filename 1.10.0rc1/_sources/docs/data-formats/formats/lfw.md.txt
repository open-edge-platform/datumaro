# LFW

## Format specification

[LFW (Labeled Faces in the Wild Home)](http://vis-www.cs.umass.edu/lfw/)
it's dataset for face identification task,
specification for this format is available
[here](http://vis-www.cs.umass.edu/lfw/README.txt).
You can also download original LFW dataset
[here](http://vis-www.cs.umass.edu/lfw/#download).

Original dataset contains images with people faces.
For each image contains information about person's name, as well as
information about images that matched with this person
and mismatched with this person.
Also LFW contains additional information about landmark points on the face.

Supported annotation types:
- `Label`
- `Points` (face landmark points)

Supported attributes:
- `negative_pairs`: list with names of mismatched persons;
- `positive_pairs`: list with names of matched persons;

## Import LFW dataset

Importing LFW dataset into the Datumaro project:
```
datum project create
datum project import -f lfw <path_to_lfw_dataset>
```
See more information about adding datasets to the project in the
[docs](../../command-reference/context/sources.md#add-dataset).

Also you can import LFW dataset from Python API:
```python
import datumaro as dm

lfw_dataset = dm.Dataset.import_from('<path_to_lfw_dataset>', 'lfw')
```

For successful importing the LFW dataset, the directory with it
should has the following structure:

```
<path_to_lfw_dataset>/
├── subset_1
│    ├── annotations
│    │   ├── landmarks.txt # list with landmark points for each image
│    │   ├── pairs.txt # list of matched and mismatched pairs of person
│    │   └── people.txt # optional file with a list of persons name
│    └── images
│        ├── name0
│        │   ├── name0_0001.jpg
│        │   ├── name0_0002.jpg
│        │   ├── ...
│        ├── name1
│        │   ├── name1_0001.jpg
│        │   ├── name1_0002.jpg
│        │   ├── ...
├── subset_2
│    ├── ...
├── ...
```

Full description of annotation `*.txt` files available
[here](http://vis-www.cs.umass.edu/lfw/README.txt).

## Export LFW dataset

With Datumaro you can convert LFW dataset into any other
format [Datumaro supports](/docs/data-formats/formats/index.rst).
Pay attention that this format should also support `Label` and/or `Points`
annotation types.

There is few ways to convert LFW dataset into other format:

```
# Converting to ImageNet with `convert` command:
datum convert -if lfw -i ./lfw_dataset \
    -f imagenet -o ./output_dir -- --save-media

# Converting to VggFace2 through the Datumaro project:
datum project create
datum project add -f lfw ./lfw_dataset
datum project export -f vgg_face2 -o ./output_dir2
```

> Note: some formats have extra export options. For particular format see the
> [docs](/docs/data-formats/formats/index.rst) to get information about it.

## Export dataset to the LFW format

With Datumaro you can export dataset that has `Label` or/and `Points`
annotations to the LFW format, example:

```
# Converting VGG Face2 dataset into the LFW format
datum convert -if vgg_face2 -i ./voc_dataset \
    -f lfw -o ./output_dir


# Export dataaset to the LFW format through the Datumaro project:
datum project create
datum project import -f voc_classification ../vgg_dataset
datum project export -f lfw -o ./output_dir -- --save-media --image-ext png
```

Available extra export options for LFW dataset format:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--image-ext IMAGE_EXT` allow to specify image extension
  for exporting dataset (by default - keep original)
