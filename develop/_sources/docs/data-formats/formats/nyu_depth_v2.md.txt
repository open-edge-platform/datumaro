# NYU Depth Dataset V2

## Format specification

The original NYU Depth Dataset V2 is available
[here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

Supported annotation types:
- `DepthAnnotation`

## Convert NYU Depth Dataset V2

The NYU Depth Dataset V2 is available for free [download](http://datasets.lids.mit.edu/nyudepthv2/).

A NYU Depth Dataset V2 can be converted in the following way:

```bash
datum convert --input-format nyu_depth_v2 --input-path <path/to/dataset> \
    --output-format <desired_format> --output-dir <output/dir>
```

It is also possible to convert the dataset using Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path/to/dataset>', 'nyu_depth_v2')
```

NYU Depth Dataset V2 directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
Dataset/
    ├── 1.h5
    ├── 2.h5
    ├── 3.h5
    └── ...
```

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/open-edge-platform/datumaro/blob/develop/tests/unit/test_nyu_depth_v2_format.py)
