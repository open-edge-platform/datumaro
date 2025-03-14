# Explain

## Run model inference explanation (explain)

Runs an explainable AI algorithm for a model.

This tool is supposed to help an AI developer to debug a model and a dataset.
Basically, it executes model inference and tries to find relation between
inputs and outputs of the trained model, i.e. determine decision boundaries
and belief intervals for the classifier.

Currently, the only available algorithm is RISE ([article](https://arxiv.org/pdf/1806.07421.pdf)),
which runs a model a single time and then re-runs a model multiple times on
each image to produce a heatmap of activations for each output of the
first inference. Each time a part of the input image is masked. As a result,
we obtain a number heatmaps, which show how specific image pixels affected
the inference result. This algorithm doesn't require any special information
about the model, but it requires the model to return all the outputs and
confidences. The original algorithm supports only classification scenario,
but Datumaro extends it for detection models.

The following use cases available:
- RISE for classification
- RISE for object detection

Usage:

```console
datum explain [-h] -m MODEL [-o SAVE_DIR] [-p PROJECT_DIR]
              [target] {rise} [RISE_ARGS]
```

Parameters:
- `<target>` (string) - Target
  [dataset revpath](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).By default,
  uses the whole current project. An image path can be specified instead.
  \<image path\> - a path to the file.
  \<revpath\> - [a dataset path or a revision path](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).
- `<method>` (string) - The algorithm to use. Currently, only `rise`
  is supported.
- `-m, --model` (string) - The model to use for inference
- `-o, --output-dir` (string) - Directory to save results to
  (default: display only)
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

- RISE options:
  - `-s, --max-samples` (number) - Number of algorithm model runs per image
    (default: mask size ^ 2).
  - `--mw, --mask-width` (number) - Mask width in pixels (default: 7)
  - `--mh, --mask-height` (number) - Mask height in pixels (default: 7)
  - `--prob` (number) - Mask pixel inclusion probability, controls
    mask density (default: 0.5)
  - `--iou, --iou-thresh` (number) - IoU match threshold for detections
    (default: 0.9)
  - `--nms, --nms-iou-thresh` (number) - IoU match threshold for detections
    for non-maxima suppression (default: no NMS)
  - `--conf, --det-conf-thresh` (number) - Confidence threshold for
    detections (default: include all)
  - `-b, --batch-size` (number) - Batch size for inference (default: 1)
  - `--display` - Visualize results during computations

Examples:
- Run RISE on an image, display results
  ```console
  datum explain <path/to/image> -m mymodel rise --max-samples 50
  ```

- Run RISE on a source revision
  ```console
  datum explain HEAD~1:source-1 -m model rise
  ```

- Run inference explanation on a single image with online visualization
  ```console
  datum project create <...>
  datum model add mymodel <...>
  datum explain -t image.png -m mymodel \
      rise --max-samples 1000 --display
  ```

> **Note**: this algorithm requires the model to return
> _all_ (or a _reasonable_ amount) the outputs and confidences unfiltered,
> i.e. all the `Label` annotations for classification models and
> all the `Bbox`es for detection models.
> You can find examples of the expected model outputs in [`tests/test_RISE.py`](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/unit/algorithms/test_RISE.py)

For OpenVINO models the output processing script would look like this:

Classification scenario:

``` python
import datumaro as dm
from datumaro.util.annotation_util import softmax

def process_outputs(inputs, outputs):
    # inputs = model input, array or images, shape = (N, C, H, W)
    # outputs = model output, logits, shape = (N, n_classes)
    # results = conversion result, [ [ Annotation, ... ], ... ]
    results = []
    for output in outputs:
        confs = softmax(output[0])
        for label, conf in enumerate(confs):
            results.append(dm.Label(int(label)), attributes={'score': float(conf)})

    return results
```

Object Detection scenario:

``` python
import datumaro as dm

# return a significant number of output boxes to make multiple runs
# statistically correct and meaningful
max_det = 1000

def process_outputs(inputs, outputs):
    # inputs = model input, array or images, shape = (N, C, H, W)
    # outputs = model output, shape = (N, 1, K, 7)
    # results = conversion result, [ [ Annotation, ... ], ... ]
    results = []
    for input, output in zip(inputs, outputs):
        input_height, input_width = input.shape[:2]
        detections = output[0]
        image_results = []
        for det in detections:
            label = int(det[1])
            conf = float(det[2])
            x = max(int(det[3] * input_width), 0)
            y = max(int(det[4] * input_height), 0)
            w = min(int(det[5] * input_width - x), input_width)
            h = min(int(det[6] * input_height - y), input_height)
            image_results.append(dm.Bbox(x, y, w, h,
                label=label, attributes={'score': conf} ))

            results.append(image_results[:max_det])

    return results
```
