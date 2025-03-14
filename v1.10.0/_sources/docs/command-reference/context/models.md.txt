# Model (context)

## Models

### Register model

Datumaro can execute deep learning models in various frameworks. Check
[the plugins section](/docs/user-manual/extending.md#built-in-plugins)
for more info.

Supported frameworks:
- OpenVINO
- Custom models via custom `launchers`

Models need to be added to the Datumaro project first. It can be done with
the `datum model add` command.

Usage:

``` bash
datum model add [-h] [-n NAME] -l LAUNCHER [--copy] [--no-check]
  [-p PROJECT_DIR] [-- EXTRA_ARGS]
```

Parameters:
- `-l, --launcher` (string) - Model launcher name
- `--copy` - Copy model data into project. By default, only the link is saved.
- `--no-check` - Don't check the model can be loaded
- `-n`, `--name` (string) - Name of the new model (default: generate
  automatically)
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.
- `<extra args>` - Additional arguments for the model launcher
  (use `-- -h` for help). Must be specified after the main command arguments.

Example: register an OpenVINO model

A model consists of a graph description and weights. There is also a script
used to convert model outputs to internal data structures.

``` bash
datum create
datum model add \
  -n <model_name> -l openvino -- \
  -d <path_to_xml> -w <path_to_bin> -i <path_to_interpretation_script>
```

Interpretation script for an OpenVINO detection model (`convert.py`):
You can find OpenVINO model interpreter samples in
`datumaro/plugins/openvino/samples` ([instruction](/docs/explanation/plugins/openvino_plugin)).

``` python
import datumaro as dm

max_det = 10
conf_thresh = 0.1

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
            if conf <= conf_thresh:
                continue

            x = max(int(det[3] * input_width), 0)
            y = max(int(det[4] * input_height), 0)
            w = min(int(det[5] * input_width - x), input_width)
            h = min(int(det[6] * input_height - y), input_height)
            image_results.append(dm.Bbox(x, y, w, h,
                label=label, attributes={'score': conf} ))

            results.append(image_results[:max_det])

    return results

def get_categories():
    # Optionally, provide output categories - label map etc.
    # Example:
    label_categories = dm.LabelCategories()
    label_categories.add('person')
    label_categories.add('car')
    return { dm.AnnotationType.label: label_categories }
```

### Remove Models

To remove a model from a project, use the `datum model remove` command.

Usage:

``` bash
datum model remove [-h] [-p PROJECT_DIR] name
```

Parameters:
- `<name>` (string) - The name of the model to be removed
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example:

``` bash
datum create
datum model add <...> -n model1
datum remove model1
```

### Run Model

This command applies model to dataset images and produces a new dataset.

Usage:

``` bash
datum model run
```

Parameters:
- `<target>` (string) - A project build target to be used.
  By default, uses the combined `project` target.
- `-m, --model` (string) - Model name
- `-o, --output-dir` (string) - Output directory. By default, results will
  be stored in an auto-generated directory in the current directory.
- `--overwrite` - Allows to overwrite existing files in the output directory,
  when it is specified and is not empty.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Example: launch inference on a dataset

``` bash
datum create
datum import <...>
datum model add mymodel <...>
datum model run -m mymodel -o inference
```
