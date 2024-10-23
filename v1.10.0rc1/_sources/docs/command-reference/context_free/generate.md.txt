# Generate

## Generate Datasets

Creates a synthetic dataset with elements of the specified type and shape,
and saves it in the provided directory.

Currently, can only generate fractal images, useful for network compression.
To create 3-channel images, you should provide the number of images, height and width.
The images are colorized with a model, which will be downloaded automatically.
Uses the algorithm from the [article](https://arxiv.org/abs/2103.13023).

Usage:

```console
datum generate [-h] -o OUTPUT_DIR -k COUNT --shape SHAPE [SHAPE ...]
               [-t {image}] [--overwrite] [--model-dir MODEL_PATH]
```

Parameters:
- `-o, --output-dir` (string) - Output directory
- `-k, --count` (integer) - Number of images to be generated
- `--shape` (integer, repeatable) - Dimensions of data to be generated (H, W)
- `-t, --type` (one of: `image`) - Specify the type of data to generate (default: `image`)
- `--model-dir` (path) - Path to load the colorization model from.
  If no model is found, the model will be downloaded (default: current dir)
- `--overwrite` - Allows overwriting existing files in the output directory,
  when it is not empty.
- `-h, --help` - Print the help message and exit.

Examples:
- Generate 300 3-channel fractal images with H=224, W=256 and store in the `images/` dir
  ```console
  datum generate -o images/ --count 300 --shape 224 256
  ```
