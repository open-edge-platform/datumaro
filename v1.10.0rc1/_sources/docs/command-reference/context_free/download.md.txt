# Download

## Describe downloadable datasets

This command reports various information about datasets that can be
downloaded with the `download` command. The information is reported either as
human-readable text (the default) or as a JSON object. The format can be selected
with the `--report-format` option.

When the JSON output format is selected, the output document has the following schema:

```json
{
    "<dataset name>": {
        "default_output_format": "<Datumaro format name>",
        "description": "<human-readable description>",
        "download_size": <total size of the downloaded files in bytes>,
        "home_url": "<URL of a web page describing the dataset>",
        "human_name": "<human-readable dataset name>",
        "num_classes": <number of classes in the dataset>,
        "subsets": {
            "<subset name>": {
                "num_items": <number of items in the subset>
            },
            ...
        },
        "version": "<version number>"
    },
    ...
}
```

`home_url` may be `null` if there is no suitable web page for the dataset.

`num_classes` may be `null` if the dataset does not involve classification.

`version` currently contains the version number supplied by TFDS.
In future versions of Datumaro, datasets might come from other sources;
the way version numbers will be set for those is to be determined.

New object members may be added in future versions of Datumaro.

Usage:

```
datum download describe [-h] [--report-format {text,json}]
                        [--report-file REPORT_FILE]
```

Parameters:

- `-h`, `--help` - Print the help message and exit.
- `--report-format` (`text` or `json`) - Format in which to report the information.
  By default, `text` is used.
- `--report-file` (string) - File to which to write the report. By default,
  the report is written to the standard output stream.

## Download datasets

This command downloads a publicly available dataset and saves it to a local
directory.
In terms of syntax, this command is similar to [`convert`](./convert.md),
but instead of taking a local directory as the source, it takes a dataset ID.
A list of supported datasets and output formats can be found in the `--help`
output of this command.

Currently, the only source of datasets is the TensorFlow Datasets library.
Therefore, to use this command you must install TensorFlow & TFDS, which you can
do as follows:

```sh
pip install datumaro[tf,tfds]
```

To use a proxy for downloading, configure it with the conventional
[curl environment variables](https://everything.curl.dev/usingcurl/proxies/env).

Usage:

```console
datum download get [-h] -i DATASET_ID [-f OUTPUT_FORMAT] [-o DST_DIR]
                   [--overwrite] [-s SUBSET] [-- EXTRA_EXPORT_ARGS]
```

Parameters:

- `-h`, `--help` - Print the help message and exit.
- `-i`, `--dataset-id` (string) - ID of the dataset to download.
- `-f`, `--output-format` (string) - Output format. By default, the format
  of the original dataset is used.
- `-o, --output-dir` (string) - Output directory. By default, a subdirectory
  in the current directory is used.
- `--overwrite` - Allows overwriting existing files in the output directory,
  when it is not empty.
- `--subset` (string) - Which subset of the dataset to save. By default, all
  subsets are saved. Note that due to limitations of TFDS, all subsets are
  downloaded even if this option is specified.
- `-- <extra export args>` - Additional arguments for the format writer
  (use `-- -h` for help). Must be specified after the main command arguments.

Examples:
- Download the MNIST dataset, saving it in the ImageNet text format
  ```console
  datum download get -i tfds:mnist -f imagenet_txt -- --save-media
  ```
