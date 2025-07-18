# Detect

## Detect dataset format

This command attempts to detect the format of a dataset in a directory.
Currently, only local directories are supported.

The detection result may be one of:

- a single format being detected.
- no formats being detected (if the dataset doesn't match any known format).
- multiple formats being detected (if the dataset is ambiguous).

The command outputs this result in a human-readable form and
optionally as a machine-readable JSON report (see `--json-report`).

The format of the machine-readable report is as follows:

```json
{
    "detected_formats": [
        "detected-format-name-1", "detected-format-name-2", ...
    ],
    "rejected_formats": {
        "rejected-format-name-1": {
            "reason": <reason-code>,
            "message": "line 1\nline 2\n...\nline N"
        },
        "rejected-format-name-2": ...,
        ...
    }
}
```

The `<reason-code>` can be one of:

- `"detection_unsupported"`: the corresponding format does not support
  detection.

- `"insufficient_confidence"`: the dataset matched the corresponding format,
  but it matched at least one other format better.

- `"unmet_requirements"`: the dataset didn't meet at least one requirement
  of the corresponding format.

Other reason codes may be defined in the future.

Usage:

```console
datum detect [-h] [-p PROJECT_DIR] [--show-rejections]
             [--json-report JSON_REPORT] [--depth DEPTH] url
```

Parameters:

- `<url>` - Path to the dataset to analyse.
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory). The project might contain local plugins with
  custom formats, which will be used for detection.
- `--show-rejections` - Describe why each supported format that wasn't
  detected was rejected. This only affects the human-readable output; the
  machine-readable report always includes rejection information.
- `--json-report` (string) - Path to which to save a JSON report describing
  detected and rejected formats. By default, no report is saved.
- `--depth` (int) - The maximum depth for recursive search. (default: 2)
- `-h`, `--help` - Print the help message and exit.

Examples:
- Detect the format of a dataset in a given directory,
  showing rejection information
  ```console
  datum detect --show-rejections <path/to/dataset/>
  ```
