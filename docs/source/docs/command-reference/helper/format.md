# Format

## List Supported Data Formats

This command shows a list of supported import/export data formats in Datumaro.
It is useful on a quick reference of data format name used for other CLI command such as [convert](../context_free/convert.md), [import](../context/sources.md#import-dataset), or [export](../context/export.md#export-datasets). For more detailed guides on each data format, please visit [our Data Formats section](../../data-formats/formats/index.rst).

Usage:

```console
usage: datum format [-h] [-li | -le] [-d DELIMITER]
```

Parameters:
- `-h, --help` - Print the help message and exit.
- `-d DELIMITER, --delimiter DELIMITER` - Separator used to list data format names (default: `\n`). For example, `datum format -d ','` command displays
  ```console
  Supported import formats:
  ade20k2017,ade20k2020,align_celeba,...
  ```
- `-li, --list-import` - List all supported import data format names
- `-le, --list-export` - List all supported export data format names
