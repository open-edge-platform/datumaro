# Explore

## Explore datasets

This command explore similar data results for query on dataset. You can use your own query with any image file or text description, even put it on the list. The result includes top-k similar data among target dataset and the visualization of result is saved as png file. This feature is supposed to help users to figure out dataset property easier.

Explorer is a feature that operates on hash basis. Once you put dataset that use as a datasetbase, Explorer calculates hash for every datasetitems in the dataset. Currently, hash of each data is computed based on the CLIP([article](https://arxiv.org/abs/2103.00020)), which could support both image and text modality. Supported model format is Openvino IR and those are uploaded in [openvinotoolkit storage](https://storage.openvinotoolkit.org/repositories/datumaro/models/). When you call Explorer class, hash of whole dataset is started to compute. For database, we use hash for image of each datasetitem. Through CLIP, we extracted feature of image, converted it to binary value and pack the elements into bits. Each hash information is saved as `HashKey` in annotations. Hence, once you call Explorer for the dataset, all datasetitems in dataset have `HashKey` in each annotations.

To explore similar data in dataset, you need to set query first. Query could be image, text, list of images, list of texts and list of images and texts. The query does not need to be an image that exists in the dataset. You can put in any data that you want to explore similar dataset. And you need to set top-k that how much you want to find similar data. The default value for top-k is 50, so if you hope to find more smaller results, you would set top-k. For single query, we computed hamming distance of hash between whole dataset and query. And we sorted those distance and select top-k data which have short distance. For list query, we repeated computing distance for each query and select top-k data based on distance among all dataset.

The command can be applied to a dataset. And if you want to use multiple dataset as database, you could use merged dataset. The current project (`-p/--project`) is also used a context for plugins, so it can be useful for dataset paths having custom formats. When not specified, the current project's working tree is used. To save visualized result (`-s/--save`) is turned off as default. This visualized result is based on [Visualizer](../../jupyter_notebook_examples/notebooks/03_visualize).

Usage:
```console
datum explore [target] [--query-img-path <path/to/image>]
              [--query-item-id </id/of/image/datasetitem> --query-item-subset <subset/of/image>]
              [--query-str <text_query>] [-topk TOPK] [-p PROJECT_DIR] [-s SAVE] [-o DST_DIR] [--stage STAGE]
```

Parameters:
- `<target>` (string) - Target [dataset revpath](../../user-manual/how_to_use_datumaro.md#dataset-path-concepts).
  By default, prints info about the joined `project` dataset.
- `--query-img-path` (string) - Image path to use as query or list of it.
- `--query-item-id` (string) - Datasetitem id of Image to use as query or list of it.
- `--query-item-subset` (string) - Datasetitem subset of Image to use as query or list of it. (default: None)
- `--query-str` (string) - Test to use as query or list of it.
- `-topk` (int) - Number how much you want to find similar data.
- `-p, --project` (string) - Directory of the project to operate on (default: current directory).
- `-s, --save` (bool) - Save explorer result files on explore_result folder.
- `-o, --output-dir` (string) - Output directory. By default, a new directory is created in the current directory.
- `--stage` (bool) - Include this action as a project build step.
  If true, this operation will be saved in the project
  build tree, allowing to reproduce the resulting dataset later.
  Applicable only to main project targets (i.e. data sources
  and the `project` target, but not intermediate stages). Enabled by default.

Examples:
- Explore top10 similar images of image path query.
  ```console
  datum explore -query-img-path <path/to/image> -topk 10 <path/to/dataset/>
  ```

- Explore top10 similar images using image datasetitem id query. Setting the subset of a DatasetItem is not mandatory.
  ```console
  datum explore --query-item-id <id/of/image/datasetitem> -topk 10 <path/to/dataset/>
  ```

- Explore top10 similar images using image datasetitem id and subset query.
  ```console
  datum explore --query-item-id <id/of/image/datasetitem> --query-item-subset <subset/of/image/datasetitem> -topk 10 <path/to/dataset/>
  ```

- Explore top10 similar images of image query within project.
  ```console
  datum project create <...>
  datum project import -f <format> <path/to/dataset/>
  datum explore -query-img-path <path/to/image> -topk 10
  ```

- Explore top10 similar images of text query, elephant
  ```console
  datum explore --query-str elephant -topk 10 <path/to/dataset/>
  ```

- Explore top50 similar images of image query list
  ```console
  datum explore --query-img-path <path/to/image1> <path/to/image2> <path/to/image3> -topk 50 <path/to/dataset/>
  ```

- Explore top50 similar images of text query list
  ```console
  datum explore --query-str motorcycle bus train -topk 50 <path/to/dataset/>
  ```
