Concepts
########

Basic concepts
--------------

- Dataset - A collection of dataset items, which consist of media and associated annotations.
- Dataset item - A basic single element of the dataset. Also known as `sample`, `entry`.
  In different datasets, it can be an image, a video frame, a whole video, a 3d point cloud, etc.
  Typically, it has corresponding annotations.


Dataset path concepts
---------------------

- Dataset path - A path to a dataset in a special format. They are
  supposed to specify paths to files and directories
  in a uniform way in the CLI.

  - Dataset path - A path to a dataset in the following format:
    `<dataset path>:<format>`
    - `format` is optional. If not specified, it will try to detect automatically


Others
------

- Transform - A transformation operation over dataset elements. Examples
  are image renaming, image flipping, image and subset renaming, label remapping, etc.
  Corresponds to the `transform <../command-reference/transform.md>`_.
