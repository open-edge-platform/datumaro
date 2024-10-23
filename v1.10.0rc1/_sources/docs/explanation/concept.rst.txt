Concepts
########

Basic concepts
--------------

- Dataset - A collection of dataset items, which consist of media and associated annotations.
- Dataset item - A basic single element of the dataset. Also known as `sample`, `entry`.
  In different datasets, it can be an image, a video frame, a whole video, a 3d point cloud, etc.
  Typically, it has corresponding annotations.
- Datumaro project - A combination of multiple datasets, plugins, models, and metadata.


Project versioning concepts
---------------------------

- Data source - A link to a dataset or a copy of a dataset inside a project.
  Basically, it's a URL + dataset format name.
- Project revision - A commit or a reference from Git (branch, tag,
  HEAD~3, etc.). A revision is referenced by data hash. The `HEAD`
  revision is the currently selected revision of the project.
- Revision tree - A project build tree and plugins at a specified revision.
- Working tree - The revision tree in the working directory of a project.
- Data source revision - A state of a data source at a specific stage.
  A revision is referenced by the data hash.
- Object - The data of a revision tree or a data source revision.
  An object is referenced by the data hash.


Dataset path concepts
---------------------

- Dataset revpath - A path to a dataset in a special format. They are
  supposed to specify paths to files, directories or data source revisions
  in a uniform way in the CLI.

  - Dataset path - A path to a dataset in the following format:
    `<dataset path>:<format>`
    - `format` is optional. If not specified, it will try to detect automatically

  - Revision path - A path to a data source revision in a project.
    The syntax is:
    `<project path>@<revision>:<target name>`, any part can be omitted.
    - Default project is the current project (`-p`/`--project` CLI arg.)
      Local revpaths imply that the current project is used and this part
      should be omitted.
    - Default revision is the working tree of the project
    - Default build target is `project`

  - If a path refers to :code:`project` (i.e., target name is not set, or
    this target is exactly specified), the target dataset is the result of
    :ref:`joining <dataset_merging>` all the project data
    sources. Otherwise, if the path refers to a data source revision, the
    corresponding stage from the revision build tree will be used.


Dataset building concepts
-------------------------

- Stage - A revision of a dataset - the original dataset or its modification
  after transformation, filtration or something else. A build tree node.
  A stage is referred by a name.
- Build tree - A directed graph (tree) with root nodes at data sources
  and a single top node called :code:`project`, which represents
  a :ref:`joined <dataset_merging>` dataset.
  Each data source has a starting :code:`root` node, which corresponds to the
  original dataset. The internal graph nodes are stages.
- Build target - A data source or a stage name. Data source names correspond
  to the last stages of data sources.
- Pipeline - A subgraph of a stage, which includes all the ancestors.


Others
------

- Transform - A transformation operation over dataset elements. Examples
  are image renaming, image flipping, image and subset renaming, label remapping, etc.
  Corresponds to the `transform <../command-reference/context_free/transform>`_.
