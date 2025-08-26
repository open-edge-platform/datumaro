============================
Level 12: Framework Conversion
============================

Datumaro allows seamless conversion of datasets to popular deep learning frameworks, such as PyTorch and TensorFlow.
This is particularly useful when you are working with a dataset that needs to be used across different frameworks
without manual reformatting.

Datumaro provides the FrameworkConverter class, which can be used to convert a dataset for various tasks
like classification, detection, and segmentation.

Supported Tasks
  - Classification
  - Multilabel Classification
  - Detection
  - Instance Segmentation
  - Semantic Segmentation
  - Tabular Data

.. tab-set::

  .. tab-item:: Python

    With the PyTorch framework, you can convert a Datumaro dataset like this:

    .. code-block:: python

      from datumaro.plugins.framework_converter import FrameworkConverter
      from torchvision import transforms

      transform = transforms.Compose([transforms.ToTensor()])
      dm_dataset = ...  # Load your dataset here

    First, we have to specify the dataset, subset, and task

    .. code-block:: python

      multi_framework_dataset = FrameworkConverter(dm_dataset, subset="train", task="classification")
      train_dataset = multi_framework_dataset.to_framework(framework="torch", transform=transform)

    Through this, we convert the dataset to PyTorch format

    .. code-block:: python

      from torch.utils.data import DataLoader
      train_loader = DataLoader(train_dataset, batch_size=32)

    Now we can use the train_dataset with PyTorch DataLoader

In this example:

- `subset="train"` indicates that we are working with the training portion of the dataset.

- `task="classification"` specifies that this is a classification task.

- The dataset is converted to PyTorch-compatible format using the `to_framework` method.
