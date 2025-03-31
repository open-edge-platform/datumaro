# Contribution Guide

We appreciate any contribution to [Datumaro](https://github.com/openvinotoolkit/datumaro),
whether it's in the form of a Pull Request, Feature Request or general comments/issue that you found.
For feature requests and issues, please feel free to create a GitHub Issue in this repository.

## Related sections

- [Design document](https://openvinotoolkit.github.io/datumaro/latest/docs/explanation/architecture)
- [Developer manual](https://openvinotoolkit.github.io/datumaro/latest/docs/reference/datumaro_module)

## Development and pull requests

### Prerequisites
- Python (3.9+)

To set up your development environment, please follow the steps below.

0. Because Datumaro has some C++ and Rust implementations to improve Python performance, you should install C++ compiler (`apt-get install build-essential`) and a [Rust toolchain](https://www.rust-lang.org/tools/install) in your system to build the binary extensions.

1. Fork the [repo](https://github.com/openvinotoolkit/datumaro).

2. clone the forked repo.
    ``` bash
    git clone <forked_repo>
    ```
3. Optionally, install a virtual environment (recommended):
    ``` bash
    python -m pip install virtualenv
    python -m virtualenv venv
    . venv/bin/activate
    ```

4. Install Datumaro with [optional dependencies](#optional-dependencies):
    ``` bash
    cd /path/to/the/cloned/repo/
    pip install -e .[tf,tfds,torch,default]
    ```

5. Install dev & test dependencies:
    ``` bash
    pip install -r requirements-dev.txt
    pip install -r tests/requirements.txt
    ```

6. Set up pre-commit hooks in the repo. See [Code style](#code-style).
    ``` bash
    pre-commit install
    pre-commit run
    ```

7. Create your branch based off the `develop` branch and make changes.

8. Verify your code by running unit tests and integration tests. See [Testing](#testing)
    ``` bash
    pytest -v
    ```
    or
    ``` bash
    python -m pytest -v
    ```

9. Push your changes.

Now you are ready to create a PR(Pull Request) and get review.

## Optional dependencies

Developer should install the following optional components for running our tests:

- OpenVINO
- TensorFlow
- PyTorch
- MxNet
- Caffe

## Usage

``` bash
datum --help
python -m datumaro --help
python datumaro/ --help
python datum.py --help
```

``` python
import datumaro
```

## Code style

Try to be readable and consistent with the existing codebase.

The project uses Black for code formatting and isort for sorting import statements.
You can find corresponding configurations in `pyproject.toml` in the repository root.
No trailing whitespaces, at most 100 characters per line.

Datumaro includes a Git pre-commit hook, `.pre-commit-config.yaml` that can help you follow the style requirements. To install, make sure isort and black are installed on your system, then run `pre-commit run`.

## Environment

The recommended editor is VS Code with the Python language plugin.

<a id="testing"></a>
## Testing

It is expected that all Datumaro functionality is covered and checked by
unit tests. Tests are placed in the `tests/unit/` directory. Additional
pre-generated files for tests can be stored in the `tests/assets/` directory.
CLI tests are separated from the core tests, they are stored in the
`tests/integration/cli/` directory.

Currently, we use [`pytest`](https://docs.pytest.org/) for testing.

To run tests use:

``` bash
pytest -v
```
or
``` bash
python -m pytest -v
```

<a id="Test_case_description"></a>
### Test cases

<a id="Test_marking"></a>
### Test marking

For better integration with CI and requirements tracking,
we use special annotations for tests.

A test needs to linked with a requirement it is related to. To link a
test, use:

```python
from unittest import TestCase
from .requirements import Requirements, mark_requirement

class MyTests(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_my_requirement(self):
        ... do stuff ...
```

Such marking will apply markings from the requirement specified.
They can be overridden for a specific test:

```python
import pytest

class MyTests(TestCase):
    @pytest.mark.priority_low
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_my_requirement(self):
        ... do stuff ...
```

<a id="Requirements"></a>
#### Requirements

Requirements and other links need to be added to [`tests/requirements.py`](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/requirements.py):

```python
DATUM_244 = "Add Snyk integration"
DATUM_BUG_219 = "Return format is not uniform"
```

```python
# Fully defined in GitHub issues:
@pytest.mark.reqids(Requirements.DATUM_244, Requirements.DATUM_333)

# And defined any other way:
@pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
```

##### Available annotations for tests and requirements

Markings are defined in [`tests/conftest.py`](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/conftest.py).

**A list of requirements and bugs**
```python
@pytest.mark.requids(Requirements.DATUM_123)
@pytest.mark.bugs(Requirements.DATUM_BUG_456)
```

**A priority**
```python
@pytest.mark.priority_low
@pytest.mark.priority_medium
@pytest.mark.priority_high
```

**Component**
The marking used for indication of different system components

```python
@pytest.mark.components(DatumaroComponent.Datumaro)
```

**Skipping tests**

```python
@pytest.mark.skip(SkipMessages.NOT_IMPLEMENTED)
```

**Parametrized runs**

Parameters are used for running the same test with different parameters e.g.

```python
@pytest.mark.parametrize("numpy_array, batch_size", [
    (np.zeros([2]), 0),
    (np.zeros([2]), 1),
    (np.zeros([2]), 2),
    (np.zeros([2]), 5),
    (np.zeros([5]), 2),
])
```

<a id="TestDoc"></a>
### Test documentation

Tests are documented with docs strings. Test descriptions must contain
the following: sections: `Description`, `Expected results` and `Steps`.

```python
def test_can_convert_polygons_to_mask(self):
    """
    <b>Description:</b>
    Ensure that the dataset polygon annotation can be properly converted
    into dataset segmentation mask.

    <b>Expected results:</b>
    Dataset segmentation mask converted from dataset polygon annotation
    is equal to an expected mask.

    <b>Steps:</b>
    1. Prepare dataset with polygon annotation
    2. Prepare dataset with expected mask segmentation mode
    3. Convert source dataset to target, with conversion of annotation
      from polygon to mask.
    4. Verify that resulting segmentation mask is equal to the expected mask.
    """
```
