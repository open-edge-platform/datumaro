# How to develop a new builtin plugin

## Enroll a new builtin plugin
If you are adding a new builtin plugin under `./datumaro/plugins/`, you have to enroll it to `./datumaro/plugins/specs.json` for the lazy import.
Otherwise, you will be failed in this test (`tests/unit/test_environment.py`).
You can enroll your plugin to `./datumaro/plugins/specs.json` with Python executable script as follows.

```console
..datumaro$ python src/datumaro/plugins/specs.py
```

> **_NOTE:_**  It is not recommended to mannually modify `./src/datumaro/plugins/specs.json` by hands. Please use the above Python executable script.

## (Additional) Specify extra dependencies to plugin class

If you are trying to add a new builtin plugin that has dependencies which is not installed together by default when installing datumaro.
You need to add Python decorator to specify extra deps for the plugin class definition.
This is required during the built-in plugin registration step to determine if a plugin is available by checking if its dependencies are installed on the system.

For example, Tensorflow Detection API plugin needs `tensorflow` as an extra dependency.
Therefore, it added `@extra_deps("tensorflow", "openvino.tools.accuracy_checker")` to its class definition as follows.

```python
from datumaro.components.lazy_plugin import extra_deps

@extra_deps("tensorflow")
class TfDetectionApiBase(SubsetBase):
    ...
```
