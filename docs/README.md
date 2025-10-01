# How to build Datumaro docs
## Prerequisite

1. Install the dependencies for building documents.

    ```console
    uv pip install -r docs/requirements.txt
    ```

## Tips

1. We auto-generate Python module docs by [sphinx.ext.autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html). The auto-generated `rst` files will be created in `docs/source/docs/reference/_autosummary`. **If you changed some code structures and want to see the updated Python module API docs, you have to remove this directory and regenerate it.** This can be done by the following command.

    ```console
    make clean
    make html
    ```
