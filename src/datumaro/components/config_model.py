# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.config import Config
from datumaro.components.config import DictConfig as _DictConfig
from datumaro.components.config import SchemaBuilder as _SchemaBuilder
from datumaro.util import find

SOURCE_SCHEMA = (
    _SchemaBuilder()
    .add("url", str)
    .add("path", str)
    .add("format", str)
    .add("options", dict)
    .add("hash", str)
    .build()
)


class Source(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=SOURCE_SCHEMA)

    @property
    def is_generated(self) -> bool:
        return not self.url


MODEL_SCHEMA = _SchemaBuilder().add("options", dict).build()


class Model(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=MODEL_SCHEMA)


BUILDSTAGE_SCHEMA = (
    _SchemaBuilder()
    .add("name", str)
    .add("type", str)
    .add("kind", str)
    .add("hash", str)
    .add("params", dict)
    .build()
)


class BuildStage(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=BUILDSTAGE_SCHEMA)


BUILDTARGET_SCHEMA = _SchemaBuilder().add("stages", list).add("parents", list).build()


class BuildTarget(Config):
    def __init__(self, config=None):
        super().__init__(config, schema=BUILDTARGET_SCHEMA)
        self.stages = [BuildStage(o) for o in self.stages]

    @property
    def root(self):
        return self.stages[0]

    @property
    def head(self):
        return self.stages[-1]

    @property
    def has_stages(self) -> bool:
        return 1 < len(self.stages)

    def find_stage(self, stage):
        if stage == "root":
            return self.root
        elif stage == "head":
            return self.head
        return find(self.stages, lambda x: x.name == stage or x == stage)

    def get_stage(self, stage):
        res = self.find_stage(stage)
        if res is None:
            raise KeyError("Unknown stage '%s'" % stage)
        return res


TREE_SCHEMA = (
    _SchemaBuilder()
    .add("format_version", int)
    .add("sources", lambda: _DictConfig(lambda v=None: Source(v)))
    .add("build_targets", lambda: _DictConfig(lambda v=None: BuildTarget(v)))
    .add("base_dir", str, internal=True)
    .add("config_path", str, internal=True)
    .build()
)

TREE_DEFAULT_CONFIG = Config(
    {
        "format_version": 2,
        "config_path": "",
    },
    mutable=False,
    schema=TREE_SCHEMA,
)


class TreeConfig(Config):
    def __init__(self, config=None, mutable=True):
        super().__init__(
            config=config, mutable=mutable, fallback=TREE_DEFAULT_CONFIG, schema=TREE_SCHEMA
        )


PROJECT_SCHEMA = (
    _SchemaBuilder()
    .add("format_version", int)
    .add("models", lambda: _DictConfig(lambda v=None: Model(v)))
    .build()
)

PROJECT_DEFAULT_CONFIG = Config(
    {
        "format_version": 2,
    },
    mutable=False,
    schema=PROJECT_SCHEMA,
)


class ProjectConfig(Config):
    def __init__(self, config=None, mutable=True):
        super().__init__(
            config=config, mutable=mutable, fallback=PROJECT_DEFAULT_CONFIG, schema=PROJECT_SCHEMA
        )


class PipelineConfig(Config):
    pass


class ProjectLayout:
    aux_dir = ".datumaro"
    cache_dir = "cache"
    index_dir = "index"
    working_tree_dir = "tree"
    head_file = "head"
    tmp_dir = "tmp"
    models_dir = "models"
    plugins_dir = "plugins"
    conf_file = "config.yml"


class TreeLayout:
    conf_file = "config.yml"
    sources_dir = "sources"
