# Copyright (C) 2026 Datumaro contributors
#
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from datumaro.cli import __main__ as cli


@pytest.fixture
def cli_env(tmp_path):
    # Make installed CLI extras unavailable in a fresh interpreter without
    # changing the test environment's packages or leaking imported modules.
    (tmp_path / "sitecustomize.py").write_text(
        """import importlib.abc
import importlib.util
import os
import sys

missing = set(filter(None, os.environ.get("DATUMARO_TEST_MISSING", "").split(",")))
original_find_spec = importlib.util.find_spec

def find_spec(name, *args, **kwargs):
    if name.split(".")[0] in missing:
        return None
    return original_find_spec(name, *args, **kwargs)

class BlockMissingModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in missing:
            raise ModuleNotFoundError(f"No module named '{fullname}'", name=fullname)

importlib.util.find_spec = find_spec
sys.meta_path.insert(0, BlockMissingModules())
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(cli.__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]
    )
    env["DATUMARO_TEST_MISSING"] = ""
    return env


def _run_cli(entry_point, args, env):
    if entry_point == "console":
        script = Path(sysconfig.get_path("scripts")) / ("datum.exe" if os.name == "nt" else "datum")
        command = [str(script)]
    else:
        command = [sys.executable, "-m", entry_point]
    return subprocess.run(command + args, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("entry_point", ["console", "datumaro", "datumaro.cli"])
@pytest.mark.parametrize("missing", ["tensorboardX", "tabulate", "scipy", "matplotlib"])
def test_missing_cli_dependency_reports_install_hint(entry_point, missing, cli_env):
    cli_env["DATUMARO_TEST_MISSING"] = missing
    result = _run_cli(entry_point, ["--help"], cli_env)

    assert result.returncode == 1
    assert missing in result.stderr
    assert "pip install 'datumaro[cli]'" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("entry_point", ["console", "datumaro", "datumaro.cli"])
def test_base_library_remains_importable_without_cli_extra(entry_point, cli_env):
    cli_env["DATUMARO_TEST_MISSING"] = "tensorboardX,tabulate,scipy,matplotlib"
    library = subprocess.run(
        [sys.executable, "-c", "import datumaro; assert datumaro.Dataset is not None"],
        env=cli_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert library.returncode == 0, library.stderr

    result = _run_cli(entry_point, ["--help"], cli_env)
    assert result.returncode == 1
    for missing in ("tensorboardX", "tabulate", "scipy", "matplotlib"):
        assert missing in result.stderr
    assert "datumaro[cli]" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("entry_point", ["console", "datumaro", "datumaro.cli"])
@pytest.mark.parametrize("args", [["--help"], ["--version"]])
def test_cli_extra_keeps_help_and_version_working(entry_point, args, cli_env):
    result = _run_cli(entry_point, args, cli_env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


@pytest.mark.parametrize(
    "missing",
    [
        ("tensorboardX",),
        ("tabulate",),
        ("scipy",),
        ("matplotlib",),
        ("tensorboardX", "tabulate", "scipy", "matplotlib"),
    ],
)
def test_main_reports_missing_dependencies_before_loading_commands(missing, monkeypatch, caplog):
    def find_spec(name):
        return None if name in missing else object()

    def fail_parser():
        pytest.fail("Commands must not be loaded when CLI dependencies are missing")

    monkeypatch.setattr(cli.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(cli, "make_parser", fail_parser)

    assert cli.main(["--help"]) == 1
    assert f"Missing optional CLI dependencies: {', '.join(missing)}." in caplog.text
    assert "pip install 'datumaro[cli]'" in caplog.text


@pytest.mark.parametrize("args, expected", [(["--help"], "Dataset Framework"), (["--version"], cli.__version__)])
def test_main_with_cli_extra_keeps_help_and_version_working(args, expected, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(args)

    assert error.value.code == 0
    assert expected in capsys.readouterr().out


def test_unrelated_import_failure_is_not_hidden(monkeypatch, caplog):
    def fail_parser():
        raise ModuleNotFoundError("broken installed plugin", name="unrelated_plugin_dependency")

    monkeypatch.setattr(cli, "make_parser", fail_parser)
    with pytest.raises(ModuleNotFoundError, match="broken installed plugin"):
        cli.main(["--help"])
    assert "datumaro[cli]" not in caplog.text
