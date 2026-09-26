# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import importlib
import os
import os.path as osp
import re
import shutil
import subprocess  # nosec B404
import sys
import unicodedata
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Iterable, Iterator, List, Optional, Set, Union

from . import cast
from .definitions import DEFAULT_SUBSET_NAME

DEFAULT_MAX_DEPTH = 10
DEFAULT_MIN_DEPTH = 0


def check_instruction_set(instruction):
    return instruction == str.strip(
        # Let's ignore a warning from bandit about using shell=True.
        # In this case it isn't a security issue and we use some
        # shell features like pipes.
        subprocess.check_output('lscpu | grep -o "%s" | head -1' % instruction, shell=True).decode(  # nosec B602
            "utf-8"
        )
    )


def _win32_trim_dot_space(component: str) -> str:
    """Reproduce Win32's silent trimming of trailing dots/spaces from a path
    component when a file/directory is created through the regular (non-\\\\?\\)
    Win32 API. Trimming stops as soon as the remainder is exactly "." or "..",
    since those are special directory references, not trimmed further.
    So e.g. ".. ", "..." and ".. ." (any trailing run of dots/spaces after
    "..") all resolve to the parent directory "..".
    """
    while component and component[-1] in ". ":
        if component in (".", ".."):
            break
        component = component[:-1]
    return component


def contains_unsafe_path_component(path: str) -> bool:
    """Check if any "/" or "\\"-separated component of `path` is (or, once
    Win32's trailing dot/space trimming is accounted for, resolves to) the
    parent directory reference "..".
    """
    return any(_win32_trim_dot_space(component) == ".." for component in re.split(r"[\\/]", path))


def _resolve_existing_part(path: str) -> str:
    """Resolve symlinks in the longest prefix of `path` that already exists on
    disk, then lexically re-append the (not yet created) remaining tail.

    Plain `osp.realpath(path)` is not used for the whole path because on
    Windows it resolves nonexistent paths with a "best effort" fallback that
    can disagree with the resolution of an already-existing ancestor (e.g.
    `base_dir`) resolved via a separate call, causing spurious escape errors
    for perfectly safe, not-yet-created destination paths (this matters here
    since directories are created after this check, not before).
    """
    head = osp.abspath(path)
    tail_parts: List[str] = []
    while head and not osp.exists(head):
        head, name = osp.split(head)
        if not name:
            break
        tail_parts.append(name)

    resolved_head = osp.realpath(head) if head else head
    return osp.normpath(osp.join(resolved_head, *reversed(tail_parts))) if tail_parts else resolved_head


def _is_within_base(path: str, base_dir: str) -> bool:
    resolved_base = osp.realpath(base_dir)
    resolved_path = _resolve_existing_part(path)
    try:
        return osp.commonpath([resolved_base, resolved_path]) == resolved_base
    except ValueError:
        return False


def join_within_base(base_dir: str, *parts: str) -> str:
    """Join `parts` onto `base_dir`, and reject the result if it escapes `base_dir`.

    Untrusted values (e.g. dataset item ids) can contain '..' or be absolute,
    so a plain osp.join() can traverse outside of `base_dir`. Raises ValueError
    in that case (including when the paths don't share a common root, e.g.
    different drives on Windows, or when a part would resolve to '..' only
    after Win32's trailing dot/space trimming is applied).
    """
    for part in parts:
        if contains_unsafe_path_component(part):
            raise ValueError(f"Path component {part!r} is not allowed to reference the parent directory")

    path = osp.join(base_dir, *parts)
    if not _is_within_base(path, base_dir):
        raise ValueError(f"Resulting path '{path}' escapes base directory '{base_dir}'")
    return path


def ensure_within_base(path: str, base_dir: str) -> str:
    """Validate that an already-constructed `path` (e.g. one supplied directly
    by a caller instead of being composed from parts via `join_within_base`)
    stays within `base_dir`. Raises ValueError otherwise.
    """
    if contains_unsafe_path_component(path) or not _is_within_base(path, base_dir):
        raise ValueError(f"Path '{path}' escapes base directory '{base_dir}'")
    return path


def import_foreign_module(name, path):
    module = None
    default_path = sys.path.copy()
    try:
        sys.path = [osp.abspath(path), *default_path]
        sys.modules.pop(name, None)  # remove from cache
        module = importlib.import_module(name)
        sys.modules.pop(name)  # remove from cache
    finally:
        sys.path = default_path
    return module


def walk(path, max_depth: Optional[int] = None, min_depth: Optional[int] = None):
    if max_depth is None:
        max_depth = DEFAULT_MAX_DEPTH
    if min_depth is None:
        min_depth = DEFAULT_MIN_DEPTH

    baselevel = path.count(osp.sep)
    for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=True):
        curlevel = dirpath.count(osp.sep)
        if baselevel + min_depth > curlevel:
            continue

        if baselevel + max_depth <= curlevel:
            dirnames.clear()  # topdown=True allows to modify the list

        yield dirpath, dirnames, filenames


def find_files(
    dirpath: str,
    exts: Union[str, Iterable[str]],
    recursive: bool = False,
    max_depth: Optional[int] = None,
    min_depth: Optional[int] = None,
) -> Iterator[str]:
    if isinstance(exts, str):
        exts = {"." + exts.lower().lstrip(".")}
    else:
        exts = {"." + e.lower().lstrip(".") for e in exts}

    def _check_ext(filename: str):
        dotpos = filename.rfind(".")
        if dotpos > 0:  # exclude '.ext' cases too
            ext = filename[dotpos:].lower()
            if ext in exts:
                return True
        return False

    for d, _, filenames in walk(
        dirpath, max_depth=max_depth if recursive else 0, min_depth=min_depth if recursive else 0
    ):
        for filename in filenames:
            if not _check_ext(filename):
                continue

            yield osp.join(d, filename)


def copytree(src, dst):
    # Serves as a replacement for shutil.copytree().
    #
    # Shutil works very slow pre 3.8
    # https://docs.python.org/3/library/shutil.html#platform-dependent-efficient-copy-operations
    # https://bugs.python.org/issue33671

    if sys.version_info >= (3, 8):
        shutil.copytree(src, dst)
        return

    assert src and dst
    src = osp.abspath(src)
    dst = osp.abspath(dst)

    if not osp.isdir(src):
        raise FileNotFoundError("Source directory '%s' doesn't exist" % src)

    if osp.isdir(dst):
        raise FileExistsError("Destination directory '%s' already exists" % dst)

    dst_basedir = osp.dirname(dst)
    if dst_basedir:
        os.makedirs(dst_basedir, exist_ok=True)

    try:
        if sys.platform == "windows":
            # Ignore
            #   B603: subprocess_without_shell_equals_true
            #   B607: start_process_with_partial_path
            # In this case we control what is called and command arguments
            # PATH overriding is considered low risk
            subprocess.check_output(  # nosec B603, B607
                ["xcopy", src, dst, "/s", "/e", "/q", "/y", "/i"],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        elif sys.platform == "linux":
            # As above
            subprocess.check_output(  # nosec B603, B607
                ["cp", "-r", "--", src, dst],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        else:
            shutil.copytree(src, dst)
    except subprocess.CalledProcessError as e:
        raise Exception(
            "Failed to copy data. The command '%s' has failed with the following output: '%s'" % (e.cmd, e.stdout)
        ) from e


@contextmanager
def suppress_output(stdout: bool = True, stderr: bool = False):
    with open(os.devnull, "w") as devnull, ExitStack() as es:
        if stdout:
            es.enter_context(redirect_stdout(devnull))
        elif stderr:
            es.enter_context(redirect_stderr(devnull))

        yield


@contextmanager
def catch_output():
    stdout = StringIO()
    stderr = StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        yield stdout, stderr


def dir_items(path, ext, truncate_ext=False):
    items = []
    for f in os.listdir(path):
        ext_pos = f.rfind(ext)
        if ext_pos != -1:
            if truncate_ext:
                f = f[:ext_pos]
            items.append(f)
    return items


def split_path(path):
    path = osp.normpath(path)
    parts = []

    while True:
        path, part = osp.split(path)
        if part:
            parts.append(part)
        else:
            if path:
                parts.append(path)
            break
    parts.reverse()

    return parts


def is_subpath(path: str, base: str) -> bool:
    """
    Tests if a path is subpath of another path or the paths are equal.
    """

    base = osp.abspath(base)
    path = osp.abspath(path)
    return osp.join(path, "").startswith(osp.join(base, ""))


def make_file_name(s: str) -> str:
    # adapted from
    # https://docs.djangoproject.com/en/2.1/_modules/django/utils/text/#slugify
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore")
    s = s.decode()
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[-\s]+", "-", s)


def generate_next_name(
    names: Iterable[str],
    basename: str,
    sep: str = ".",
    suffix: str = "",
    default: Optional[str] = None,
) -> str:
    """
    Generates the "next" name by appending a next index to the occurrence
    of the basename with the highest index in the input collection.

    Returns: next string name

    Example:

    Inputs:
        name_abc

        name_base

        name_base1

        name_base5

    Basename: name_base

    Output: name_base6
    """

    pattern = re.compile(r"%s(?:%s(\d+))?%s" % tuple(map(re.escape, [basename, sep, suffix])))
    matches = [match for match in (pattern.match(n) for n in names) if match]

    max_idx = max([cast(match[1], int, 0) for match in matches], default=None)
    if max_idx is None:
        if default is not None:
            idx = sep + str(default)
        else:
            idx = ""
    else:
        idx = sep + str(max_idx + 1)
    return basename + idx + suffix


def extract_subset_name_from_parent(url: str, start: str) -> str:
    """Extract subset name from the given url.

    For example, if url = "/a/b/images/train/img.jpg" and start = "/a/b",
    it will return "train". On the other hand, if url = "/a/b/images/img.jpg"
    and start = "/a/b", it will return DEFAULT_SUBSET_NAME.

    Parameters
    ----------
    url: str
        Given url to extract subset
    start:
        The head path of url to obtain the relative path from the url

    Returns
    -------
    str
        Subset name
    """
    relpath = osp.relpath(url, start)
    relpath, _ = osp.split(relpath)
    relpath, subdir_name = osp.split(relpath)

    if relpath == "":
        return DEFAULT_SUBSET_NAME

    return subdir_name


def get_all_file_extensions(path: str, ignore_dirs: Set[str]) -> List[str]:
    extensions = set()
    for p in glob.iglob(osp.join(path, "**", "*.*"), recursive=True):
        if ignore_dirs.isdisjoint(p.split(os.sep)):
            extensions.add(osp.splitext(p)[1])
    return list(extensions)
