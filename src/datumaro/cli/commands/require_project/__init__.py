# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from . import modification, versioning

deprecated = "[DEPRECATED, will be removed in 1.12]"


def get_project_commands():
    return [
        ("Project modification:", None, deprecated),
        ("add", modification.add, f"{deprecated} Add dataset"),
        ("create", modification.create, f"{deprecated} Create empty project"),
        ("export", modification.export, f"{deprecated} Export dataset in some format"),
        ("import", modification.import_, f"{deprecated} Import dataset"),
        ("remove", modification.remove, f"{deprecated} Remove dataset"),
        ("", None, ""),
        ("Project versioning:", None, deprecated),
        ("checkout", versioning.checkout, f"{deprecated} Switch to another branch or revision"),
        ("commit", versioning.commit, f"{deprecated} Commit changes in tracked files"),
        ("log", versioning.log, f"{deprecated} List history"),
        ("info", versioning.info, f"{deprecated} Print project information"),
        ("status", versioning.status, f"{deprecated} Display current branch and revision status"),
    ]
