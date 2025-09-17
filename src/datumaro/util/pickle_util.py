# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pickle  # nosec B403

import numpy.core.multiarray


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (
            module in ["numpy.core.multiarray", "numpy._core.multiarray"]
            and name in PickleLoader.safe_numpy
        ):
            return getattr(numpy.core.multiarray, name)
        elif module == "numpy" and name in PickleLoader.safe_numpy:
            return getattr(numpy, name)
        # No in unpickling in this particular line
        # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
        raise pickle.UnpicklingError("Global '%s.%s' is forbidden" % (module, name))


class PickleLoader:
    safe_numpy = {
        "dtype",
        "ndarray",
        "_reconstruct",
    }

    def restricted_load(s):
        return RestrictedUnpickler(s, encoding="latin1").load()
