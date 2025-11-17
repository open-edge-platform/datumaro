# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import re

emoji_pattern = re.compile(
    "|".join(
        [
            "[\U0001f600-\U0001f64f]",  # emoticons
            "[\U0001f300-\U0001f5ff]",  # symbols & pictographs
            "[\U0001f680-\U0001f6ff]",  # transport & map symbols
            "[\U0001f1e0-\U0001f1ff]",  # flags (iOS)
            "[\u2600-\u26ff]",  # Miscellaneous Symbols
            "[\u2700-\u27bf]",  # Dingbats
            "[\U0001f900-\U0001f9ff]",  # Supplemental Symbols and Pictographs
        ]
    ),
    flags=re.UNICODE,
)
