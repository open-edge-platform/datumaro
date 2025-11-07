# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from functools import wraps
from inspect import isclass
from itertools import islice
from typing import Any, Dict, Iterable, Tuple, Union

import attrs
import orjson
from json_stream.base import StreamingJSONList, StreamingJSONObject

NOTSET = object()

str_to_bool = attrs.converters.to_bool


def find(iterable, pred=lambda x: True, default=None):
    return next((x for x in iterable if pred(x)), default)


def cast(value, type_conv, default=None):
    if value is None:
        return default
    try:
        return type_conv(value)
    except Exception:
        return default


def to_snake_case(s):
    if not s:
        return ""

    name = [s[0].lower()]
    for idx, char in enumerate(s[1:]):
        idx = idx + 1
        if char.isalpha() and char.isupper():
            prev_char = s[idx - 1]
            if not (prev_char.isalpha() and prev_char.isupper()):
                # avoid "HTML" -> "h_t_m_l"
                name.append("_")
            name.append(char.lower())
        else:
            name.append(char)
    return "".join(name)


def pairs(iterable):
    a = iter(iterable)
    return zip(a, a)


def take_by(iterable, count):
    """
    Returns elements from the input iterable by batches of N items.
    ('abcdefg', 3) -> ['a', 'b', 'c'], ['d', 'e', 'f'], ['g']
    """

    it = iter(iterable)
    while True:
        batch = list(islice(it, count))
        if len(batch) == 0:
            break

        yield batch


def filter_dict(d, exclude_keys):
    return {k: v for k, v in d.items() if k not in exclude_keys}


def parse_str_enum_value(value, enum_class, default=NOTSET, unknown_member_error=None):
    if value is None and default is not NOTSET:
        value = default
    elif isinstance(value, str):
        try:
            value = enum_class[value]
        except KeyError:
            raise ValueError(
                (unknown_member_error or "Unknown element of {cls} '{value}'. The only known are: {available}").format(
                    cls=enum_class.__name__,
                    value=value,
                    available=", ".join(e.name for e in enum_class),
                )
            )
    elif isinstance(value, enum_class):
        pass
    else:
        raise TypeError("Expected value type string or %s, but got %s" % (enum_class.__name__, type(value).__name__))
    return value


def escape(s: str, escapes: Iterable[Tuple[str, str]]) -> str:
    """
    'escapes' is an iterable of (pattern, substitute) pairs
    """

    for pattern, sub in escapes:
        s = s.replace(pattern, sub)
    return s


def unescape(s: str, escapes: Iterable[Tuple[str, str]]) -> str:
    """
    'escapes' is an iterable of (pattern, substitute) pairs
    """

    for pattern, sub in escapes:
        s = s.replace(sub, pattern)
    return s


def is_method_redefined(method_name, base_class, target) -> bool:
    target_method = getattr(target, method_name, None)
    if not isclass(target) and target_method:
        target_method = getattr(target_method, "__func__", None)
    return getattr(base_class, method_name) != target_method


def optional_arg_decorator(fn):
    @wraps(fn)
    def wrapped_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return fn(args[0], **kwargs)

        def real_decorator(decoratee):
            return fn(decoratee, *args, **kwargs)

        return real_decorator

    return wrapped_decorator


def parse_json(data: Union[str, bytes]):
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError as e:
        e.msg += f"\n, error data={data}"
        raise e


def parse_json_file(path: str):
    with open(path, "rb") as f:
        return parse_json(f.read())


def dump_json(
    data: Any,
    *,
    sort_keys: bool = False,
    allow_numpy: bool = True,
    indent: bool = False,
    append_newline: bool = False,
) -> bytes:
    flags = 0
    if sort_keys:
        flags |= orjson.OPT_SORT_KEYS
    if allow_numpy:
        flags |= orjson.OPT_SERIALIZE_NUMPY
    if indent:
        flags |= orjson.OPT_INDENT_2
    if append_newline:
        flags |= orjson.OPT_APPEND_NEWLINE

    return orjson.dumps(data, option=flags)


def dump_json_file(
    path: str,
    data: Any,
    *,
    sort_keys: bool = False,
    allow_numpy: bool = True,
    indent: bool = False,
    append_newline: bool = False,
) -> None:
    with open(path, "wb") as outfile:
        outfile.write(
            dump_json(
                data,
                sort_keys=sort_keys,
                allow_numpy=allow_numpy,
                indent=indent,
                append_newline=append_newline,
            )
        )


def to_dict_from_streaming_json(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, StreamingJSONObject):
        return {k: to_dict_from_streaming_json(v) for k, v in obj.items()}
    if isinstance(obj, StreamingJSONList):
        return [to_dict_from_streaming_json(v) for v in obj]
    return obj
