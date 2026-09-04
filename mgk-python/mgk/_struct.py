"""Struct-compatible recursive mapping used by the public solver."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class Struct(dict):
    """Dictionary with attribute access and recursive value conversion."""

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("__") and dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        return super().__getattribute__(name)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, as_struct(value))

    def update(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        for key, value in dict.items(values):
            self[key] = value

    def copy(self) -> "Struct":
        return Struct(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Struct":
        result = Struct()
        memo[id(self)] = result
        for key, value in dict.items(self):
            result[key] = deepcopy(value, memo)
        return result

    def deepcopy(self) -> "Struct":
        return deepcopy(self)


def as_struct(value: Any) -> Any:
    if isinstance(value, Struct):
        return value
    if isinstance(value, Mapping):
        return Struct(value)
    if isinstance(value, list):
        return [as_struct(item) for item in value]
    if isinstance(value, tuple):
        return tuple(as_struct(item) for item in value)
    return value
