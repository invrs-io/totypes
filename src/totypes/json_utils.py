"""Utility functions related to json packing and unpacking.

Copyright (c) 2023 Martin F. Schubert
"""

import dataclasses
import functools
import io
import json
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import numpy as onp
from jax import tree_util

from totypes import types

PyTree = Any

NUMPY_ASCII_FORMAT = "latin-1"
PREFIX_NUMPY = "\x93NUMPY"

CUSTOM_TYPES_AND_PREFIXES = (
    (types.BoundedArray, "\x93TYPES.BOUNDED_ARRAY"),
    (types.Density2DArray, "\x93TYPES.DENSITY_2D_ARRAY"),
)


def json_from_pytree(
    pytree: PyTree,
    extra_custom_types_and_prefixes: Tuple[Tuple[Any, str], ...] = (),
) -> str:
    """Serializes a pytree containing arrays into a json string."""
    custom_types_and_prefixes = tuple(
        set(CUSTOM_TYPES_AND_PREFIXES + extra_custom_types_and_prefixes)
    )
    _validate_prefixes(custom_types_and_prefixes)
    custom_types, _ = zip(*custom_types_and_prefixes)
    pytree_with_serialized = tree_util.tree_map(
        f=functools.partial(
            _maybe_serialize,
            custom_types_and_prefixes=custom_types_and_prefixes,
        ),
        tree=pytree,
        is_leaf=lambda x: isinstance(x, custom_types),
    )
    return json.dumps(pytree_with_serialized)


def _maybe_serialize(
    obj: Any,
    custom_types_and_prefixes: Tuple[Tuple[Any, str], ...],
) -> Any:
    """Serializes `obj` if it is a recognized custom type."""
    if isinstance(obj, (onp.ndarray, jnp.ndarray)):
        return _serialize_array(obj)
    for custom_type, prefix in custom_types_and_prefixes:
        if isinstance(obj, custom_type):
            return (
                f"{prefix}{json_from_pytree(_asdict(obj), custom_types_and_prefixes)}"
            )
    return obj


def _asdict(x: Any) -> Dict[str, Any]:
    """Converts dataclasses or namedtuples to dictionaries."""
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    try:
        return x._asdict()  # type: ignore[no-any-return]
    except AttributeError as exc:
        raise ValueError(
            f"`x` must be a dataclass or a namedtuple, but got {type(x)}"
        ) from exc


def _serialize_array(obj: Union[onp.ndarray, jnp.ndarray]) -> str:
    """Serializes a numpy array to a string."""
    obj = onp.asarray(obj)
    memfile = io.BytesIO()
    onp.save(memfile, obj)
    return memfile.getvalue().decode(NUMPY_ASCII_FORMAT)


def pytree_from_json(
    serialized: str,
    extra_custom_types_and_prefixes: Tuple[Tuple[Any, str], ...] = (),
) -> PyTree:
    """Deserializes a json string into a pytree of arrays."""
    custom_types_and_prefixes = tuple(
        set(CUSTOM_TYPES_AND_PREFIXES + extra_custom_types_and_prefixes)
    )
    _validate_prefixes(custom_types_and_prefixes)
    pytree_with_serialized_arrays = json.loads(serialized)
    return tree_util.tree_map(
        functools.partial(
            _maybe_deserialize,
            custom_types_and_prefixes=custom_types_and_prefixes,
        ),
        pytree_with_serialized_arrays,
    )


def _maybe_deserialize(
    maybe_serialized: Any,
    custom_types_and_prefixes: Tuple[Tuple[Any, str], ...],
) -> Any:
    """Deserializes data if it is in a recognized format."""
    if not isinstance(maybe_serialized, str):
        return maybe_serialized

    if maybe_serialized.startswith(PREFIX_NUMPY):
        return _deserialize_array(maybe_serialized)
    for custom_type, prefix in custom_types_and_prefixes:
        if maybe_serialized.startswith(prefix):
            data = pytree_from_json(maybe_serialized[len(prefix) :])
            return custom_type(**data)

    return maybe_serialized


def _deserialize_array(serialized_array: Any) -> Any:
    """Deserializes a numpy array from a string."""
    memfile = io.BytesIO()
    memfile.write(serialized_array.encode(NUMPY_ASCII_FORMAT))
    memfile.seek(0)
    return onp.load(memfile)


def _validate_prefixes(
    custom_types_and_prefixes: Tuple[Tuple[Any, str], ...],
) -> None:
    """Validates that prefixes are compatible."""
    prefixes = [prefix for _, prefix in custom_types_and_prefixes]
    prefixes.append(PREFIX_NUMPY)

    for i, test_prefix in enumerate(prefixes):
        if any(p.startswith(test_prefix) for p in prefixes[i + 1 :]):
            raise ValueError(f"Found conflicting prefixes, got {prefixes}")
