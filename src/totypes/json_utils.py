"""Utility functions related to json packing and unpacking.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import functools
import io
import json
from typing import Any, Dict, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as onp
from jax import tree_util

PyTree = Any

_NUMPY_ASCII_FORMAT = "latin-1"
_PREFIX_NUMPY = "\x93NUMPY"
_CUSTOM_TYPE_REGISTRY: Dict[str, Any] = {}


def register_custom_type(custom_type: Any) -> None:
    """Register a custom type so that it can be serialized and deserialized.

    To support serialization and deserialization, custom types must be either
    dataclasses or namedtuples and their attributes must be numpy or jax arrays
    or have types that are json-serializable by default (e.g. strings).

    Args:
        custom_type: The custom type to be serialized.
    """
    if not (dataclasses.is_dataclass(custom_type) or _is_namedtuple(custom_type)):
        raise ValueError(
            f"Only dataclasses and namedtuples are supported, but got {custom_type}."
        )
    if custom_type in _CUSTOM_TYPE_REGISTRY.values():
        raise ValueError(f"Duplicate custom type registration for {custom_type}.")
    prefix = _prefix_for_custom_type(custom_type)
    _validate_prefix(prefix, list(_CUSTOM_TYPE_REGISTRY.keys()))
    _CUSTOM_TYPE_REGISTRY[prefix] = custom_type


def _prefix_for_custom_type(custom_type: Any) -> str:
    """Return the prefix for a custom type."""
    type_str = str(custom_type)
    type_hash = hash(type_str)
    type_hash_str = f"{'p' if type_hash > 0 else 'n'}{abs(type_hash)}"
    return f"\x93TYPES.{type_hash_str}.{type_str}."


def _validate_prefix(test_prefix: str, existing_prefixes: Sequence[str]) -> None:
    """Validate that `test_prefix` does not conflict with `existing_prefixes."""
    for p in existing_prefixes:
        if (
            p.startswith(_PREFIX_NUMPY)
            or p.startswith(test_prefix)
            or test_prefix.startswith(p)
        ):
            raise ValueError(
                f"Prefix {test_prefix} confilcts with existing {existing_prefixes}."
            )


def _validate_prefixes(prefixes: Sequence[str]) -> None:
    """Validates that prefixes are not in conflict."""
    for i, p in enumerate(prefixes):
        _validate_prefix(p, prefixes[i + 1 :])


def _is_namedtuple(custom_type: Any) -> bool:
    """Return `True` if `custom_type` is a `namedtuple`."""
    return hasattr(custom_type, "_asdict")


# -----------------------------------------------------------------------------
# Serialization functions.
# -----------------------------------------------------------------------------


def json_from_pytree(
    pytree: PyTree,
    extra_custom_types_and_prefixes: Tuple[Tuple[Any, str], ...] = (),
) -> str:
    """Serializes a pytree containing arrays into a json string.

    Extra custom types and prefixes can be manually specified so as to allow user-
    defined objects to be serialized, so long as they are dataclasses or namedtuples,
    and have leaves that are jax or numpy arrays, or are types that are natively
    json-serializable (e.g. strings).

    In addition to manually specifying custom types, custom types can be registered
    using the `register_custom_type` function.

    Args:
        pytree: The pytree to be serialized.
        extra_custom_types_and_prefixes: The additional custom types and prefixes.
            Note that any manually-specified custom types may override registered
            custom types.

    Returns:
        The serialized pytree.
    """
    custom_types_and_prefixes = tuple(
        set(
            tuple([(t, p) for p, t in _CUSTOM_TYPE_REGISTRY.items()])
            + extra_custom_types_and_prefixes
        )
    )
    custom_types, prefixes = zip(*custom_types_and_prefixes)
    _validate_prefixes(prefixes)
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
    return memfile.getvalue().decode(_NUMPY_ASCII_FORMAT)


def pytree_from_json(
    serialized: str,
    extra_custom_types_and_prefixes: Tuple[Tuple[Any, str], ...] = (),
) -> PyTree:
    """Deserializes a json string into a pytree of arrays.

    Extra custom types and prefixes can be manually specified so as to allow user-
    defined objects to be serialized, so long as they are dataclasses or namedtuples,
    and have leaves that are jax or numpy arrays, or are types that are natively
    json-serializable (e.g. strings).

    In addition to manually specifying custom types, custom types can be registered
    using the `register_custom_type` function.

    Args:
        serialized: The serialized pytree.
        extra_custom_types_and_prefixes: The additional custom types and prefixes.
            Note that any manually-specified custom types may override registered
            custom types.

    Returns:
        The deserialized pytree.
    """
    custom_types_and_prefixes = tuple(
        set(
            tuple([(t, p) for p, t in _CUSTOM_TYPE_REGISTRY.items()])
            + extra_custom_types_and_prefixes
        )
    )
    _, prefixes = zip(*custom_types_and_prefixes)
    _validate_prefixes(prefixes)
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

    if maybe_serialized.startswith(_PREFIX_NUMPY):
        return _deserialize_array(maybe_serialized)
    for custom_type, prefix in custom_types_and_prefixes:
        if maybe_serialized.startswith(prefix):
            data = pytree_from_json(maybe_serialized[len(prefix) :])
            return custom_type(**data)

    return maybe_serialized


def _deserialize_array(serialized_array: Any) -> Any:
    """Deserializes a numpy array from a string."""
    memfile = io.BytesIO()
    memfile.write(serialized_array.encode(_NUMPY_ASCII_FORMAT))
    memfile.seek(0)
    return onp.load(memfile)
