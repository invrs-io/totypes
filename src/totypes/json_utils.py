"""Utility functions related to json packing and unpacking.

Copyright (c) 2023 The INVRS-IO authors.
"""

import base64
import dataclasses
import json
import warnings
from typing import Any, Dict, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as onp
from jax import tree_util

PyTree = Any


_CUSTOM_TYPE_REGISTRY: Dict[str, Any] = {}

_PREFIX_ARRAY = "\x93TYPES.ARRAY"
_PREFIX_COMPLEX = "\x93TYPES.COMPLEX"


def register_custom_type(custom_type: Any) -> None:
    """Register a custom type so that it can be serialized and deserialized.

    To support serialization and deserialization, custom types must be either
    dataclasses or namedtuples and their attributes must be numpy or jax arrays
    or have types that are json-serializable by default (e.g. strings).

    Args:
        custom_type: The custom type to be serialized.
    """
    if type(custom_type) is not type:
        raise ValueError(f"`custom_type` must be a type, but got {type(custom_type)}.")
    if not (dataclasses.is_dataclass(custom_type) or _is_namedtuple(custom_type)):
        raise ValueError(
            f"Only dataclasses and namedtuples are supported, but got {custom_type}."
        )
    if custom_type in _CUSTOM_TYPE_REGISTRY.values():
        warnings.warn(f"Duplicate custom type registration for {custom_type}.")
    prefix = _prefix_for_custom_type(custom_type)
    _validate_prefix(prefix, list(_CUSTOM_TYPE_REGISTRY.keys()))
    _CUSTOM_TYPE_REGISTRY[prefix] = custom_type


def _prefix_for_custom_type(custom_type: Any) -> str:
    """Return the prefix for a custom type."""
    return f"\x93TOTYPES.REGISTERED_CUSTOM_TYPE.{str(custom_type)}"


def _validate_prefix(test_prefix: str, existing_prefixes: Sequence[str]) -> None:
    """Validate that `test_prefix` does not conflict with `existing_prefixes."""
    for p in existing_prefixes:
        if (
            p.startswith(_PREFIX_ARRAY)
            or p.startswith(_PREFIX_COMPLEX)
            or (p.startswith(test_prefix) and p != test_prefix)
            or (test_prefix.startswith(p) and p != test_prefix)
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
    **dumps_kwargs: Any,
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
        **dumps_kwargs: Keyword arguments passed to `json.dumps`.

    Returns:
        The serialized pytree.
    """
    custom_types_and_prefixes = tuple(
        set(
            tuple([(t, p) for p, t in _CUSTOM_TYPE_REGISTRY.items()])
            + extra_custom_types_and_prefixes
        )
    )
    pytree_with_serialized = _prepare_for_json_serialization(
        pytree, custom_types_and_prefixes
    )
    return json.dumps(pytree_with_serialized, **dumps_kwargs)


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
    converted = json.loads(serialized)
    return _restore_pytree(converted, custom_types_and_prefixes)


# -----------------------------------------------------------------------------
# Functions for converting pytrees to prepare for json-serialization.
# -----------------------------------------------------------------------------


def _prepare_for_json_serialization(
    pytree: Any,
    custom_types_and_prefixes: Sequence[Tuple[Any, str]],
) -> Any:
    """Convert `pytree` so that it can be json-serialized.

    Args:
        pytree: The pytree to be prepared for serialization.
        custom_types_and_prefixes: Custom types and corresponding prefixes. Any
            instances of a custom type in `pytree` are converted to dictionaries.

    Returns:
        The converted pytree, suitable for json-serialization.
    """
    # Dictionary with keys that are custom types, and values that are prefixes.
    _, prefixes = zip(*custom_types_and_prefixes)
    _validate_prefixes(prefixes)
    custom_type_dict = _custom_type_dict(custom_types_and_prefixes)
    custom_types = tuple(custom_type_dict.keys())

    def convert_fn(obj: Any) -> Any:
        if isinstance(obj, (onp.ndarray, jnp.ndarray)):
            return _convert_array(obj)
        if isinstance(obj, (complex)):
            return _convert_complex(obj)
        if isinstance(obj, custom_types):
            return (
                custom_type_dict[type(obj)],  # Value is the prefix.
                _prepare_for_json_serialization(
                    _asdict(obj), custom_types_and_prefixes
                ),
            )
        return obj

    return tree_util.tree_map(
        convert_fn, pytree, is_leaf=lambda x: isinstance(x, custom_types)
    )


def _custom_type_dict(
    custom_types_and_prefixes: Sequence[Tuple[Any, str]]
) -> Dict[Any, str]:
    """Dictionary that maps custom types to their prefixes."""
    custom_type_dict = {}
    for custom_type, prefix in custom_types_and_prefixes:
        custom_type_dict[custom_type] = prefix
    return custom_type_dict


def _convert_array(arr: Union[onp.ndarray, jnp.ndarray]) -> Tuple[str, Dict[str, Any]]:
    """Converts a numpy or jax array so that it can be json-serialized."""
    assert isinstance(arr, (onp.ndarray, jnp.ndarray))
    return (
        _PREFIX_ARRAY,
        {
            # Keys match the argument names of `_restore_array`.
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "bytes": base64.b64encode(arr.tobytes()).decode("ASCII"),
        },
    )


def _convert_complex(val: complex) -> Tuple[str, Dict[str, float]]:
    """Converts a complex so that it can be json serialized."""
    assert isinstance(val, complex)
    # Keys match the argument names of `_restore_complex`.
    return (_PREFIX_COMPLEX, {"real": val.real, "imag": val.imag})


def _asdict(x: Any) -> Dict[str, Any]:
    """Converts dataclasses or namedtuples to dictionaries."""
    if dataclasses.is_dataclass(x):
        return {field.name: getattr(x, field.name) for field in dataclasses.fields(x)}
    try:
        return x._asdict()  # type: ignore[no-any-return]
    except AttributeError as exc:
        raise ValueError(
            f"`x` must be a dataclass or a namedtuple, but got {type(x)}"
        ) from exc


# -----------------------------------------------------------------------------
# Functions for undoing the conversion required for json-serialization.
# -----------------------------------------------------------------------------


def _restore_pytree(
    pytree: Any,
    custom_types_and_prefixes: Sequence[Tuple[Any, str]],
) -> Any:
    """Restores a pytree array, undoing the conversion needed for json-serialization.

    This function effectively undoes a `_prepare_for_json_serialization` operation.

    Args:
        pytree: The pytree to be restored.
        custom_types_and_prefixes: Custom types and corresponding prefixes. Any
            instances of a custom type in `pytree` are restored.

    Returns:
        The converted pytree, suitable for json-serialization."""
    _, prefixes = zip(*custom_types_and_prefixes)
    _validate_prefixes(prefixes)
    prefix_dict = _prefix_dict(custom_types_and_prefixes)
    prefixes = tuple(prefix_dict.keys())

    def restore_fn(obj: Any) -> Any:
        if _is_array_leaf(obj):
            _, data = obj
            return _restore_array(**data)
        if _is_complex_leaf(obj):
            _, data = obj
            return _restore_complex(**data)
        if _is_custom_leaf(obj, prefixes):
            prefix, data = obj
            return prefix_dict[prefix](
                **_restore_pytree(data, custom_types_and_prefixes)
            )
        return obj

    return tree_util.tree_map(
        restore_fn,
        pytree,
        is_leaf=lambda x: _is_array_leaf(x)
        or _is_custom_leaf(x, prefixes)
        or _is_complex_leaf(x),
    )


def _prefix_dict(
    custom_types_and_prefixes: Sequence[Tuple[Any, str]]
) -> Dict[str, Any]:
    """Dictionary that maps prefixes to the corresponding custom type."""
    prefix_dict = {}
    for custom_type, prefix in custom_types_and_prefixes:
        prefix_dict[prefix] = custom_type
    return prefix_dict


def _is_array_leaf(obj: Any) -> bool:
    """Return `True` if `obj` is a converted array leaf."""
    return isinstance(obj, (tuple, list)) and len(obj) == 2 and obj[0] == _PREFIX_ARRAY


def _is_complex_leaf(obj: Any) -> bool:
    """Return `True` if `obj` is a converted complex leaf."""
    return (
        isinstance(obj, (tuple, list)) and len(obj) == 2 and obj[0] == _PREFIX_COMPLEX
    )


def _is_custom_leaf(obj: Any, prefixes: Sequence[str]) -> bool:
    """Return `True` if `obj` is a converted custom leaf."""
    return isinstance(obj, (tuple, list)) and len(obj) == 2 and obj[0] in prefixes


def _restore_array(shape: Tuple[int, ...], dtype: str, bytes: str) -> onp.ndarray:
    """Restores an array from its serialized attributes."""
    array = onp.frombuffer(base64.b64decode(bytes), dtype=dtype)
    return array.reshape(shape)


def _restore_complex(real: float, imag: float) -> complex:
    """Restores a complex from its serialized attributes."""
    return real + 1j * imag
