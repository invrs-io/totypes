"""Utilities for partitioning pytrees including custom types.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Any, Callable, Tuple

from jax import tree_util

from totypes import types

PyTree = Any


def _is_custom_type_or_none(x: Any) -> bool:
    """Returns `True` if `x` is a recognized custom type."""
    return x is None or isinstance(x, types.CUSTOM_TYPES)


def partition(
    tree: PyTree,
    select_fn: Callable[[Any], bool],
    is_leaf: Callable[[Any], bool] = _is_custom_type_or_none,
) -> Tuple[PyTree, ...]:
    """"""
    selected = tree_util.tree_map(
        lambda x: x if select_fn(x) else None, tree, is_leaf=is_leaf
    )
    other = tree_util.tree_map(
        lambda x: None if select_fn(x) else x, tree, is_leaf=is_leaf
    )
    return selected, other


def combine(
    *trees: PyTree,
    is_leaf: Callable[[Any], bool] = _is_custom_type_or_none,
) -> PyTree:
    """Combines two or more pytrees obtained by `partition`."""
    return tree_util.tree_map(
        lambda *leaves: _first_not_none_from(*leaves), *trees, is_leaf=is_leaf
    )


def _first_not_none_from(*args: Any) -> Any:
    """Returns the first arg which is not `None`."""
    for arg in args:
        if arg is not None:
            break
    return arg
