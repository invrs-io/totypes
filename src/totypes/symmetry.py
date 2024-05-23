"""Defines symmetries for arrays.

Copyright (c) 2023 The INVRS-IO authors.
"""

from typing import Any, Tuple

import jax.numpy as jnp
from jax import tree_util

REFLECTION_NE_SW = "reflection_ne_sw"
REFLECTION_NW_SE = "reflection_nw_se"
REFLECTION_N_S = "reflection_n_s"
REFLECTION_E_W = "reflection_e_w"
ROTATION_180 = "rotation_180"
ROTATION_90 = "rotation_90"


def symmetrize(
    custom_type: Any,
    symmetries: Tuple[str, ...],
) -> Any:
    """Apply specified symmetries to a custom type with a single leaf node."""
    if not isinstance(symmetries, tuple):
        raise ValueError(
            f"`symmetries` must be a tuple of `str`, but got {symmetries}."
        )
    (array,), treedef = tree_util.tree_flatten(custom_type)
    for symmetry in symmetries:
        if symmetry not in SYMMETRY_FNS:
            raise ValueError(f"Unrecognized symmetry: {symmetry}")
        array = SYMMETRY_FNS[symmetry](array)
    return tree_util.tree_unflatten(treedef, (array,))


def _reflection_ne_sw(array: jnp.ndarray) -> jnp.ndarray:
    """Transform `array` to have reflection symmetry about the ne-sw axis."""
    assert array.shape[-2] == array.shape[-1]
    return (array + jnp.swapaxes(array[..., ::-1, ::-1], -2, -1)) / 2


def _reflection_nw_se(array: jnp.ndarray) -> jnp.ndarray:
    """Transform `array` to have reflection symmetry about the nw-se axis."""
    assert array.shape[-2] == array.shape[-1]
    return (array + jnp.swapaxes(array, -2, -1)) / 2


def _reflection_n_s(array: jnp.ndarray) -> jnp.ndarray:
    """Transform `array` to have reflection symmetry about the n-s axis."""
    return (array + array[..., ::-1, :]) / 2


def _reflection_e_w(array: jnp.ndarray) -> jnp.ndarray:
    """Transform `array` to have reflection symmetry about the e-w axis."""
    return (array + array[..., ::-1]) / 2


def _rotation_180(array: jnp.ndarray) -> jnp.ndarray:
    """Transform `array` to have 180-degree rotational symmetry."""
    return (array + jnp.rot90(array, 2, axes=(-2, -1))) / 2


def _rotation_90(array: jnp.ndarray) -> jnp.ndarray:
    """Transform `array` to have 90-degree rotational symmetry."""
    assert array.shape[-2] == array.shape[-1]
    return (
        array
        + jnp.rot90(array, 1, axes=(-2, -1))
        + jnp.rot90(array, 2, axes=(-2, -1))
        + jnp.rot90(array, 3, axes=(-2, -1))
    ) / 4


SYMMETRY_FNS = {
    REFLECTION_NE_SW: _reflection_ne_sw,
    REFLECTION_NW_SE: _reflection_nw_se,
    REFLECTION_N_S: _reflection_n_s,
    REFLECTION_E_W: _reflection_e_w,
    ROTATION_180: _rotation_180,
    ROTATION_90: _rotation_90,
}

SYMMETRIES_REQUIRING_SQUARE_ARRAYS = (
    REFLECTION_NE_SW,
    REFLECTION_NW_SE,
    ROTATION_90,
)
