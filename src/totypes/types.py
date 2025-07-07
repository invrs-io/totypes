"""Custom datatypes useful in an topology optimization setting.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util

from totypes import json_utils, symmetry

Array = Union[jnp.ndarray, onp.ndarray]
ArrayOrScalar = Union[Array, float, int]
PyTree = Any


# -----------------------------------------------------------------------------
# Custom type for arrays with upper and lower bounds.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class BoundedArray:
    """Stores an array, along with optional declared lower and upper bounds.

    Attributes:
        array: A jax or numpy array, or a python scalar.
        lower_bound: The optional declared lower bound for `array`; must be broadcast
            compatible with `array`.
        upper_bound: The optional declared upper bound for `array`.
    """

    array: ArrayOrScalar
    lower_bound: Optional[ArrayOrScalar]
    upper_bound: Optional[ArrayOrScalar]

    def __post_init__(self) -> None:
        # Attributes may be strings if they are serialized, or jax tracers
        # e.g. when computing gradients. Avoid validation in these cases.
        if not isinstance(self.array, (jnp.ndarray, onp.ndarray, int, float)):
            return

        if self.lower_bound is not None and (
            jnp.ndim(self.lower_bound) > jnp.ndim(self.array)
            or not _shapes_compatible(
                jnp.shape(self.array), jnp.shape(self.lower_bound)
            )
        ):
            raise ValueError(
                f"`lower_bound` has shape incompatible with `array`; got "
                f"shapes {jnp.shape(self.lower_bound)} and {jnp.shape(self.array)}."
            )
        if self.upper_bound is not None and (
            jnp.ndim(self.upper_bound) > jnp.ndim(self.array)
            or not _shapes_compatible(
                jnp.shape(self.array), jnp.shape(self.upper_bound)
            )
        ):
            raise ValueError(
                f"`upper_bound` has shape incompatible with `array`; got "
                f"shapes {jnp.shape(self.upper_bound)} and {jnp.shape(self.array)}."
            )

        if self.upper_bound is not None and self.lower_bound is not None:
            with jax.ensure_compile_time_eval():
                invalid_bounds = self.lower_bound >= self.upper_bound
                is_array = isinstance(invalid_bounds, (jnp.ndarray, onp.ndarray))
                if (
                    is_array
                    and invalid_bounds.any()  # type: ignore[union-attr]
                    or (not is_array and invalid_bounds)
                ):
                    raise ValueError(
                        "`upper_bound` must be strictly greater than `lower_bound`."
                    )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the array."""
        return jnp.shape(self.array)  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array."""
        return int(jnp.ndim(self.array))


def _flatten_bounded_array(
    bounded_array: BoundedArray,
) -> Tuple[Tuple[ArrayOrScalar], Tuple["HashableWrapper", "HashableWrapper"]]:
    """Flattens a `BoundedArray` into children and auxilliary data."""
    return (
        (bounded_array.array,),
        (
            HashableWrapper(bounded_array.lower_bound),
            HashableWrapper(bounded_array.upper_bound),
        ),
    )


def _unflatten_bounded_array(
    aux: Tuple["HashableWrapper", "HashableWrapper"],
    children: Tuple[ArrayOrScalar],
) -> BoundedArray:
    """Unflattens a flattened `BoundedArray`."""
    (array,) = children
    wrapped_lower_bound, wrapped_upper_bound = aux
    return BoundedArray(
        array=array,
        lower_bound=wrapped_lower_bound.array,
        upper_bound=wrapped_upper_bound.array,
    )


tree_util.register_pytree_node(
    BoundedArray,
    flatten_func=_flatten_bounded_array,
    unflatten_func=_unflatten_bounded_array,
)

json_utils.register_custom_type(BoundedArray)


# -----------------------------------------------------------------------------
# Custom type for two-dimensional density arrays.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Density2DArray:
    """Stores an array representing a 2D density.

    Intended constraints such as the bounds, and minimum width and spacing of features
    are specified by the attributes of a `Density`. Note that these are merely the
    intended constraints and characteristics, and these must be recognized and enforced
    by an optimizer to ensure that densities meeting the constraints are obtained.

    Attributes:
        array: A jax or numpy array representing density, with at least rank 2.
        lower_bound: The numeric value associated with solid pixels.
        upper_bound: The numeric value associated with void pixels.
        fixed_solid: Optional array identifying pixels to be fixed solid; must not be
            a jax array.
        fixed_void: Optional array identifying pixels to be fixed void.
        minimum_width: The minimum width of solid features.
        minimum_spacing: The minimum spacing of solid features.
        periodic: Specifies which of the two spatial dimensions should use periodic
            boundary conditions.
        symmetries: A sequence of strings specifying the symmetries of the array. Some
            Some symmetries require that the array have a square shape.
    """

    array: Array
    lower_bound: float = -1.0
    upper_bound: float = 1.0
    fixed_solid: Optional[onp.ndarray] = None
    fixed_void: Optional[onp.ndarray] = None
    minimum_width: int = 1
    minimum_spacing: int = 1
    periodic: Tuple[bool, bool] = (False, False)
    symmetries: Sequence[str] = ()

    def __post_init__(self) -> None:
        if len(self.periodic) != 2:
            raise ValueError(f"`periodic` must be length-2, but got {self.periodic}")

        self.periodic = (self.periodic[0], self.periodic[1])
        self.symmetries = tuple(self.symmetries)

        # Attributes may be strings if they are serialized, or jax tracers
        # e.g. when computing gradients. Avoid validation in these cases.
        if not isinstance(self.array, (jnp.ndarray, onp.ndarray)):
            return

        if self.array.ndim < 2:
            raise ValueError(
                f"`array` must be at least rank-2, but got shape {self.array.shape}"
            )
        if (
            not isinstance(self.lower_bound, (float, int))
            or not isinstance(self.upper_bound, (float, int))
            or self.upper_bound <= self.lower_bound
        ):
            raise ValueError(
                f"`lower_bound` and `upper_bound` must both be floats, with "
                f"`lower_bound` strictly less than `upper_bound` but got "
                f"{self.lower_bound} and {self.upper_bound}"
            )

        if self.fixed_solid is not None and not isinstance(
            self.fixed_solid, onp.ndarray
        ):
            raise ValueError(
                f"`fixed_solid` must be `None` or a numpy array, but got "
                f"{type(self.fixed_solid)}"
            )
        if self.fixed_solid is not None and (
            jnp.ndim(self.fixed_solid) > jnp.ndim(self.array)
            or self.fixed_solid.shape[-2:] != self.array.shape[-2:]
            or not _shapes_compatible(
                jnp.shape(self.array), jnp.shape(self.fixed_solid)
            )
        ):
            raise ValueError(
                f"`fixed_solid` must have shape matching `array`, but got shape "
                f"{self.fixed_solid.shape} when `array` has shape {self.array.shape}."
            )

        if self.fixed_void is not None and not isinstance(self.fixed_void, onp.ndarray):
            raise ValueError(
                f"`fixed_void` must be `None` or a numpy array, but got "
                f"{type(self.fixed_void)}"
            )
        if self.fixed_void is not None and (
            jnp.ndim(self.fixed_void) > jnp.ndim(self.array)
            or self.fixed_void.shape[-2:] != self.array.shape[-2:]
            or not _shapes_compatible(jnp.shape(self.array), jnp.shape(self.fixed_void))
        ):
            raise ValueError(
                f"`fixed_void` must have shape matching `array`, but got shape "
                f"{self.fixed_void.shape} when `array` has shape {self.array.shape}."
            )

        if self.fixed_solid is not None and self.fixed_solid.dtype != bool:
            raise ValueError(
                f"`fixed_solid` must be bool-typed but got {self.fixed_solid.dtype}."
            )
        if self.fixed_void is not None and self.fixed_void.dtype != bool:
            raise ValueError(
                f"`fixed_void` must be bool-typed but got {self.fixed_void.dtype}."
            )

        with jax.ensure_compile_time_eval():
            if (
                self.fixed_solid is not None
                and self.fixed_void is not None
                and (self.fixed_solid & self.fixed_void).any()
            ):
                raise ValueError(
                    "Got incompatible `fixed_solid` and `fixed_void`; these must "
                    "not be `True` at the same indices."
                )

        if not isinstance(self.minimum_width, int) or self.minimum_width < 1:
            raise ValueError(
                f"`minimum_width` must be a postive int, but got {self.minimum_width}"
            )
        if not isinstance(self.minimum_spacing, int) or self.minimum_spacing < 1:
            raise ValueError(
                f"`minimum_spacing` must be a postive int, but got "
                f"{self.minimum_spacing}"
            )
        if len(self.periodic) != 2 or any(
            not isinstance(p, bool) for p in self.periodic
        ):
            raise ValueError(
                f"`periodic` must be length-2 sequence of `bool` but got "
                f"{self.periodic}."
            )
        if not all(s in symmetry.SYMMETRY_FNS for s in self.symmetries):
            raise ValueError(f"Found unrecognized symmetry: {self.symmetries}.")
        if (self.array.shape[-2] != self.array.shape[-1]) and any(
            s in symmetry.SYMMETRIES_REQUIRING_SQUARE_ARRAYS for s in self.symmetries
        ):
            raise ValueError(
                f"Some specified symmetries require a square array shape, but got a "
                f"shape of {self.array.shape} for symmetries {self.symmetries}."
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the array."""
        return jnp.shape(self.array)  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array."""
        return int(jnp.ndim(self.array))


def _flatten_density_2d(
    density: Density2DArray,
) -> Tuple[
    Tuple[Array],
    Tuple[
        float,
        float,
        "HashableWrapper",
        "HashableWrapper",
        int,
        int,
        Sequence[bool],
        Sequence[str],
    ],
]:
    """Flattens a `Density2D` into children and auxilliary data."""
    return (
        (density.array,),
        (
            density.lower_bound,
            density.upper_bound,
            HashableWrapper(density.fixed_solid),
            HashableWrapper(density.fixed_void),
            density.minimum_width,
            density.minimum_spacing,
            density.periodic,
            density.symmetries,
        ),
    )


def _unflatten_density_2d(
    aux: Tuple[
        float,
        float,
        "HashableWrapper",
        "HashableWrapper",
        int,
        int,
        Sequence[bool],
        Sequence[str],
    ],
    children: Tuple[Array],
) -> Density2DArray:
    """Unflattens a flattened `Density2D`."""
    (array,) = children
    (
        lower_bound,
        upper_bound,
        wrapped_fixed_solid,
        wrapped_fixed_void,
        minimum_width,
        minimum_spacing,
        periodic,
        symmetries,
    ) = aux
    return Density2DArray(
        array=array,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        fixed_solid=wrapped_fixed_solid.array,
        fixed_void=wrapped_fixed_void.array,
        minimum_width=minimum_width,
        minimum_spacing=minimum_spacing,
        periodic=tuple(periodic),  # type: ignore[arg-type]
        symmetries=tuple(symmetries),
    )


tree_util.register_pytree_node(
    Density2DArray,
    flatten_func=_flatten_density_2d,
    unflatten_func=_unflatten_density_2d,
)


json_utils.register_custom_type(Density2DArray)


def symmetrize_density(density: Density2DArray) -> Density2DArray:
    """Return a `density` with array having the specified `symmetries`."""
    symmetrized: Density2DArray = symmetry.symmetrize(
        density, tuple(density.symmetries)
    )
    return symmetrized


# -----------------------------------------------------------------------------
# Functions used by multiple custom types.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class HashableWrapper:
    """Wraps arrays or scalars, making them hashable."""

    array: Optional[ArrayOrScalar]

    def __hash__(self) -> int:
        if isinstance(self.array, (onp.ndarray, jnp.ndarray)):
            return hash((self.array.dtype, self.array.shape, self.array.tobytes()))
        return hash(self.array)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, HashableWrapper):
            raise NotImplementedError(
                f"Comparison with {type(other)} is not implemented."
            )
        is_array_both = (
            isinstance(other.array, (onp.ndarray, jnp.ndarray)),
            isinstance(self.array, (onp.ndarray, jnp.ndarray)),
        )
        if is_array_both not in ((True, True), (False, False)):
            return False
        if all(is_array_both):
            _array: Array = self.array
            _other_array: Array = other.array
            with jax.ensure_compile_time_eval():
                return bool(
                    _array.shape == _other_array.shape
                    and _array.dtype == _other_array.dtype
                    and (_array == _other_array).all()
                )
        return self.array == other.array


def _shapes_compatible(shape: Tuple[int, ...], other_shape: Tuple[int, ...]) -> bool:
    """Returns `True` if `shape` and `other_shape` are broadcast compatible."""
    matched_ndim = min(len(shape), len(other_shape))
    return all(
        ds == 1 or do == 1 or ds == do
        for ds, do in zip(shape[-matched_ndim:], other_shape[-matched_ndim:])
    )


def extract_lower_bound(params: PyTree) -> PyTree:
    """Extracts the lower bound for all leaves in `params`."""

    def _extract_fn(leaf: ArrayOrScalar) -> Optional[ArrayOrScalar]:
        if isinstance(leaf, (BoundedArray, Density2DArray)):
            return leaf.lower_bound
        return None

    return tree_util.tree_map(_extract_fn, params, is_leaf=_has_bounds)


def extract_upper_bound(params: PyTree) -> PyTree:
    """Extracts the upper bound for all leaves in `params`."""

    def _extract_fn(leaf: ArrayOrScalar) -> Optional[ArrayOrScalar]:
        if isinstance(leaf, (BoundedArray, Density2DArray)):
            return leaf.upper_bound
        return None

    return tree_util.tree_map(_extract_fn, params, is_leaf=_has_bounds)


def _has_bounds(x: Any) -> bool:
    """Returns `True` if `x` has bounds."""
    return isinstance(x, (BoundedArray, Density2DArray))


# Declare the custom types defined in this module.
CUSTOM_TYPES = (
    BoundedArray,
    Density2DArray,
)
