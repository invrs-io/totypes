"""Custom datatypes supported by optimizers."""

import dataclasses
from typing import Any, Optional, Tuple, Union

import jax
import jax.core
import jax.numpy as jnp
import numpy as onp

Array = Union[jnp.ndarray, onp.ndarray]  # type: ignore
ArrayLike = Union[Array, float, int]
PyTree = Any


# -----------------------------------------------------------------------------
# Custom type for arrays with upper and lower bounds.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class BoundedArray:
    """Stores a quantity, along with optional declared lower and upper bounds.

    Attributes:
        value: A jax or numpy array, or a python scalar.
        lower_bound: The optional declared lower bound for `value`; must be
            broadcast compatible with `value`.
        upper_bound: The optional declared upper bound for `value`.
    """

    value: ArrayLike
    lower_bound: Optional[ArrayLike]
    upper_bound: Optional[ArrayLike]

    def __post_init__(self) -> None:
        # Attributes may be strings if they are serialized, or jax tracers
        # e.g. when computing gradients. Avoid validation in these cases.
        if not isinstance(self.value, (jnp.ndarray, onp.ndarray, int, float)):
            return

        if self.lower_bound is not None and (
            jnp.ndim(self.lower_bound) > jnp.ndim(self.value)
            or not _shapes_compatible(
                jnp.shape(self.value), jnp.shape(self.lower_bound)
            )
        ):
            raise ValueError(
                f"`lower_bound` has shape incompatible with `value`; got "
                f"shapes {jnp.shape(self.lower_bound)} and {jnp.shape(self.value)}."
            )
        if self.upper_bound is not None and (
            jnp.ndim(self.upper_bound) > jnp.ndim(self.value)
            or not _shapes_compatible(
                jnp.shape(self.value), jnp.shape(self.upper_bound)
            )
        ):
            raise ValueError(
                f"`upper_bound` has shape incompatible with `value`; got "
                f"shapes {jnp.shape(self.upper_bound)} and {jnp.shape(self.value)}."
            )

        if self.upper_bound is not None and self.lower_bound is not None:
            invalid_bounds = self.lower_bound >= self.upper_bound
            is_array = isinstance(invalid_bounds, (jnp.ndarray, onp.ndarray))
            if is_array and invalid_bounds.any() or not is_array and invalid_bounds:
                raise ValueError(
                    "`upper_bound` must be strictly greater than `lower_bound`."
                )


def _flatten_bounded_array(
    bounded_array: BoundedArray,
) -> Tuple[Tuple[Array], Tuple["_HashableWrapper", "_HashableWrapper"]]:
    """Flattens a `BoundedArray` into children and auxilliary data."""
    return (
        (bounded_array.value,),
        (
            _HashableWrapper(bounded_array.lower_bound),
            _HashableWrapper(bounded_array.upper_bound),
        ),
    )


def _unflatten_bounded_array(
    aux: Tuple["_HashableWrapper", "_HashableWrapper"],
    children: Tuple[Array],
) -> BoundedArray:
    """Unflattens a flattened `BoundedArray`."""
    (value,) = children
    wrapped_lower_bound, wrapped_upper_bound = aux
    return BoundedArray(
        value=value,
        lower_bound=wrapped_lower_bound.value,
        upper_bound=wrapped_upper_bound.value,
    )


jax.tree_util.register_pytree_node(
    BoundedArray,
    flatten_func=_flatten_bounded_array,
    unflatten_func=_unflatten_bounded_array,
)


# -----------------------------------------------------------------------------
# Custom type for two-dimensional density arrays.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Density2D:
    """Stores an array representing a 2D density.

    Intended constraints such as the bounds, and minimum width and spacing
    of features are specified by the attributes of a `Density`. Note that
    these are merely the intended constraints, and these must be recognized
    by an optimizer to ensure that densities meeting the constraints are
    obtained.

    Attributes:
        value: The value of the density, an array with at least rank 2.
        lower_bound: The numeric value associated with solid pixels.
        upper_bound: The numeric value associated with void pixels.
        fixed_solid: Optional array identifying pixels to be fixed solid.
        fixed_void: Optional array identifying pixels to be fixed void.
        minimum_width: The minimum width of solid features.
        minimum_spacing: The minimum spacing of solid features.
    """

    value: Array
    lower_bound: float
    upper_bound: float
    fixed_solid: Optional[Array]
    fixed_void: Optional[Array]
    minimum_width: int
    minimum_spacing: int

    def __post_init__(self) -> None:
        # Attributes may be strings if they are serialized, or jax tracers
        # e.g. when computing gradients. Avoid validation in these cases.
        if not isinstance(self.value, (jnp.ndarray, onp.ndarray)):
            return

        if self.value.ndim < 2:
            raise ValueError(
                f"`value` must be at least rank-2, but got shape {self.value.shape}"
            )

        if (
            not isinstance(self.lower_bound, (float, int))
            or not isinstance(self.upper_bound, (float, int))
            or self.upper_bound <= self.lower_bound
        ):
            raise ValueError(
                f"`lower_bound` and `upper_bound` must both be floats, with "
                f"`lower_bound` strictly less than `upper_bound` but got {self.lower_bound} "
                f"and {self.upper_bound}"
            )

        if self.fixed_solid is not None and self.fixed_solid.shape != self.value.shape:
            raise ValueError(
                f"`fixed_solid` must have shape matching `value`, but got shape "
                f"{self.fixed_solid.shape} when `value` has shape {self.value.shape}."
            )
        if self.fixed_void is not None and self.fixed_void.shape != self.value.shape:
            raise ValueError(
                f"`fixed_void` must have shape matching `value`, but got shape "
                f"{self.fixed_void.shape} when `value` has shape {self.value.shape}."
            )

        if self.fixed_solid is not None and self.fixed_solid.dtype != bool:
            raise ValueError(
                f"`fixed_solid` must be bool-typed but got {self.fixed_solid.dtype}."
            )
        if self.fixed_void is not None and self.fixed_void.dtype != bool:
            raise ValueError(
                f"`fixed_void` must be bool-typed but got {self.fixed_void.dtype}."
            )
        if (
            self.fixed_solid is not None
            and self.fixed_void is not None
            and (self.fixed_solid & self.fixed_void).any()
        ):
            raise ValueError(
                "Got incompatible `fixed_solid` and `fixed_void`; these must "
                "not be `True` at the same indices."
            )


def _flatten_zero_centered_density_2d(
    density: Density2D,
) -> Tuple[
    Tuple[Array], Tuple[float, float, "_HashableWrapper", "_HashableWrapper", int, int]
]:
    """Flattens a `Density2D` into children and auxilliary data."""
    return (
        (density.value,),
        (
            density.lower_bound,
            density.upper_bound,
            _HashableWrapper(density.fixed_solid),
            _HashableWrapper(density.fixed_void),
            density.minimum_width,
            density.minimum_spacing,
        ),
    )


def _unflatten_zero_centered_density_2d(
    aux: Tuple[float, float, "_HashableWrapper", "_HashableWrapper", int, int],
    children: Tuple[Array],
) -> Density2D:
    """Unflattens a flattened `Density2D`."""
    (value,) = children
    (
        lower_bound,
        upper_bound,
        wrapped_fixed_solid,
        wrapped_fixed_void,
        minimum_width,
        minimum_spacing,
    ) = aux
    return Density2D(
        value=value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        fixed_solid=wrapped_fixed_solid.value,
        fixed_void=wrapped_fixed_void.value,
        minimum_width=minimum_width,
        minimum_spacing=minimum_spacing,
    )


jax.tree_util.register_pytree_node(
    Density2D,
    flatten_func=_flatten_zero_centered_density_2d,
    unflatten_func=_unflatten_zero_centered_density_2d,
)


# -----------------------------------------------------------------------------
# Functions used by multiple custom types.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class _HashableWrapper:
    """Wraps arrays or scalars, making them hashable."""

    value: Optional[Union[onp.ndarray, jnp.ndarray, float]]

    def __hash__(self) -> int:
        if isinstance(self.value, (onp.ndarray, jnp.ndarray)):
            return hash((self.value.dtype, self.value.shape, self.value.tobytes()))
        return hash(self.value)

    def __eq__(self, other: "_HashableWrapper") -> bool:
        is_array_both = (
            isinstance(other.value, (onp.ndarray, jnp.ndarray)),
            isinstance(self.value, (onp.ndarray, jnp.ndarray)),
        )
        if is_array_both not in ((True, True), (False, False)):
            return False
        if all(is_array_both):
            return (
                self.value.shape == other.value.shape
                and self.value.dtype == other.value.dtype
                and (self.value == other.value).all()
            )
        return self.value == other.value


def _shapes_compatible(shape: Tuple[int, ...], other_shape: Tuple[int, ...]) -> bool:
    """Returns `True` if `shape` and `other_shape` are broadcast compatible."""
    matched_ndim = min(len(shape), len(other_shape))
    return all(
        ds == 1 or do == 1 or ds == do
        for ds, do in zip(shape[-matched_ndim:], other_shape[-matched_ndim:])
    )


def extract_lower_bound(params: PyTree) -> PyTree:
    """Extracts the lower bound for all leaves in `params`."""

    def _extract_fn(leaf):
        if isinstance(leaf, (BoundedArray, Density2D)):
            return leaf.lower_bound
        return None

    return jax.tree_util.tree_map(_extract_fn, params, is_leaf=_has_bounds)


def extract_upper_bound(params: PyTree) -> PyTree:
    """Extracts the upper bound for all leaves in `params`."""

    def _extract_fn(leaf):
        if isinstance(leaf, (BoundedArray, Density2D)):
            return leaf.upper_bound
        return None

    return jax.tree_util.tree_map(_extract_fn, params, is_leaf=_has_bounds)


def _has_bounds(x: Union[BoundedArray, Density2D, ArrayLike]) -> bool:
    """Returns `True` if `x` has bounds."""
    return isinstance(x, (BoundedArray, Density2D))
