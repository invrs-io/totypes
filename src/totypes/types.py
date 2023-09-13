"""Custom datatypes useful in an topology optimization setting."""

import dataclasses
from typing import Any, Optional, Tuple, Union

import jax
import jax.core
import jax.numpy as jnp
import numpy as onp

ArrayOrScalar = Union[jnp.ndarray, float, int]
PyTree = Any


# -----------------------------------------------------------------------------
# Custom type for arrays with upper and lower bounds.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class BoundedArray:
    """Stores an array, along with optional declared lower and upper bounds.

    Attributes:
        array: A jax array or a python scalar.
        lower_bound: The optional declared lower bound for `array`; must be
            broadcast compatible with `array`.
        upper_bound: The optional declared upper bound for `array`.
    """

    array: ArrayOrScalar
    lower_bound: Optional[ArrayOrScalar]
    upper_bound: Optional[ArrayOrScalar]

    def __post_init__(self) -> None:
        # Attributes may be strings if they are serialized, or jax tracers
        # e.g. when computing gradients. Avoid validation in these cases.
        if not isinstance(self.array, (jnp.ndarray, int, float)):
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
        """Returns the shape of the array."""
        return jnp.shape(self.array)  # type: ignore[no-any-return]


def _flatten_bounded_array(
    bounded_array: BoundedArray,
) -> Tuple[Tuple[ArrayOrScalar], Tuple["_HashableWrapper", "_HashableWrapper"]]:
    """Flattens a `BoundedArray` into children and auxilliary data."""
    return (
        (bounded_array.array,),
        (
            _HashableWrapper(bounded_array.lower_bound),
            _HashableWrapper(bounded_array.upper_bound),
        ),
    )


def _unflatten_bounded_array(
    aux: Tuple["_HashableWrapper", "_HashableWrapper"],
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


jax.tree_util.register_pytree_node(
    BoundedArray,
    flatten_func=_flatten_bounded_array,
    unflatten_func=_unflatten_bounded_array,
)


# -----------------------------------------------------------------------------
# Custom type for two-dimensional density arrays.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Density2DArray:
    """Stores an array representing a 2D density.

    Intended constraints such as the bounds, and minimum width and spacing
    of features are specified by the attributes of a `Density`. Note that
    these are merely the intended constraints, and these must be recognized
    by an optimizer to ensure that densities meeting the constraints are
    obtained.

    Attributes:
        array: The density array, with at least rank 2.
        lower_bound: The numeric value associated with solid pixels.
        upper_bound: The numeric value associated with void pixels.
        fixed_solid: Optional array identifying pixels to be fixed solid.
        fixed_void: Optional array identifying pixels to be fixed void.
        minimum_width: The minimum width of solid features.
        minimum_spacing: The minimum spacing of solid features.
    """

    array: jnp.ndarray
    lower_bound: float
    upper_bound: float
    fixed_solid: Optional[jnp.ndarray]
    fixed_void: Optional[jnp.ndarray]
    minimum_width: int
    minimum_spacing: int

    def __post_init__(self) -> None:
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

        if self.fixed_solid is not None and self.fixed_solid.shape != self.array.shape:
            raise ValueError(
                f"`fixed_solid` must have shape matching `array`, but got shape "
                f"{self.fixed_solid.shape} when `array` has shape {self.array.shape}."
            )
        if self.fixed_void is not None and self.fixed_void.shape != self.array.shape:
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
        if (
            self.fixed_solid is not None
            and self.fixed_void is not None
            and (self.fixed_solid & self.fixed_void).any()
        ):
            raise ValueError(
                "Got incompatible `fixed_solid` and `fixed_void`; these must "
                "not be `True` at the same indices."
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the array."""
        return jnp.shape(self.array)  # type: ignore[no-any-return]


def _flatten_density_2d(
    density: Density2DArray,
) -> Tuple[
    Tuple[jnp.ndarray],
    Tuple[float, float, "_HashableWrapper", "_HashableWrapper", int, int],
]:
    """Flattens a `Density2D` into children and auxilliary data."""
    return (
        (density.array,),
        (
            density.lower_bound,
            density.upper_bound,
            _HashableWrapper(density.fixed_solid),
            _HashableWrapper(density.fixed_void),
            density.minimum_width,
            density.minimum_spacing,
        ),
    )


def _unflatten_density_2d(
    aux: Tuple[float, float, "_HashableWrapper", "_HashableWrapper", int, int],
    children: Tuple[jnp.ndarray],
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
    ) = aux
    return Density2DArray(
        array=array,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        fixed_solid=wrapped_fixed_solid.array,
        fixed_void=wrapped_fixed_void.array,
        minimum_width=minimum_width,
        minimum_spacing=minimum_spacing,
    )


jax.tree_util.register_pytree_node(
    Density2DArray,
    flatten_func=_flatten_density_2d,
    unflatten_func=_unflatten_density_2d,
)


# -----------------------------------------------------------------------------
# Functions used by multiple custom types.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class _HashableWrapper:
    """Wraps arrays or scalars, making them hashable."""

    array: Optional[Union[onp.ndarray, jnp.ndarray, float]]

    def __hash__(self) -> int:
        if isinstance(self.array, (onp.ndarray, jnp.ndarray)):
            return hash((self.array.dtype, self.array.shape, self.array.tobytes()))
        return hash(self.array)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _HashableWrapper):
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
            return bool(
                self.array.shape == other.array.shape  # type: ignore[union-attr]
                and self.array.dtype == other.array.dtype  # type: ignore[union-attr]
                and (self.array == other.array).all()  # type: ignore[union-attr]
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

    return jax.tree_util.tree_map(_extract_fn, params, is_leaf=_has_bounds)


def extract_upper_bound(params: PyTree) -> PyTree:
    """Extracts the upper bound for all leaves in `params`."""

    def _extract_fn(leaf: ArrayOrScalar) -> Optional[ArrayOrScalar]:
        if isinstance(leaf, (BoundedArray, Density2DArray)):
            return leaf.upper_bound
        return None

    return jax.tree_util.tree_map(_extract_fn, params, is_leaf=_has_bounds)


def _has_bounds(x: Any) -> bool:
    """Returns `True` if `x` has bounds."""
    return isinstance(x, (BoundedArray, Density2DArray))
