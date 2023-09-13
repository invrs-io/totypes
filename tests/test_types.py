"""Defines tests for the `totypes` module."""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

import totypes


class BoundedArrayTest(unittest.TestCase):
    @parameterized.parameterized.expand([[(1, 5, 4, 3)], [(5, 1, 2)]])
    def test_lower_bound_shape_validation(self, invalid_bound_shape):
        with self.assertRaisesRegex(
            ValueError, "`lower_bound` has shape incompatible "
        ):
            totypes.BoundedArray(
                value=onp.ones((5, 4, 3)),
                lower_bound=onp.ones(invalid_bound_shape),
                upper_bound=None,
            )

    @parameterized.parameterized.expand([[(1, 5, 4, 3)], [(5, 1, 2)]])
    def test_upper_bound_shape_validation(self, invalid_bound_shape):
        with self.assertRaisesRegex(
            ValueError, "`upper_bound` has shape incompatible "
        ):
            totypes.BoundedArray(
                value=onp.ones((5, 4, 3)),
                lower_bound=None,
                upper_bound=onp.ones(invalid_bound_shape),
            )

    def test_flatten_unflatten_single_array(self):
        ba = totypes.BoundedArray(
            value=onp.arange(0, 10),
            lower_bound=onp.arange(-1, 9),
            upper_bound=onp.arange(1, 11),
        )
        leaves, treedef = jax.tree_util.tree_flatten(ba)
        restored_ba = jax.tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(ba, restored_ba)  # type: ignore


class Density2DTest(unittest.TestCase):
    def test_density_ndim_validation(self):
        with self.assertRaisesRegex(ValueError, "`value` must be at least rank-2,"):
            totypes.Density2D(
                value=jnp.arange(10),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.zeros((5, 2), dtype=bool),
                fixed_void=jnp.zeros((5, 2), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_fixed_solid_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`fixed_solid` must have shape matching `value`"
        ):
            totypes.Density2D(
                value=jnp.arange(10).reshape(5, 2),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.zeros((5, 3), dtype=bool),
                fixed_void=jnp.zeros((5, 2), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_fixed_void_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`fixed_void` must have shape matching `value`"
        ):
            totypes.Density2D(
                value=jnp.arange(10).reshape(5, 2),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.zeros((5, 2), dtype=bool),
                fixed_void=jnp.zeros((5, 3), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_fixed_solid_dtype_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`fixed_solid` must be bool-typed but got"
        ):
            totypes.Density2D(
                value=jnp.arange(10).reshape(5, 2),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.zeros((5, 2), dtype=int),
                fixed_void=jnp.zeros((5, 2), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_fixed_void_dtype_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`fixed_void` must be bool-typed but got"
        ):
            totypes.Density2D(
                value=jnp.arange(10).reshape(5, 2),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.zeros((5, 2), dtype=bool),
                fixed_void=jnp.zeros((5, 2), dtype=int),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_fixed_solid_fixed_void_compatible_validation(self):
        with self.assertRaisesRegex(
            ValueError, "Got incompatible `fixed_solid` and `fixed_void`"
        ):
            totypes.Density2D(
                value=jnp.arange(10).reshape(5, 2),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.ones((5, 2), dtype=bool),
                fixed_void=jnp.ones((5, 2), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_flatten_unflatten_single_density(self):
        density = totypes.Density2D(
            value=onp.arange(0, 10).reshape(2, 5),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=(onp.arange(0, 10).reshape(2, 5) < 3),
            fixed_void=(onp.arange(0, 10).reshape(2, 5) > 7),
            minimum_width=1,
            minimum_spacing=2,
        )
        leaves, treedef = jax.tree_util.tree_flatten(density)
        restored_density = jax.tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(density, restored_density)  # type: ignore
