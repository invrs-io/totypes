"""Defines tests for the `totypes.types` module."""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from totypes import types


class BoundedArrayTest(unittest.TestCase):
    @parameterized.parameterized.expand([[(1, 5, 4, 3)], [(5, 1, 2)]])
    def test_lower_bound_shape_validation(self, invalid_bound_shape):
        with self.assertRaisesRegex(
            ValueError, "`lower_bound` has shape incompatible "
        ):
            types.BoundedArray(
                array=jnp.ones((5, 4, 3)),
                lower_bound=jnp.ones(invalid_bound_shape),
                upper_bound=None,
            )

    @parameterized.parameterized.expand([[(1, 5, 4, 3)], [(5, 1, 2)]])
    def test_upper_bound_shape_validation(self, invalid_bound_shape):
        with self.assertRaisesRegex(
            ValueError, "`upper_bound` has shape incompatible "
        ):
            types.BoundedArray(
                array=jnp.ones((5, 4, 3)),
                lower_bound=None,
                upper_bound=jnp.ones(invalid_bound_shape),
            )

    def test_flatten_unflatten_single_array_jax(self):
        ba = types.BoundedArray(
            array=jnp.arange(0, 10),
            lower_bound=jnp.arange(-1, 9),
            upper_bound=jnp.arange(1, 11),
        )
        leaves, treedef = jax.tree_util.tree_flatten(ba)
        restored_ba = jax.tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(ba, restored_ba)


class Density2DArrayTest(unittest.TestCase):
    def test_density_ndim_validation(self):
        with self.assertRaisesRegex(ValueError, "`array` must be at least rank-2,"):
            types.Density2DArray(
                array=jnp.arange(10),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.zeros((5, 2), dtype=bool),
                fixed_void=jnp.zeros((5, 2), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_fixed_solid_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`fixed_solid` must have shape matching `array`"
        ):
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.zeros((5, 3), dtype=bool),
                fixed_void=jnp.zeros((5, 2), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_fixed_void_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`fixed_void` must have shape matching `array`"
        ):
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2),
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
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2),
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
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2),
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
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2),
                lower_bound=-1.0,
                upper_bound=1.0,
                fixed_solid=jnp.ones((5, 2), dtype=bool),
                fixed_void=jnp.ones((5, 2), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    def test_flatten_unflatten_single_density(self):
        density = types.Density2DArray(
            array=jnp.arange(0, 10).reshape(2, 5),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=(jnp.arange(0, 10).reshape(2, 5) < 3),
            fixed_void=(jnp.arange(0, 10).reshape(2, 5) > 7),
            minimum_width=1,
            minimum_spacing=2,
        )
        leaves, treedef = jax.tree_util.tree_flatten(density)
        restored_density = jax.tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(density, restored_density)
