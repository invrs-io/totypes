"""Defines tests for the `totypes.types` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import optax
from jax import tree_util
from parameterized import parameterized

from totypes import symmetry, types


class BoundedArrayTest(unittest.TestCase):
    @parameterized.expand([[(1, 5, 4, 3)], [(5, 1, 2)]])
    def test_lower_bound_shape_validation(self, invalid_bound_shape):
        with self.assertRaisesRegex(
            ValueError, "`lower_bound` has shape incompatible "
        ):
            types.BoundedArray(
                array=jnp.ones((5, 4, 3)),
                lower_bound=jnp.ones(invalid_bound_shape),
                upper_bound=None,
            )

    @parameterized.expand([[(1, 5, 4, 3)], [(5, 1, 2)]])
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
                fixed_solid=onp.zeros((5, 2), dtype=bool),
                fixed_void=onp.zeros((5, 2), dtype=bool),
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
                fixed_solid=onp.zeros((5, 3), dtype=bool),
                fixed_void=onp.zeros((5, 2), dtype=bool),
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
                fixed_solid=onp.zeros((5, 2), dtype=bool),
                fixed_void=onp.zeros((5, 3), dtype=bool),
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
                fixed_solid=onp.zeros((5, 2), dtype=int),
                fixed_void=onp.zeros((5, 2), dtype=bool),
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
                fixed_solid=onp.zeros((5, 2), dtype=bool),
                fixed_void=onp.zeros((5, 2), dtype=int),
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
                fixed_solid=onp.ones((5, 2), dtype=bool),
                fixed_void=onp.ones((5, 2), dtype=bool),
                minimum_width=0,
                minimum_spacing=1,
            )

    @parameterized.expand([[0], [2.0]])
    def test_minimum_width_validation(self, invalid_min_width):
        with self.assertRaisesRegex(ValueError, "`minimum_width` must be a"):
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2), minimum_width=invalid_min_width
            )

    @parameterized.expand([[0], [2.0]])
    def test_minimum_spacing_validation(self, invalid_min_spacing):
        with self.assertRaisesRegex(ValueError, "`minimum_spacing` must be a"):
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2), minimum_spacing=invalid_min_spacing
            )

    @parameterized.expand([[(True, True, True)], [(False,)], [(1, False)]])
    def test_periodic_validation(self, invalid_periodic):
        with self.assertRaisesRegex(ValueError, "`periodic` must be length-2"):
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2), periodic=invalid_periodic
            )

    def test_symmetry_validation(self):
        with self.assertRaisesRegex(ValueError, "Found unrecognized symmetry:"):
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2), symmetries=("invalid_symmetry",)
            )

    @parameterized.expand([["rotation_90"], ["reflection_ne_sw"], ["reflection_nw_se"]])
    def test_symmetry_requiring_square_array(self, symmetry):
        with self.assertRaisesRegex(
            ValueError, "Some specified symmetries require a square"
        ):
            types.Density2DArray(
                array=jnp.arange(10).reshape(5, 2), symmetries=(symmetry,)
            )

    def test_flatten_unflatten_single_density(self):
        density = types.Density2DArray(
            array=jnp.arange(0, 10).reshape(2, 5),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=(onp.arange(0, 10).reshape(2, 5) < 3),
            fixed_void=(onp.arange(0, 10).reshape(2, 5) > 7),
            minimum_width=1,
            minimum_spacing=2,
        )
        leaves, treedef = jax.tree_util.tree_flatten(density)
        restored_density = jax.tree_util.tree_unflatten(treedef, leaves)
        onp.testing.assert_array_equal(density, restored_density)

    def test_broadcast_fixed_pixels(self):
        density = types.Density2DArray(
            array=jnp.arange(0, 30).reshape(3, 2, 5).astype(float),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=onp.ones((2, 5), dtype=bool),
            fixed_void=onp.zeros((2, 5), dtype=bool),
            minimum_width=1,
            minimum_spacing=2,
        )
        self.assertTrue(
            types._shapes_compatible(density.array.shape, density.fixed_solid.shape)
        )

    def test_jacobian(self):
        density = types.Density2DArray(
            array=jnp.arange(0, 10).reshape(2, 5).astype(float),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=None,
            fixed_void=None,
            minimum_width=1,
            minimum_spacing=2,
        )

        jac = jax.jacrev(lambda x: x)(density)
        (jac,) = tree_util.tree_leaves(jac)
        expected_jac = jnp.diag(jnp.ones(10)).reshape((2, 5, 2, 5))
        onp.testing.assert_array_equal(jac, expected_jac)

    def test_jacobian_with_fixed_pixels(self):
        density = types.Density2DArray(
            array=jnp.arange(0, 10).reshape(2, 5).astype(float),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=(onp.arange(0, 10).reshape(2, 5) < 3),
            fixed_void=(onp.arange(0, 10).reshape(2, 5) > 7),
            minimum_width=1,
            minimum_spacing=2,
        )

        jac = jax.jacrev(lambda x: x)(density)
        (jac,) = tree_util.tree_leaves(jac)
        expected_jac = jnp.diag(jnp.ones(10)).reshape((2, 5, 2, 5))
        onp.testing.assert_array_equal(jac, expected_jac)


class OptimzizeTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                types.BoundedArray(
                    array=jnp.ones((10, 10)),
                    lower_bound=-1.0,
                    upper_bound=1.0,
                )
            ],
            [
                types.Density2DArray(
                    array=jnp.ones((10, 10)),
                    fixed_solid=None,
                    fixed_void=None,
                )
            ],
            [
                types.Density2DArray(
                    array=jnp.ones((10, 10)),
                    fixed_solid=onp.zeros((10, 10), dtype=bool),
                    fixed_void=None,
                )
            ],
            [
                types.Density2DArray(
                    array=jnp.ones((10, 10)),
                    fixed_solid=onp.zeros((10, 10), dtype=bool),
                    fixed_void=onp.zeros((10, 10), dtype=bool),
                )
            ],
        ]
    )
    def test_optax_optimize(self, params):
        def loss_fn(params):
            return jnp.sum(jnp.abs(params.array))

        opt = optax.adam(0.1)
        state = opt.init(params)

        for _ in range(10):
            grad = jax.grad(loss_fn)(params)
            updates, state = opt.update(grad, state)
            params = optax.apply_updates(params, updates)


class ApplySymmetryTest(unittest.TestCase):
    def test_symmetrize_density(self):
        density = types.Density2DArray(
            array=jnp.asarray(
                [
                    [0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            symmetries=(symmetry.ROTATION_180,),
        )
        symmetrized = types.symmetrize_density(density)
        expected = jnp.asarray(
            [
                [0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
            ]
        )
        onp.testing.assert_array_equal(symmetrized.array, expected)
