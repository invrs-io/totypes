"""Defines tests for the `totypes.symmetry` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from totypes import symmetry, types

TEST_ARRAY = jnp.asarray(
    [
        [0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

TEST_BOUNDED_ARRAY = types.BoundedArray(
    array=TEST_ARRAY,
    lower_bound=None,
    upper_bound=None,
)

TEST_DENSITY_2D = types.Density2DArray(
    array=TEST_ARRAY,
)


def _assert_array_equal(x, expected_array):
    if isinstance(x, jnp.ndarray):
        onp.testing.assert_array_equal(x, expected_array)
    else:
        onp.testing.assert_array_equal(x.array, expected_array)


class SymmetryFunctionTest(unittest.TestCase):
    @parameterized.expand([[TEST_ARRAY], [TEST_BOUNDED_ARRAY], [TEST_DENSITY_2D]])
    def test_rotation_180(self, obj):
        result = symmetry.symmetrize(obj, (symmetry.ROTATION_180,))
        expected = jnp.asarray(
            [
                [0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
            ]
        )
        _assert_array_equal(result, expected)

    @parameterized.expand([[TEST_ARRAY], [TEST_BOUNDED_ARRAY], [TEST_DENSITY_2D]])
    def test_rotation_90(self, obj):
        result = symmetry.symmetrize(obj, (symmetry.ROTATION_90,))
        expected = jnp.asarray(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        )
        _assert_array_equal(result, expected)

    @parameterized.expand([[TEST_ARRAY], [TEST_BOUNDED_ARRAY], [TEST_DENSITY_2D]])
    def test_reflection_n_s(self, obj):
        result = symmetry.symmetrize(obj, (symmetry.REFLECTION_N_S,))
        expected = jnp.asarray(
            [
                [0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
            ]
        )
        _assert_array_equal(result, expected)

    @parameterized.expand([[TEST_ARRAY], [TEST_BOUNDED_ARRAY], [TEST_DENSITY_2D]])
    def test_reflection_e_w(self, obj):
        result = symmetry.symmetrize(obj, (symmetry.REFLECTION_E_W,))
        expected = jnp.asarray(
            [
                [0, 2, 0, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        _assert_array_equal(result, expected)

    @parameterized.expand([[TEST_ARRAY], [TEST_BOUNDED_ARRAY], [TEST_DENSITY_2D]])
    def test_reflection_ne_sw(self, obj):
        result = symmetry.symmetrize(obj, (symmetry.REFLECTION_NE_SW,))
        expected = jnp.asarray(
            [
                [0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0],
            ]
        )
        _assert_array_equal(result, expected)

    @parameterized.expand([[TEST_ARRAY], [TEST_BOUNDED_ARRAY], [TEST_DENSITY_2D]])
    def test_reflection_nw_se(self, obj):
        result = symmetry.symmetrize(obj, (symmetry.REFLECTION_NW_SE,))
        expected = jnp.asarray(
            [
                [0, 2, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        _assert_array_equal(result, expected)

    @parameterized.expand([[TEST_ARRAY], [TEST_BOUNDED_ARRAY], [TEST_DENSITY_2D]])
    def test_multiple_symmetry(self, obj):
        result = symmetry.symmetrize(
            obj,
            (
                symmetry.ROTATION_90,
                symmetry.ROTATION_180,
                symmetry.REFLECTION_E_W,
                symmetry.REFLECTION_N_S,
                symmetry.REFLECTION_NE_SW,
                symmetry.REFLECTION_NW_SE,
            ),
        )
        expected = jnp.asarray(
            [
                [0.0, 0.5, 0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.5, 0.0, 0.5, 0.0],
            ]
        )
        _assert_array_equal(result, expected)

    @parameterized.expand(
        [
            [symmetry.REFLECTION_E_W],
            [symmetry.REFLECTION_N_S],
            [symmetry.REFLECTION_NE_SW],
            [symmetry.REFLECTION_NW_SE],
            [symmetry.ROTATION_180],
            [symmetry.ROTATION_90],
        ]
    )
    def test_with_batch(self, sym):
        arr = jnp.ones((8, 13, 13))
        symmetry.symmetrize(arr, (sym,))
