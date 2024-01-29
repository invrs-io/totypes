"""Defines tests for the `totypes.types` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import itertools
import unittest

import jax
import numpy as onp
from jax import tree_util
from parameterized import parameterized

from totypes import partition_utils, types

TREES = [
    [1, 2, 3],
    [1, 2, 3, None],
    [types.BoundedArray(array=0.1, lower_bound=0, upper_bound=2)],
    [onp.ones((3, 3)), None],
    [types.Density2DArray(array=onp.zeros((2, 2)))],
    {
        "a": types.BoundedArray(array=0.1, lower_bound=0, upper_bound=2),
        "b": {
            "c": types.Density2DArray(array=onp.zeros((2, 2))),
            "d": None,
            "e": {"f": 1, "g": 0, "h": 3.14159},
        },
    },
]


SELECT_FNS = [
    lambda x: isinstance(x, int),
    lambda x: isinstance(x, types.BoundedArray),
    lambda x: isinstance(x, types.Density2DArray),
    lambda x: not isinstance(x, types.Density2DArray),
]


def _is_custom_type_or_none(x):
    return x is None or isinstance(x, types.CUSTOM_TYPES)


class PartitionTest(unittest.TestCase):
    @parameterized.expand(
        [
            [t, selected_type]
            for t, selected_type in itertools.product(
                TREES, [int, types.BoundedArray, types.Density2DArray]
            )
        ]
    )
    def test_select(self, tree, selected_type):
        a, b = partition_utils.partition(tree, lambda x: isinstance(x, selected_type))
        a_leaves = tree_util.tree_leaves(
            a, is_leaf=lambda x: isinstance(x, selected_type)
        )
        b_leaves = tree_util.tree_leaves(
            b, is_leaf=lambda x: isinstance(x, selected_type)
        )
        self.assertTrue(
            all([isinstance(x, selected_type) or x is None for x in a_leaves])
        )
        self.assertFalse(
            any([isinstance(x, selected_type) or x is None for x in b_leaves])
        )

    @parameterized.expand([[t] for t in TREES])
    def test_partition_with_always_select(self, tree):
        a, b = partition_utils.partition(tree, lambda x: True)
        self.assertEqual(
            tree_util.tree_structure(a, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        self.assertEqual(
            tree_util.tree_structure(b, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        for la, lt in zip(tree_util.tree_leaves(a), tree_util.tree_leaves(tree)):
            onp.testing.assert_array_equal(la, lt)
        self.assertListEqual(tree_util.tree_leaves(b), [])

    @parameterized.expand([[t] for t in TREES])
    def test_partition_with_never_select(self, tree):
        a, b = partition_utils.partition(tree, lambda x: False)
        self.assertEqual(
            tree_util.tree_structure(a, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        self.assertEqual(
            tree_util.tree_structure(b, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        self.assertListEqual(tree_util.tree_leaves(a), [])
        for la, lt in zip(tree_util.tree_leaves(b), tree_util.tree_leaves(tree)):
            onp.testing.assert_array_equal(la, lt)

    @parameterized.expand(
        [[tree, fn] for tree, fn in itertools.product(TREES, SELECT_FNS)]
    )
    def test_partion_combine_is_noop(self, tree, select_fn):
        a, b = partition_utils.partition(tree, select_fn)
        restored = partition_utils.combine(a, b)
        self.assertEqual(
            tree_util.tree_structure(tree), tree_util.tree_structure(restored)
        )
        for lr, lt in zip(tree_util.tree_leaves(restored), tree_util.tree_leaves(tree)):
            onp.testing.assert_array_equal(lr, lt)

    def test_can_nest(self):
        tree = {
            "a": (
                1,
                None,
                types.BoundedArray(3.14, 0, 100),
                types.Density2DArray(onp.zeros((1, 1))),
            ),
            "b": (
                2,
                None,
                types.BoundedArray(6.28, 0, 100),
                types.Density2DArray(onp.zeros((2, 2))),
            ),
            "c": (
                3,
                None,
                types.BoundedArray(9.42, 0, 100),
                types.Density2DArray(onp.zeros((3, 3))),
            ),
        }
        ints, other = partition_utils.partition(tree, lambda x: isinstance(x, int))
        bas, other = partition_utils.partition(
            other, lambda x: isinstance(x, types.BoundedArray)
        )
        das, other = partition_utils.partition(
            other, lambda x: isinstance(x, types.Density2DArray)
        )

        self.assertEqual(
            tree_util.tree_structure(ints, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        self.assertEqual(
            tree_util.tree_structure(bas, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        self.assertEqual(
            tree_util.tree_structure(das, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        self.assertEqual(
            tree_util.tree_structure(other, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )

        restored = partition_utils.combine(ints, bas, das, other)
        self.assertEqual(
            tree_util.tree_structure(restored, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        for lr, lt in zip(tree_util.tree_leaves(restored), tree_util.tree_leaves(tree)):
            onp.testing.assert_array_equal(lr, lt)

    def test_can_jit(self):
        tree = {
            "a": (
                1,
                None,
                types.BoundedArray(2.5, 0, 100),
                types.Density2DArray(onp.zeros((1, 1))),
            ),
            "b": (
                2,
                None,
                types.BoundedArray(5, 0, 100),
                types.Density2DArray(onp.zeros((2, 2))),
            ),
            "c": (
                3,
                None,
                types.BoundedArray(7.5, 0, 100),
                types.Density2DArray(onp.zeros((3, 3))),
            ),
        }

        densities, other = partition_utils.partition(
            tree, lambda x: isinstance(x, types.Density2DArray)
        )

        @jax.jit
        def restore_fn(densities):
            restored = partition_utils.combine(densities, other)
            return restored

        restored = restore_fn(densities)
        self.assertEqual(
            tree_util.tree_structure(restored, is_leaf=_is_custom_type_or_none),
            tree_util.tree_structure(tree, is_leaf=_is_custom_type_or_none),
        )
        for lr, lt in zip(tree_util.tree_leaves(restored), tree_util.tree_leaves(tree)):
            onp.testing.assert_array_equal(lr, lt)
