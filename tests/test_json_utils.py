"""Defines tests for the `totypes.json_utils` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import unittest
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from totypes import json_utils, symmetry, types

ARRAYS = [
    1,
    2,
    jnp.ones((2, 3)),
    onp.ones((2, 3)),
]

BOUNDED_ARRAYS = [
    types.BoundedArray(array=1, lower_bound=0, upper_bound=2),
    types.BoundedArray(array=1, lower_bound=None, upper_bound=2),
    types.BoundedArray(array=1, lower_bound=0, upper_bound=None),
    types.BoundedArray(array=onp.ones((2, 2)), lower_bound=0, upper_bound=2),
    types.BoundedArray(array=onp.ones((2, 2)), lower_bound=None, upper_bound=2),
    types.BoundedArray(array=onp.ones((2, 2)), lower_bound=0, upper_bound=None),
]

DENSITY_2D_ARRAYS = [
    types.Density2DArray(
        array=jnp.ones((3, 3)),
        lower_bound=0,
        upper_bound=2,
        fixed_solid=jnp.zeros((3, 3), dtype=bool),
        fixed_void=jnp.ones((3, 3), dtype=bool),
        minimum_width=2,
        minimum_spacing=3,
        periodic=(True, False),
        symmetries=(symmetry.ROTATION_180, symmetry.REFLECTION_N_S),
    ),
    types.Density2DArray(
        array=onp.ones((3, 3)),
        lower_bound=0,
        upper_bound=2,
        fixed_solid=onp.zeros((3, 3), dtype=bool),
        fixed_void=onp.ones((3, 3), dtype=bool),
        minimum_width=2,
        minimum_spacing=3,
        periodic=(True, True),
        symmetries=(),
    ),
]


class SerializeDeserializeTest(unittest.TestCase):
    @parameterized.expand([[a] for a in ARRAYS + BOUNDED_ARRAYS + DENSITY_2D_ARRAYS])
    def test_serialize_deserialize_arrays(self, array):
        (expected_leaf,), expected_pytreedef = jax.tree_util.tree_flatten(array)

        serialized = json_utils.json_from_pytree(array)
        restored = json_utils.pytree_from_json(serialized)
        (leaf,), pytreedef = jax.tree_util.tree_flatten(restored)

        self.assertEqual(pytreedef, expected_pytreedef)
        onp.testing.assert_array_equal(leaf, expected_leaf)

    def test_serialize_deserialize_pytree(self):
        tree = {
            "arrays": ARRAYS,
            "bounded_arrays": BOUNDED_ARRAYS,
            "density_2d_arrays": DENSITY_2D_ARRAYS,
        }
        serialized = json_utils.json_from_pytree(tree)
        restored = json_utils.pytree_from_json(serialized)

        leaves, pytreedef = jax.tree_util.tree_flatten(restored)
        expected_leaves, expected_pytreedef = jax.tree_util.tree_flatten(restored)

        self.assertEqual(pytreedef, expected_pytreedef)
        for leaf, expected_leaf in zip(leaves, expected_leaves):
            onp.testing.assert_array_equal(leaf, expected_leaf)

    def test_serialize_with_extra_custom_namedtuple(self):
        class CustomObject(NamedTuple):
            x: onp.ndarray
            y: int
            z: str

        prefix = "CUSTOM_PREFIX"
        custom_types_and_prefixes = ((CustomObject, prefix),)

        obj = CustomObject(onp.arange(5), 22, "abc")

        serialized = json_utils.json_from_pytree(obj, custom_types_and_prefixes)
        restored = json_utils.pytree_from_json(serialized, custom_types_and_prefixes)

        self.assertIsInstance(restored, CustomObject)
        onp.testing.assert_array_equal(restored.x, obj.x)
        self.assertEqual(restored.y, obj.y)
        self.assertEqual(restored.z, obj.z)

    def test_serialize_with_extra_custom_dataclass(self):
        @dataclasses.dataclass
        class CustomObject:
            x: onp.ndarray
            y: int
            z: str

        prefix = "CUSTOM_PREFIX"
        custom_types_and_prefixes = ((CustomObject, prefix),)

        obj = CustomObject(onp.arange(5), 22, "abc")

        serialized = json_utils.json_from_pytree(obj, custom_types_and_prefixes)
        restored = json_utils.pytree_from_json(serialized, custom_types_and_prefixes)

        self.assertIsInstance(restored, CustomObject)
        onp.testing.assert_array_equal(restored.x, obj.x)
        self.assertEqual(restored.y, obj.y)
        self.assertEqual(restored.z, obj.z)

    def test_serialize_with_registered_custom_namedtuple(self):
        class CustomObject(NamedTuple):
            x: onp.ndarray
            y: int
            z: str

        json_utils.register_custom_type(CustomObject)

        obj = CustomObject(onp.arange(5), 22, "abc")

        serialized = json_utils.json_from_pytree(obj)
        restored = json_utils.pytree_from_json(serialized)

        self.assertIsInstance(restored, CustomObject)
        onp.testing.assert_array_equal(restored.x, obj.x)
        self.assertEqual(restored.y, obj.y)
        self.assertEqual(restored.z, obj.z)

    def test_serialize_with_registered_custom_dataclass(self):
        @dataclasses.dataclass
        class CustomObject:
            x: onp.ndarray
            y: int
            z: str

        json_utils.register_custom_type(CustomObject)

        obj = CustomObject(onp.arange(5), 22, "abc")

        serialized = json_utils.json_from_pytree(obj)
        restored = json_utils.pytree_from_json(serialized)

        self.assertIsInstance(restored, CustomObject)
        onp.testing.assert_array_equal(restored.x, obj.x)
        self.assertEqual(restored.y, obj.y)
        self.assertEqual(restored.z, obj.z)


class TestAllCustomTypesHaveAPrefix(unittest.TestCase):
    def test_all_custom_types_have_a_prefix(self):
        custom_types = json_utils._CUSTOM_TYPE_REGISTRY.values()
        for t in types.CUSTOM_TYPES:
            self.assertTrue(t in custom_types)
