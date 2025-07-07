"""Defines tests for the `totypes.json_utils` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import os
import sys
import tempfile
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
    1 + 1j,
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
        fixed_solid=onp.zeros((3, 3), dtype=bool),
        fixed_void=onp.ones((3, 3), dtype=bool),
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

    def test_serialized_array_size_is_expected(self):
        # Tests the size of an array serialized with `json_from_pytree`.
        array = onp.ones((345, 678), dtype=onp.float64)
        expected_size = 8 * onp.prod(array.shape)

        serialized = json_utils.json_from_pytree(array)
        serialized_size = sys.getsizeof(serialized)
        serialized_ratio = serialized_size / expected_size

        with tempfile.TemporaryDirectory() as tempdir:
            fname = tempdir + "/test.json"
            with open(fname, "w") as f:
                f.write(serialized)
            disk_size = os.path.getsize(fname)
        disk_ratio = disk_size / expected_size

        self.assertLess(serialized_ratio, 1.4)
        self.assertLess(disk_ratio, 1.4)

    def test_serialized_pytree_size_is_expected(self):
        # Tests the size of a serialized pytree.
        pytree = {
            "array": onp.ones((500, 500)),
            "density": types.Density2DArray(
                array=onp.ones((300, 300)),
                lower_bound=0,
                upper_bound=2,
                fixed_solid=None,
                fixed_void=None,
                minimum_width=2,
                minimum_spacing=3,
                periodic=(True, True),
                symmetries=(),
            ),
            "bounded_array": types.BoundedArray(
                array=onp.ones((200, 200)), lower_bound=None, upper_bound=2
            ),
        }

        pytree_size = onp.sum(
            [sys.getsizeof(leaf) for leaf in jax.tree_util.tree_leaves(pytree)]
        )

        serialized = json_utils.json_from_pytree(pytree)
        serialized_size = sys.getsizeof(serialized)
        serialized_ratio = serialized_size / pytree_size

        with tempfile.TemporaryDirectory() as tempdir:
            fname = tempdir + "/test.json"
            with open(fname, "w") as f:
                f.write(serialized)
            disk_size = os.path.getsize(fname)
        disk_ratio = disk_size / pytree_size

        self.assertLess(serialized_ratio, 1.4)
        self.assertLess(disk_ratio, 1.4)

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

    def test_serialize_deserialize_pytree_from_disk(self):
        tree = {
            "arrays": ARRAYS,
            "bounded_arrays": BOUNDED_ARRAYS,
            "density_2d_arrays": DENSITY_2D_ARRAYS,
        }

        with tempfile.TemporaryDirectory() as tempdir:
            fname = tempdir + "/test.json"
            with open(fname, "w") as f:
                f.write(json_utils.json_from_pytree(tree))
            with open(fname) as f:
                restored = json_utils.pytree_from_json(f.read())

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

    def test_serialize_with_custom_namedtuple_having_internal_custom_type(self):
        class CustomObject(NamedTuple):
            x: types.Density2DArray
            y: types.BoundedArray
            z: str

        json_utils.register_custom_type(CustomObject)

        obj = CustomObject(
            x=types.Density2DArray(array=onp.zeros((5, 5))),
            y=types.BoundedArray(onp.ones((3,)), 0, 2),
            z="test",
        )

        serialized = json_utils.json_from_pytree(obj)
        restored = json_utils.pytree_from_json(serialized)
        self.assertIsInstance(restored, CustomObject)
        self.assertIsInstance(restored.x, types.Density2DArray)
        self.assertIsInstance(restored.y, types.BoundedArray)
        self.assertIsInstance(restored.z, str)

    def test_serialize_with_custom_dataclass_having_internal_custom_type(self):
        @dataclasses.dataclass
        class CustomObject:
            x: types.Density2DArray
            y: types.BoundedArray
            z: str

        json_utils.register_custom_type(CustomObject)

        obj = CustomObject(
            x=types.Density2DArray(array=onp.zeros((5, 5))),
            y=types.BoundedArray(onp.ones((3,)), 0, 2),
            z="test",
        )

        serialized = json_utils.json_from_pytree(obj)
        restored = json_utils.pytree_from_json(serialized)
        self.assertIsInstance(restored, CustomObject)
        self.assertIsInstance(restored.x, types.Density2DArray)
        self.assertIsInstance(restored.y, types.BoundedArray)
        self.assertIsInstance(restored.z, str)

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


class RegisterCustomTypeValidation(unittest.TestCase):
    def test_is_not_dataclass_or_namedtuple(self):
        class MyClass123:
            pass

        with self.assertRaisesRegex(
            ValueError, "Only dataclasses and namedtuples are supported"
        ):
            json_utils.register_custom_type(MyClass123)

    def test_is_not_type(self):
        @dataclasses.dataclass
        class MyClass123:
            pass

        json_utils.register_custom_type(MyClass123)
        self.assertTrue(
            any(
                ["MyClass123" in key for key in json_utils._CUSTOM_TYPE_REGISTRY.keys()]
            )
        )
        with self.assertRaisesRegex(
            ValueError, "`custom_type` must be a type, but got"
        ):
            json_utils.register_custom_type(MyClass123())

    def test_duplicate_registration(self):
        @dataclasses.dataclass
        class MyClass123:
            pass

        json_utils.register_custom_type(MyClass123)
        with self.assertWarnsRegex(UserWarning, "Duplicate custom type registration"):
            json_utils.register_custom_type(MyClass123)
