"""Defines tests for the `totypes.json_utils` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import functools
import unittest
from concurrent import futures
from typing import NamedTuple

import numpy as onp


class CustomObject(NamedTuple):
    x: onp.ndarray
    y: int
    z: str


def get_prefixes(register_type):
    from totypes import json_utils

    if register_type:
        json_utils.register_custom_type(CustomObject)
    return tuple(json_utils._CUSTOM_TYPE_REGISTRY.keys())


def serialize():
    from totypes import json_utils

    json_utils.register_custom_type(CustomObject)
    return json_utils.json_from_pytree(
        CustomObject(
            x=onp.ones((1, 1)),
            y=2,
            z="test",
        )
    )


def deserialize(serialized):
    from totypes import json_utils

    json_utils.register_custom_type(CustomObject)
    return json_utils.pytree_from_json(serialized)


def serialize_deserialize():
    return deserialize(serialize())


class TestServerClient(unittest.TestCase):
    def test_error_if_client_server_do_not_both_define_object(self):
        # Test that two separate processes must both register custom types. If the
        # two functions were to run on a single process, this owuld not be the case.
        with futures.ProcessPoolExecutor() as executor:
            running_tasks = [
                executor.submit(task)
                for task in [
                    functools.partial(get_prefixes, register_type=True),
                    functools.partial(get_prefixes, register_type=False),
                ]
            ]
            server_prefixes, client_prefixes = (task.result() for task in running_tasks)

        self.assertFalse(len(server_prefixes) == len(client_prefixes))

    def test_client_serialize_server_deserialize(self):
        # Perform serialization and deserialization in two separate processes, each
        # of which independently imports `json_utils` and registers `CustomObject`.
        with futures.ProcessPoolExecutor() as executor:
            serialized = executor.submit(serialize).result()
            pytree = executor.submit(deserialize, serialized).result()
        self.assertIsInstance(pytree, CustomObject)
        onp.testing.assert_array_equal(pytree.x, onp.ones((1, 1)))
        self.assertEqual(pytree.y, 2)
        self.assertEqual(pytree.z, "test")

        # As a sanity check, try to perform the same operation on a single process.
        # This should fail, because it results in duplicate registration.
        with self.assertRaisesRegex(
            ValueError, "Duplicate custom type registration for"
        ):
            with futures.ProcessPoolExecutor() as executor:
                executor.submit(serialize_deserialize).result()
