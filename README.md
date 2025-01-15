# totypes - Custom types for topology optimization
![Continuous integration](https://github.com/invrs-io/totypes/actions/workflows/build-ci.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/totypes)

## Overview

The `totypes` package defines custom jax-compatible datatypes for use in a topology optimization, inverse design, or AI-guided design context. The custom types are pytree nodes consisting of standard jax arrays along with metadata that describe the desired characteristics of the arrays.
- `BoundedArray`, an array with optional lower and/or upper bounds, used e.g. for representing layer thicknesses.
- `Density2DArray`, an array with lower and upper bounds and characteristics such as fixed pixels, minimum feature size, or symmetry, used for representing layer density as is common in topology optimization.

Custom types do not modify the underlying arrays to ensure that e.g. bounds are obeyed. Instead, it is intended that any scheme for generating arrays be aware of and respect the metadata. For example, a topology optimization scheme could read the minimum width and spacing of layers, and use these quantities to construct an appropriate filtering scheme.

Several related utilities are also provided. The `json_utils` module provides functions for serializing and deserializing pytrees containing the custom types. And, the `symmetry` module provides functions that symmetrize arrays; the allowed symmetries of the `Density2DArray` are restricted to those implemented in the `symmetry` module.

## Install

```
pip install totypes
```
