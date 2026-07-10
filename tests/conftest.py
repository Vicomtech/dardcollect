"""Pytest configuration for the DARDcollect unit suite.

The suite is CPU-only (no GPU, no models, no network). Importing `dardcollect`
triggers the package `__init__`, which preloads NVIDIA libs (a no-op on
CPU-only installs) and imports the heavy detection/audio submodules; this adds
a few seconds of collection overhead but no GPU is required.
"""
