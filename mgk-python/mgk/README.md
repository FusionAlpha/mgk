# MGK1D Python Package

This directory contains the importable Python solver together with its
examples and regression tests:

```text
mgk/                    importable Python solver package
mgk/examples/           quick start and published-case configuration builders
mgk/tests/              Python regression test suite
```

The importable solver package is at the repository root as `mgk/`; the public
API for this release is `import mgk`. A small top-level `mgk1d.py` shim is
included only for older scripts and should not be used in new code.
