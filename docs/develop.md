# Developer Notes


## Contributing

Contributions to `symmray` are very welcome, whether they are bug reports,
documentation fixes, examples, tests, or new features. If you are planning a
larger change, opening an issue first is often the easiest way to check the
approach before spending too much time on implementation.

Please also read the
[`symmray` Code of Conduct](https://github.com/jcmgray/symmray/blob/main/CODE_OF_CONDUCT.md).

Things to check if new functionality is added:

1. Ensure functions are unit tested. Heavy use of `@pytest.mark.parametrize`
   over symmetry types (`Z2`, `ZN`, `U1`, `Z2Z2`, `U1U1`), shapes, and seeds is
   the standard pattern. After any mutating op, call `x.check()` for internal
   consistency — it is a no-op unless `SYMMRAY_DEBUG=1` is set (the `test` and
   `pytest` pixi tasks set it automatically).
2. New method names must not collide across the MRO of any user-facing class.
   The single-definition MRO rule is enforced by
   `tests/test_no_method_overrides.py` (only `__init__` is allow-listed).
   When backend-specific delegation is needed, use `_<method>_<mixin>` helpers
   (e.g. `_copy_abelian`, `_transpose_abelian`).
3. Ensure functions have
   [NumPy-style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
4. Ensure code is formatted and linted with `pixi run lint`.
5. Add to `symmray/__init__.py` and `"__all__"` if appropriate.
6. Heavy dependencies (numpy, ...) must be imported inside functions, not at
   module top, to keep the import graph lean.
7. Add to `docs/changelog.md` and elsewhere in docs if appropriate


### AI Policy

Please treat the [numpy AI policy](https://numpy.org/devdocs/dev/ai_policy.html) as a rough guide.


## Development Setup

`symmray` uses [pixi](https://pixi.sh) to manage development environments and
reproducible tasks. The environments and tasks are defined in `pyproject.toml`,
which is the source of truth for the commands below.

After cloning the repository, install the pixi environments from the project
root:

```bash
git clone https://github.com/jcmgray/symmray.git
cd symmray
pixi install
```

You can then run project tasks with `pixi run ...`. For example, to run a
short Python command inside the default test environment:

```bash
pixi run -e testpymid python -c "import symmray as sr; print(sr.__version__)"
```


## Running the Tests

Testing `symmray` is handled by pixi tasks. The most common commands are:

```bash
pixi run -e testpymid test    # full suite with coverage, matches CI
```

The `test` task expands to:

```bash
SYMMRAY_DEBUG=1 pytest tests/ \
    --cov=symmray \
    --cov-report=xml \
    --verbose \
    --durations=10
```

For a narrower check, use the `pytest` task (which also sets `SYMMRAY_DEBUG=1`)
and forward arguments after `--`:

```bash
pixi run pytest -- tests/test_sparse/test_xxx.py -v
pixi run pytest -- tests/ -k "test_tensordot" -v
```

To run the full suite in a specific environment, use `-e`:

```bash
pixi run -e testpyold test
pixi run -e testpymid test
pixi run -e testpynew test
```

If you invoke pytest directly (without pixi), set `SYMMRAY_DEBUG=1` yourself to
enable `check()` calls throughout the library.


## Formatting the Code

`symmray` uses [`ruff`](https://docs.astral.sh/ruff/) to format imports and
code style. Use the predefined pixi tasks rather than running the tools
directly:

```bash
pixi run lint
pixi run format
```

The `format-all` task also runs notebook cleanup with `squeaky`:

```bash
pixi run format-all
```


## Building the Docs Locally

The documentation dependencies are managed by pixi. To build, clean, and serve
the docs locally, use:

```bash
pixi run docs
pixi run docs-clean
pixi run docs-serve
```

The local server hosts the built docs at `http://localhost:8000/`. The
generated HTML is in `docs/_build/html/`.

On ReadTheDocs, the build is driven by `.readthedocs.yml` and uses the
dedicated `readthedocs` pixi task.


## Minting a Release

`symmray` uses
[`hatch-vcs`](https://github.com/ofek/hatch-vcs) to derive the version
from git tags, and [GitHub Actions](https://github.com/jcmgray/symmray/actions)
to publish to [PyPI](https://pypi.org/project/symmray/). To mint a new
release:

1. Make sure all the
   [tests are passing on CI](https://github.com/jcmgray/symmray/actions/workflows/tests.yml).
2. `git tag` the release with the next `vX.Y.Z`.
3. Push the tag to GitHub: `git push --tags`. The release workflow will
   build the sdist and wheel and upload them to the
   [PyPI **test** server](https://test.pypi.org/project/symmray/).
4. If the test-pypi build looks good, create a GitHub release from the
   tag. Publishing the release triggers the same workflow to upload to
   the [PyPI **production** server](https://pypi.org/project/symmray/).
5. The [`conda-forge/symmray-feedstock`](https://github.com/conda-forge/symmray-feedstock)
   repo should automatically pick up the new PyPI release and build a
   new [conda package](https://anaconda.org/conda-forge/symmray); the
   recipe should only need to be manually updated if there are, for
   example, new dependencies.

Alternate manual release steps (after tagging):

1. Remove any old builds: `rm -rf dist/*`
2. Build the sdist and wheel: `python -m build`
3. Upload using twine: `twine upload dist/*`
