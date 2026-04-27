# Installation

`symmray` is available on both [pypi](https://pypi.org/project/symmray/) and
[conda-forge](https://anaconda.org/conda-forge/symmray). While `symmray` is
pure python itself, the preferred way to install it is with
[pixi](https://pixi.sh), which creates isolated and reproducible environments
that can mix packages from [`conda-forge`](https://conda-forge.org/) (the
default) and also [`pypi`](https://pypi.org/).

**Installing with `pixi` (preferred):**
```bash
pixi init symmray-project
cd symmray-project
pixi add symmray
```

**Installing with `pip`:**
```bash
pip install symmray
# or
uv pip install symmray
```
It is recommended to use [`uv`](https://docs.astral.sh/uv/) to install and
manage purely pypi based environments.

**Installing with `conda` / `mamba`:**
```bash
conda install -c conda-forge symmray
```
[`miniforge`](https://github.com/conda-forge/miniforge) is the recommended way
to manage and install a conda-based environment.


**Installing the latest version directly from github:**

If you want to checkout the latest version of features and fixes, you can
install directly from the github repository:
```bash
pip install -U git+https://github.com/jcmgray/symmray.git
```

**Installing a local, editable development version:**

If you want to make changes to the source code and test them out, you can
install a local editable version of the package:
```bash
git clone https://github.com/jcmgray/symmray.git
pip install --no-deps -U -e symmray/
```
