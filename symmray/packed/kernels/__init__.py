"""Interchangeable sets of compute kernels for the packed backend.

Each kernel set is a module with the same functions, acting only on numpy
arrays. All packed array logic is shared between sets:

- ``plan_copy(src_base, src_strides, dst_base, dst_strides, shapes, dst_size,
  fill_zero, coalesced=False)``: prepare a strided copy. Entry ``i`` copies a
  block of shape ``shapes[i]`` from ``src``, starting at ``src_base[i]`` with
  element strides ``src_strides[i]``, into ``dst``, starting at ``dst_base[i]``
  with strides ``dst_strides[i]``. If ``fill_zero``, parts of ``dst`` not
  written to are zero. Set ``coalesced`` if the entries come from this kernel
  set's ``fuse_assemble``, which may already have merged contiguous dims.

- ``run_copy(plan, src, scales=None)``: run a copy plan, return a new buffer of
  length ``dst_size``. Optional ``scales`` multiply each entry.

- ``plan_muldiag(offsets, shapes, axes, vbases)``: prepare to multiply each C
  contiguous block ``i``, at ``offsets[i]`` with shape ``shapes[i]``, along
  each axis ``axes[k]`` by the vector elements starting at ``vbases[i, k]``.

- ``run_muldiag(plan, src, v, power)``: run a multiply plan. ``power`` is 1 to
  multiply or -1 to divide.

- ``plan_gemm(a_off, rows, inner, b_off, cols, c_off)``: prepare matrix
  products. Entry ``i`` multiplies the C contiguous ``(rows[i], inner[i])``
  matrix at ``a_off[i]`` in ``a`` by the ``(inner[i], cols[i])`` matrix at
  ``b_off[i]`` in ``b``, into ``c_off[i]`` of the output.

- ``run_gemm(plan, a, b, out_size)``: run the products into a new buffer.

- ``run_scale(src, offsets, scales)``: multiply each C contiguous block of
  ``src``, delimited by ``offsets``, by its entry in ``scales``.

- ``permutation_phases(parities, parity_offsets, sectors, axes, flips)``: the
  int8 phase of each block from permuting its axes into the order ``axes`` and
  flipping the axes ``flips``. None if all +1. The parity of charge id ``c`` on
  axis ``ax`` is ``parities[parity_offsets[ax] + c]``.

Plans can be cached and reused for arrays with the same structure.
"""

import functools
import os

_KERNEL_MODULES = {
    "numpy": "symmray.packed.kernels.numpy_kernels",
    "numba": "symmray.packed.kernels.numba_kernels",
}

_current = None


def register_kernels(name, module_name):
    """Register a new kernel set, given the name of its module."""
    _KERNEL_MODULES[name] = module_name


@functools.cache
def load_kernels(name):
    """Import and return the kernel module ``name``."""
    import importlib

    return importlib.import_module(_KERNEL_MODULES[name])


def _default_kernels_name():
    name = os.environ.get("SYMMRAY_PACKED_KERNELS", None)
    if name is not None:
        return name
    try:
        import numba  # noqa: F401

        return "numba"
    except ImportError:
        return "numpy"


def get_kernels():
    """Get the current kernel set module."""
    global _current
    if _current is None:
        _current = load_kernels(_default_kernels_name())
    return _current


def get_kernels_name():
    """Get the name of the current kernel set."""
    return get_kernels().NAME


def set_packed_kernels(name):
    """Set the kernel set of the packed backend, e.g. ``"numpy"`` or
    ``"numba"``. Clears cached plans, since they are kernel specific.
    """

    global _current
    _current = load_kernels(name)

    from ..packed_array_common import clear_plan_cache

    clear_plan_cache()
