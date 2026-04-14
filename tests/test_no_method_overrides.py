"""To avoid accidental and ambiguous inheritance of methods, we test here that
every method the main user facing classes inherits, is defined in exactly one
class of the MRO.
"""

import inspect

import pytest

import symmray as sr

USER_FACING_CLASSES = [
    sr.AbelianArray,
    sr.FermionicArray,
    sr.AbelianArrayFlat,
    sr.FermionicArrayFlat,
    sr.BlockVector,
    sr.FlatVector,
]


ALLOWED_OVERRIDES = {"__init__"}


def _iter_methods(cls):
    """Yield `name` for every function-like attribute defined directly on
    ``cls`` (not inherited). Covers regular methods, ``classmethod``,
    ``staticmethod``, and ``property`` (via fget).
    """
    for name, value in vars(cls).items():
        if inspect.isfunction(value):
            yield name
        elif isinstance(value, (classmethod, staticmethod)):
            yield name
        elif isinstance(value, property):
            yield name


@pytest.mark.parametrize("cls", USER_FACING_CLASSES, ids=lambda c: c.__name__)
def test_no_method_overrides_in_mro(cls):
    """No method name should be defined in more than one class of ``cls``'s
    MRO (aside from a small allow-list of construction hooks).
    """
    definitions = {}
    for base in cls.__mro__:
        if base is object:
            continue
        for name in _iter_methods(base):
            if name in ALLOWED_OVERRIDES:
                continue
            definitions.setdefault(name, []).append(base.__name__)

    duplicates = {
        name: owners for name, owners in definitions.items() if len(owners) > 1
    }

    assert not duplicates, (
        f"{cls.__name__} has methods defined in multiple MRO classes: "
        + ", ".join(
            f"{name} -> {owners}" for name, owners in duplicates.items()
        )
    )
