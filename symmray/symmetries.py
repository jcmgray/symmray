"""Definitions of various symmetries."""

import functools
from abc import ABC, abstractmethod

from .utils import get_rng


class Symmetry(ABC):
    __slots__ = ()

    @abstractmethod
    def valid(self, *charges):
        """Check if all charges are valid for the symmetry."""
        raise NotImplementedError

    @abstractmethod
    def combine(self, *charges):
        """Combine / add charges according to the symmetry."""
        raise NotImplementedError

    @abstractmethod
    def sign(self, charge, dual=True):
        """Negate a charge according to the symmetry and flag ``dual``."""
        raise NotImplementedError

    @abstractmethod
    def parity(self, charge):
        """Return the parity, 0 or 1, of a charge according to the symmetry."""
        raise NotImplementedError

    @abstractmethod
    def random_charge(self, seed=None):
        """Return a random valid charge, for testing purposes."""
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


@functools.lru_cache(2**14)
def sign_scalar(charge, dual=True):
    if dual:
        return -charge
    return charge


@functools.lru_cache(2**14)
def sign_tuple(charge, dual=True):
    return tuple(sign_scalar(c, dual) for c in charge)


class ZN(Symmetry):
    __slots__ = ("N",)

    def valid(self, *charges: int) -> bool:
        return all(charge in range(self.N) for charge in charges)

    def combine(self, *charges: int) -> int:
        return sum(charges) % self.N

    def sign(self, charge: int, dual=True) -> int:
        if dual:
            return (self.N - charge) % self.N
        return charge

    def parity(self, charge: int) -> int:
        return charge % 2

    def random_charge(self, seed=None) -> int:
        rng = get_rng(seed)
        return int(rng.integers(0, self.N))

    def __reduce__(self):
        return (get_symmetry, (f"Z{self.N}",))


class Z2(ZN):
    N = 2

    def sign(self, charge: int, dual=True) -> int:
        # shortcut: Z2 is self-inverse
        return charge


@functools.cache
def get_zn_symmetry_cls(N: int) -> type:
    """Get a ZN symmetry class."""
    if N == 2:
        return Z2
    return type(f"Z{N}", (ZN,), {"N": N})


class U1(Symmetry):
    __slots__ = ()

    def valid(self, *charges: int) -> bool:
        return all(isinstance(charge, int) for charge in charges)

    def combine(self, *charges: int) -> int:
        return sum(charges)

    def sign(self, charge: int, dual=True) -> int:
        return sign_scalar(charge, dual)

    def parity(self, charge: int) -> int:
        return charge % 2

    def random_charge(self, seed=None) -> int:
        rng = get_rng(seed)
        return int(rng.integers(-1, 2))


class Z2Z2(Symmetry):
    __slots__ = ()

    def valid(self, *charges: tuple[int, int]) -> bool:
        return all(
            isinstance(charge, tuple)
            and charge[0] in {0, 1}
            and charge[1] in {0, 1}
            for charge in charges
        )

    def combine(self, *charges: tuple[int, int]) -> tuple[int, int]:
        c0, c1 = 0, 0
        for cl, cr in charges:
            c0 ^= cl
            c1 ^= cr
        return (c0, c1)

    def sign(self, charge: tuple[int, int], dual=True) -> tuple[int, int]:
        # Z2Z2 is self-inverse
        return charge

    def parity(self, charge: tuple[int, int]) -> int:
        return charge[0] ^ charge[1]

    def random_charge(self, seed=None) -> tuple[int, int]:
        rng = get_rng(seed)
        charge = rng.integers(0, 2, size=2)
        return tuple(map(int, charge))


class U1U1(Symmetry):
    __slots__ = ()

    def valid(self, *charges: tuple[int, int]) -> bool:
        return all(
            isinstance(charge, tuple)
            and isinstance(charge[0], int)
            and isinstance(charge[1], int)
            for charge in charges
        )

    def combine(self, *charges: tuple[int, int]) -> tuple[int, int]:
        c0, c1 = 0, 0
        for cl, cr in charges:
            c0 += cl
            c1 += cr
        return (c0, c1)

    def sign(self, charge: tuple[int, int], dual=True) -> tuple[int, int]:
        return sign_tuple(charge, dual)

    def parity(self, charge: tuple[int, int]) -> int:
        return (charge[0] + charge[1]) % 2

    def random_charge(self, seed=None) -> tuple[int, int]:
        rng = get_rng(seed)
        charge = rng.integers(-1, 2, size=2)
        return tuple(map(int, charge))


@functools.lru_cache(2**14)
def get_symmetry(symmetry: str | Symmetry) -> Symmetry:
    """Get a symmetry instance by name.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry to get.

    Returns
    -------
    Symmetry
        The symmetry instance.
    """
    import re

    if isinstance(symmetry, Symmetry):
        return symmetry
    elif symmetry == "Z2":
        return Z2()
    elif m := re.fullmatch(r"Z(\d+)", symmetry):
        N = int(m.group(1))
        return get_zn_symmetry_cls(N)()
    elif symmetry == "U1":
        return U1()
    elif symmetry == "Z2Z2":
        return Z2Z2()
    elif symmetry == "U1U1":
        return U1U1()
    else:
        raise ValueError(f"Unknown symmetry: {symmetry}")


@functools.lru_cache(maxsize=2**15)
def calc_phase_permutation(
    parities: tuple[int, ...],
    perm: tuple[int, ...] = None,
) -> int:
    """Given sequence of parities and a permutation, compute the phase of the
    permutation acting on the odd charges. I.e. whether the number of swaps
    of odd sectors is even or odd.

    Parameters
    ----------
    parities : tuple of int
        The parities of the sectors.
    perm : tuple of int, optional
        The permutation of axes, by default None, which flips all axes.

    Returns
    -------
    int
        The phase of the permutation, either 1 or -1.
    """
    if perm is None:
        # assume flipping all
        if sum(parities) // 2 % 2:
            return -1
        return 1

    moved = set()
    swaps = 0
    for ax in perm:
        # we are moving charge at ax to the beginning
        if parities[ax]:
            # if it is odd, count how many odd charges it crosses
            for other_ax in range(ax):
                if other_ax not in moved and parities[other_ax]:
                    swaps += 1
        moved.add(ax)

    if swaps % 2:
        return -1
    return 1
