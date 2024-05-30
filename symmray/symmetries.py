""" """

import functools
from abc import ABC, abstractmethod


class Symmetry(ABC):
    __slots__ = ()

    @abstractmethod
    def valid(self, *charges):
        """Check if all charges are valid for the symmetry.
        """
        raise NotImplementedError

    @abstractmethod
    def combine(self, *charges):
        """Combine / add charges according to the symmetry.
        """
        raise NotImplementedError

    @abstractmethod
    def negate(self, charge, flow):
        """Negate a charge according to the symmetry.
        """
        raise NotImplementedError

    @abstractmethod
    def parity(self, charge):
        """Return the parity, 0 or 1, of a charge according to the symmetry.
        """
        raise NotImplementedError


@functools.lru_cache(2**14)
def scalar_negate(charge, flow):
    if flow:
        return -charge
    return charge


class Z2(Symmetry):
    __slots__ = ()

    def valid(self, *charges):
        return all(charge in {0, 1} for charge in charges)

    def combine(self, *charges):
        return sum(charges) % 2

    def negate(self, charge, flow):
        return scalar_negate(charge, flow)

    def parity(self, charge):
        return charge % 2


class U1(Symmetry):
    __slots__ = ()

    def valid(self, *charges):
        return all(isinstance(charge, int) for charge in charges)

    def combine(self, *charges):
        return sum(charges)

    def negate(self, charge, flow):
        return scalar_negate(charge, flow)

    def parity(self, charge):
        return charge % 2


@functools.lru_cache(2**14)
def get_symmetry(symmetry):
    """Get a symmetry object by name.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry to get.

    Returns
    -------
    Symmetry
        The symmetry object.
    """
    if symmetry == "Z2":
        return Z2()
    elif symmetry == "U1":
        return U1()
    elif isinstance(symmetry, Symmetry):
        return symmetry
    raise ValueError(f"Unknown symmetry: {symmetry}")



@functools.lru_cache(maxsize=2**15)
def calc_phase_permutation(parities, perm=None):
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