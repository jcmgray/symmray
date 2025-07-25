import functools

import autoray as ar

from .abelian_core import AbelianArray, BlockIndex
from .block_core import BlockVector
from .fermionic_core import FermionicArray
from .utils import DEBUG


def norm(x):
    """Compute the frobenius norm of an AbelianArray."""
    return x.norm()


@functools.cache
def _get_qr_fn(backend, stabilized=False):
    """The lower level qr_stabilized is not necessarily already defined."""
    _qr = ar.get_lib_fn(backend, "linalg.qr")

    if not stabilized:
        return _qr

    try:
        _qr_stab = ar.get_lib_fn(backend, "qr_stabilized")

        def _qr(x):
            q, _, r = _qr_stab(x)
            return q, r

    except ImportError:
        _qr_ubstab = _qr
        _diag = ar.get_lib_fn(backend, "diag")
        _reshape = ar.get_lib_fn(backend, "reshape")
        _abs = ar.get_lib_fn(backend, "abs")

        def _sgn(x):
            x0 = x == 0.0
            return (x + x0) / (_abs(x) + x0)

        def _qr(x):
            q, r = _qr_ubstab(x)
            s = _sgn(_diag(r))
            q = q * _reshape(s, (1, -1))
            r = r * _reshape(s, (-1, 1))
            return q, r

    return _qr


@functools.singledispatch
def qr(x, stabilized=False):
    """QR decomposition of an AbelianArray.

    Parameters
    ----------
    x : AbelianArray
        The block symmetric array to decompose.
    stabilized : bool, optional
        Whether to use a stabilized QR decomposition, that is, with positive
        diagonal elements in the R factor. Default is False.

    Returns
    -------
    q : AbelianArray
        The orthogonal matrix.
    r : AbelianArray
        The upper triangular matrix.
    """
    if x.ndim != 2:
        raise NotImplementedError(
            "qr only implemented for 2D AbelianArrays,"
            f" got {x.ndim}D. Consider fusing first."
        )

    # get the 'lower' qr function that acts on the blocks
    _qr = _get_qr_fn(x.backend, stabilized=stabilized)

    q_blocks = {}
    r_blocks = {}
    new_chargemap = {}

    for sector, array in x.blocks.items():
        q, r = _qr(array)
        q_blocks[sector] = q
        new_chargemap[sector[1]] = ar.shape(q)[1]
        # on r charge is 0, and dualnesses always opposite
        r_sector = (sector[1], sector[1])
        r_blocks[r_sector] = r

    bond_index = BlockIndex(new_chargemap, dual=x.indices[1].dual)

    q = x.copy_with(
        indices=(x.indices[0], bond_index),
        blocks=q_blocks,
    )
    # XXX: use copy_with here?
    r = x.__class__(
        indices=(bond_index.conj(), x.indices[1]),
        charge=x.symmetry.combine(),
        blocks=r_blocks,
        symmetry=x.symmetry,
    )

    if DEBUG:
        q.check()
        r.check()
        q.check_with(r, (1,), (0,))

    return q, r


@qr.register(FermionicArray)
def qr_fermionic(x, stabilized=False):
    q, r = qr.dispatch(AbelianArray)(x, stabilized=stabilized)

    if r.indices[0].dual:
        # inner index is like |x><x| so introduce a phase flip
        r.phase_flip(0, inplace=True)

    return q, r


def qr_stabilized(x):
    q, r = qr(x, stabilized=True)
    return q, None, r


def get_numpy_svd_with_fallback(x):
    import numpy as np

    def svd_with_fallback(x):
        try:
            return np.linalg.svd(x, full_matrices=False)
        except np.linalg.LinAlgError:
            import scipy.linalg as sla

            return sla.svd(x, full_matrices=False, lapack_driver="gesvd")

    return svd_with_fallback


# used by quimb
ar.register_function("symmray", "qr_stabilized", qr_stabilized)


@functools.singledispatch
def svd(x):
    if x.ndim != 2:
        raise NotImplementedError(
            "svd only implemented for 2D AbelianArrays,"
            f" got {x.ndim}D. Consider fusing first."
        )

    if x.backend == "numpy":
        _svd = get_numpy_svd_with_fallback(x)
    else:
        _svd = ar.get_lib_fn(x.backend, "linalg.svd")

    u_blocks = {}
    s_store = {}
    v_blocks = {}
    new_chargemap = {}

    for sector, array in x.blocks.items():
        u, s, v = _svd(array)
        u_blocks[sector] = u
        # v charge is 0, and dualnesses always opposite
        s_charge = sector[1]
        v_sector = (s_charge, s_charge)
        s_store[s_charge] = s
        v_blocks[v_sector] = v
        new_chargemap[sector[1]] = ar.shape(u)[1]

    bond_index = BlockIndex(new_chargemap, dual=x.indices[1].dual)

    u = x.copy_with(
        indices=(x.indices[0], bond_index),
        blocks=u_blocks,
    )
    s = BlockVector(s_store)
    # XXX: use copy_with here?
    v = x.__class__(
        indices=(bond_index.conj(), x.indices[1]),
        charge=x.symmetry.combine(),
        blocks=v_blocks,
        symmetry=x.symmetry,
    )

    if DEBUG:
        u.check()
        s.check()
        v.check()
        u.check_with(s, 1)
        u.check_with(v, (1,), (0,))
        v.check_with(s, 0)

    return u, s, v


@svd.register(FermionicArray)
def svd_fermionic(x):
    u, s, vh = svd.dispatch(AbelianArray)(x)

    if vh.indices[0].dual:
        # inner index is like |x><x| so introduce a phase flip
        vh.phase_flip(0, inplace=True)

    return u, s, vh


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


@functools.lru_cache(maxsize=2**14)
def calc_sub_max_bonds(sizes, max_bond):
    if max_bond < 0:
        # no limit
        return sizes

    # overall fraction of the total bond dimension to use
    frac = max_bond / sum(sizes)
    if frac >= 1.0:
        # keep all singular values
        return sizes

    # number of singular values to keep in each sector
    sub_max_bonds = [int(frac * sz) for sz in sizes]

    # distribute any remaining singular values to the smallest sectors
    rem = max_bond - sum(sub_max_bonds)

    for i in argsort(sub_max_bonds)[:rem]:
        sub_max_bonds[i] += 1

    return tuple(sub_max_bonds)


@functools.singledispatch
def svd_truncated(
    x, cutoff=-1.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0
):
    """Truncated svd or raw array ``x``.

    Parameters
    ----------
    cutoff : float
        Singular value cutoff threshold.
    cutoff_mode : {1, 2, 3, 4, 5, 6}
        How to perform the truncation:

        - 1: ['abs'], trim values below ``cutoff``
        - 2: ['rel'], trim values below ``s[0] * cutoff``
        - 3: ['sum2'], trim s.t. ``sum(s_trim**2) < cutoff``.
        - 4: ['rsum2'], trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
        - 5: ['sum1'], trim s.t. ``sum(s_trim**1) < cutoff``.
        - 6: ['rsum1'], trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int
        An explicit maximum bond dimension, use -1 for none.
    absorb : {-1, 0, 1, None}
        How to absorb the singular values. -1: left, 0: both, 1: right and
        None: don't absorb (return).
    renorm : {0, 1}
        Whether to renormalize the singular values (depends on `cutoff_mode`).
    """
    backend = x.backend

    # first perform untruncated svd
    U, s, VH = svd(x)

    if renorm:
        raise NotImplementedError("renorm not implemented yet.")

    if cutoff > 0.0:
        # first combine all singular values into a single, sorted array
        sall = s.to_dense()
        sall = ar.do("sort", sall, like=backend)

        if cutoff_mode == 1:
            # absolute cutoff
            abs_cutoff = cutoff
        elif cutoff_mode == 2:
            # relative cutoff
            abs_cutoff = sall[-1] * cutoff
        else:
            # possible square singular values
            power = {3: 2, 4: 2, 5: 1, 6: 1}[cutoff_mode]
            if power == 1:
                # sum1 or rsum1
                cum_spow = ar.do("cumsum", sall, 0, like=backend)
            else:
                # sum2 or rsum2
                cum_spow = ar.do("cumsum", sall**power, 0, like=backend)

            if cutoff_mode in (4, 6):
                # rsum1 or rsum2: relative cumulative cutoff
                cond = cum_spow >= cutoff * cum_spow[-1]
            else:
                # sum1 or sum2: absolute cumulative cutoff
                cond = cum_spow >= cutoff

            # translate to total number of singular values to keep
            n_chi_all = ar.do("count_nonzero", cond, like=backend)
            # and then to an absolute cutoff value
            abs_cutoff = sall[-n_chi_all]

        if 0 < max_bond < ar.size(sall):
            # also take into account a total maximum bond
            max_bond_cutoff = sall[-max_bond]
            if max_bond_cutoff > abs_cutoff:
                abs_cutoff = max_bond_cutoff

        # now find number of values to keep per sector
        sub_max_bonds = [
            int(ar.do("count_nonzero", ss >= abs_cutoff, like=backend))
            for ss in s.blocks.values()
        ]
    else:
        # size of each sector
        sector_sizes = tuple(map(ar.size, s.blocks.values()))
        # distribute max_bond proportionally to sector sizes
        sub_max_bonds = calc_sub_max_bonds(sector_sizes, max_bond)

    new_inner_chargemap = {}
    for (c0, c1), n_chi in zip(U.sectors, sub_max_bonds):
        # check how many singular values from this sector are valid
        if n_chi == 0:
            # remove this sector entirely
            U.blocks.pop((c0, c1))
            s.blocks.pop(c1)
            VH.blocks.pop((c1, c1))
            continue

        # slice the values and left and right vectors
        U.blocks[(c0, c1)] = U.blocks[(c0, c1)][:, :n_chi]
        s.blocks[c1] = s.blocks[c1][:n_chi]
        VH.blocks[(c1, c1)] = VH.blocks[(c1, c1)][:n_chi, :]

        # make sure the index chargemaps are updated too
        new_inner_chargemap[c1] = n_chi

    new_inner_chargemap = dict(sorted(new_inner_chargemap.items()))

    U.modify(
        indices=(
            U.indices[0],
            U.indices[1].copy_with(chargemap=new_inner_chargemap),
        )
    )
    VH.modify(
        indices=(
            VH.indices[0].copy_with(chargemap=new_inner_chargemap),
            VH.indices[1],
        )
    )

    if absorb is None:
        if DEBUG:
            U.check()
            U.check_with(s, 1)
            s.check()
            VH.check()
            VH.check_with(s, 0)
            U.check_with(VH, (1,), (0,))

        return U, s, VH

    # absorb the singular values block by block
    for c0, c1 in U.sectors:
        if absorb in (-1, "left"):
            U.blocks[(c0, c1)] = U.blocks[(c0, c1)] * s.blocks[c1].reshape(
                (1, -1)
            )
        elif absorb in (1, "right"):
            VH.blocks[(c1, c1)] = VH.blocks[(c1, c1)] * s.blocks[c1].reshape(
                (-1, 1)
            )
        elif absorb in (0, "both"):
            s_sqrt = ar.do("sqrt", s.blocks[c1], like=backend)
            U.blocks[(c0, c1)] = U.blocks[(c0, c1)] * s_sqrt.reshape((1, -1))
            VH.blocks[(c1, c1)] = VH.blocks[(c1, c1)] * s_sqrt.reshape((-1, 1))
        else:
            raise ValueError(f"Unknown absorb value: {absorb}")

    if DEBUG:
        U.check()
        U.check_with(s, 1)
        s.check()
        VH.check()
        VH.check_with(s, 0)
        U.check_with(VH, (1,), (0,))

    return U, None, VH


# used by quimb
ar.register_function("symmray", "svd_truncated", svd_truncated)


@functools.singledispatch
def eigh(a):
    """Perform a hermitian eigendecomposition on a AbelianArray."""
    if a.ndim != 2:
        raise NotImplementedError(
            "eigh only implemented for 2D AbelianArrays,"
            f" got {a.ndim}D. Consider fusing first."
        )
    if a.charge != a.symmetry.combine():
        raise ValueError("Total charge much be the identity (zero) element.")

    _eigh = ar.get_lib_fn(a.backend, "linalg.eigh")

    eval_blocks = {}
    evec_blocks = {}

    for sector, array in a.blocks.items():
        evals, evecs = _eigh(array)
        charge = sector[1]
        eval_blocks[charge] = evals
        evec_blocks[sector] = evecs

    eigenvalues = BlockVector(eval_blocks)
    eigenvectors = a.copy_with(blocks=evec_blocks)

    if DEBUG:
        eigenvectors.check()
        eigenvectors.check_with(eigenvalues, 1)
        eigenvalues.check()

    return eigenvalues, eigenvectors


@eigh.register(FermionicArray)
def eigh_fermionic(a):
    eigenvalues, eigenvectors = eigh.dispatch(AbelianArray)(a)

    if not a.indices[1].dual:
        symm = a.symmetry
        # inner index is like |x><x| so introduce a phase flip,
        # we don't explicitly have Wdag so put phase in eigenvalues
        # XXX: is this the most compatible thing to do?
        # it means ev @ diag(el) @ ev.H == a always
        for c in eigenvalues.sectors:
            if symm.parity(c):
                eigenvalues.blocks[c] = -eigenvalues.blocks[c]

    return eigenvalues, eigenvectors


@functools.singledispatch
def solve(a, b):
    if (a.ndim, b.ndim) != (2, 1):
        raise NotImplementedError(
            "solve only implemented for 2D AbelianArrays and 1D BlockVectors,"
            f" got {a.ndim}D and {b.ndim}D. Consider fusing first."
        )

    _solve = ar.get_lib_fn(a.backend, "linalg.solve")

    x_blocks = {}
    for sector, array in a.blocks.items():
        b_sector = (sector[0],)
        if b_sector in b.blocks:
            x_sector = (sector[1],)
            x_blocks[x_sector] = _solve(array, b.blocks[b_sector])

    # c_x = c_b - c_A
    sym = a.symmetry
    x_charge = sym.combine(b.charge, sym.sign(a.charge))

    x = b.copy_with(
        blocks=x_blocks,
        indices=(a.indices[1].conj(),),
        charge=x_charge,
    )

    if DEBUG:
        x.check()
        a.check_with(x, (1,), (0,))

    return x


@solve.register(FermionicArray)
def solve_fermionic(a, b):
    x = solve.dispatch(AbelianArray)(a, b)

    if x.indices[0].dual:
        # inner index is like |x><x| so introduce a phase flip
        x.phase_flip(0, inplace=True)

    return x
