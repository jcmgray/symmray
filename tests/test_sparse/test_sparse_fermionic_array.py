import numpy as np
import pytest

import symmray as sr

all_symmetries = ["Z2", "Z4", "U1"]


@pytest.mark.parametrize("symmetry", all_symmetries)
@pytest.mark.parametrize("subsizes", ["equal", "maximal", None])
@pytest.mark.parametrize("seed", range(3))
def test_fermi_norm(symmetry, subsizes, seed):
    x = sr.utils.get_rand(
        symmetry,
        (3, 4, 5, 6),
        fermionic=True,
        subsizes=subsizes,
        # set all duals to False
        duals=False,
        seed=seed,
    )
    x.randomize_phases(seed + 1, inplace=True)
    ne = x.norm()
    xc = x.conj()
    assert xc.phases != x.phases
    xx = sr.tensordot(x, xc, axes=4)
    n1 = float(xx) ** 0.5
    assert ne == pytest.approx(n1)
    xx = sr.tensordot(xc, x, axes=4)
    n2 = float(xx) ** 0.5
    assert ne == pytest.approx(n2)
    xd = x.H
    xx = sr.tensordot(xd, x, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    n3 = float(xx) ** 0.5
    assert ne == pytest.approx(n3)
    xx = sr.tensordot(x, xd, axes=[(0, 1, 2, 3), (3, 2, 1, 0)])
    n4 = float(xx) ** 0.5
    assert ne == pytest.approx(n4)


@pytest.mark.parametrize(
    "symmetry, charge",
    [
        ("Z2", 0),
        ("Z2", 1),
        ("Z4", 0),
        ("Z4", 1),
        ("Z4", 2),
        ("Z4", 3),
        ("U1", 0),
        ("U1", 1),
        ("U1", -1),
    ],
)
@pytest.mark.parametrize("seed", range(3))
def test_fermi_norm_phase_dual(symmetry, charge, seed):
    x = sr.utils.get_rand(
        symmetry,
        (3, 4, 5, 6),
        fermionic=True,
        charge=charge,
        oddpos="x",
        seed=seed,
    )
    n2 = x.norm() ** 2
    xd = x.dagger(phase_dual=True)
    na = sr.tensordot(xd, x, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    nb = sr.tensordot(x, xd, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    assert na == pytest.approx(nb)
    assert na == pytest.approx(n2)
    assert nb == pytest.approx(n2)

    xc = x.conj(phase_dual=True)
    nc = sr.tensordot(xc, x, axes=[(0, 1, 2, 3), (0, 1, 2, 3)])
    nd = sr.tensordot(x, xc, axes=[(0, 1, 2, 3), (0, 1, 2, 3)])
    assert nc == pytest.approx(nd)
    assert nc == pytest.approx(n2)
    assert nd == pytest.approx(n2)

    xct = x.conj(phase_dual=True).transpose()
    ne = sr.tensordot(xct, x, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    nf = sr.tensordot(x, xct, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    assert ne == pytest.approx(nf)
    assert ne == pytest.approx(n2)
    assert nf == pytest.approx(n2)


@pytest.mark.parametrize("symmetry", all_symmetries)
@pytest.mark.parametrize("subsizes", ["equal", "maximal", None])
@pytest.mark.parametrize("seed", range(3))
def test_transpose(symmetry, subsizes, seed):
    rng = sr.utils.get_rng(seed)
    x = sr.utils.get_rand(
        symmetry,
        (2, 3, 2, 3, 2),
        seed=rng,
        fermionic=True,
        subsizes=subsizes,
    )
    x.randomize_phases(seed + 1, inplace=True)
    perm = tuple(rng.permutation(x.ndim))
    y = x.transpose(perm)
    y.phase_sync(inplace=True)
    x = x.transpose(perm)
    perm = tuple(rng.permutation(x.ndim))
    y = x.transpose(perm)
    y.phase_sync(inplace=True)
    x = x.transpose(perm)
    perm = tuple(rng.permutation(x.ndim))
    y = x.transpose(perm)
    y.phase_sync(inplace=True)
    x = x.transpose(perm)
    assert x.allclose(y)


@pytest.mark.parametrize("seed", range(100))
def test_fuse_with_tensordot(seed):
    rng = np.random.default_rng(seed)

    ixs = {
        "a": sr.BlockIndex({0: 2, 1: 3}, dual=rng.choice([0, 1])),
        "b": sr.BlockIndex({0: 3, 1: 4}, dual=rng.choice([0, 1])),
        "c": sr.BlockIndex({0: 4, 1: 5}, dual=rng.choice([0, 1])),
        "d": sr.BlockIndex({0: 2, 1: 1}, dual=rng.choice([0, 1])),
        "e": sr.BlockIndex({0: 1, 1: 2}, dual=rng.choice([0, 1])),
    }

    ndim_left = rng.integers(1, 6)
    ndim_right = rng.integers(1, 6)
    left = rng.choice(list(ixs.keys()), size=ndim_left, replace=False).tolist()
    right = rng.choice(
        list(ixs.keys()), size=ndim_right, replace=False
    ).tolist()
    shared = set(left).intersection(right)
    ncon = len(shared)
    nleft = ndim_left - ncon
    nright = ndim_right - ncon
    ndim_out = nleft + nright
    perm_reverse = (*range(nright, ndim_out), *range(nright))

    axes_x = tuple(left.index(c) for c in shared)
    axes_y = tuple(right.index(c) for c in shared)

    x = sr.Z2FermionicArray.random(
        indices=[ixs[k] for k in left],
        seed=rng,
    )
    x.randomize_phases(seed + 1, inplace=True)
    y = sr.Z2FermionicArray.random(
        indices=[ixs[k].conj() for k in right],
        seed=rng,
    )
    y.randomize_phases(seed + 2, inplace=True)

    # basic
    z_b = sr.tensordot(
        x, y, axes=(axes_x, axes_y), mode="blockwise", preserve_array=True
    )

    # fused tensordot
    z_f = sr.tensordot(
        x, y, axes=(axes_x, axes_y), mode="fused", preserve_array=True
    )
    assert z_b.allclose(z_f)

    # reversed then transposed
    z_rb = sr.tensordot(
        y, x, axes=(axes_y, axes_x), mode="blockwise", preserve_array=True
    )
    z_rb.transpose(perm_reverse, inplace=True)
    assert z_b.allclose(z_rb)

    # reversed then transposed
    z_rf = sr.tensordot(
        y, x, axes=(axes_y, axes_x), mode="fused", preserve_array=True
    )
    z_rf.transpose(perm_reverse, inplace=True)
    assert z_b.allclose(z_rf)

    if axes_x:
        faxes_a = (min(axes_x),)
    else:
        faxes_a = ()

    if axes_y:
        faxes_b = (min(axes_y),)
    else:
        faxes_b = ()

    # fused-explicit then blockwise
    z_efb = sr.tensordot(
        x.fuse(axes_x, expand_empty=False),
        y.fuse(axes_y, expand_empty=False),
        [faxes_a, faxes_b],
        mode="blockwise",
        preserve_array=True,
    )
    assert z_b.allclose(z_efb)

    # fused-explicit then fused contraction
    z_eff = sr.tensordot(
        x.fuse(axes_x, expand_empty=False),
        y.fuse(axes_y, expand_empty=False),
        [faxes_a, faxes_b],
        mode="fused",
        preserve_array=True,
    )
    assert z_b.allclose(z_eff)

    # reverse fused-explicit then blockwise
    z_refb = sr.tensordot(
        y.fuse(axes_y, expand_empty=False),
        x.fuse(axes_x, expand_empty=False),
        [faxes_b, faxes_a],
        mode="blockwise",
        preserve_array=True,
    ).transpose(perm_reverse)
    assert z_b.allclose(z_refb)

    # reverse fused-explicit then fused contraction
    z_reff = sr.tensordot(
        y.fuse(axes_y, expand_empty=False),
        x.fuse(axes_x, expand_empty=False),
        [faxes_b, faxes_a],
        mode="fused",
        preserve_array=True,
    ).transpose(perm_reverse)
    assert z_b.allclose(z_reff)


@pytest.mark.parametrize("symmetry", all_symmetries)
@pytest.mark.parametrize("subsizes", ["equal", "maximal", None])
@pytest.mark.parametrize("seed", range(10))
def test_fuse_unfuse(symmetry, seed, subsizes):
    rng = sr.utils.get_rng(seed)
    x = sr.utils.get_rand(
        symmetry,
        (2, 3, 4, 3, 4),
        fermionic=True,
        seed=rng,
        subsizes=subsizes,
    )
    x.randomize_phases(rng, inplace=True)
    nfuse = rng.integers(1, x.ndim)
    axes = tuple(rng.choice(x.ndim, size=nfuse, replace=False))
    position = min(axes)
    new_order = (
        *range(position),
        *axes,
        *(ax for ax in range(position, x.ndim) if ax not in axes),
    )
    perm_back = sorted(range(x.ndim), key=lambda i: new_order[i])
    xf = x.fuse(axes)
    if len(axes) > 1:
        y = xf.unfuse(position)
    else:
        y = xf
    yt = y.transpose(perm_back)
    x.test_allclose(yt)


@pytest.mark.parametrize("symmetry", all_symmetries)
@pytest.mark.parametrize("subsizes", ["equal", "maximal", None])
@pytest.mark.parametrize("seed", range(10))
def test_fuse_unfuse_matrix(symmetry, seed, subsizes):
    rng = np.random.default_rng(seed)
    x = sr.utils.get_rand(
        symmetry=symmetry,
        shape=(2, 3, 4, 5, 6),
        fermionic=True,
        dist="uniform",
        seed=rng,
        subsizes=subsizes,
    )
    x.randomize_phases(rng, inplace=True)
    axes = tuple(rng.permutation(x.ndim))
    nleft = rng.integers(1, x.ndim - 1)

    axes_left = axes[:nleft]
    axes_right = axes[nleft:]
    order = (*axes_left, *axes_right)
    perm_back = sorted(range(x.ndim), key=lambda i: order[i])

    xf = x.fuse(axes_left, axes_right)

    assert xf.unfuse_all().transpose(perm_back).allclose(x)


@pytest.mark.parametrize("seed", range(10))
def test_transpose_vs_pyblock3(seed):
    pytest.importorskip("pyblock3")

    rng = np.random.default_rng(seed)

    ixs = [
        sr.BlockIndex({0: 2, 1: 3}, dual=rng.choice([0, 1])),
        sr.BlockIndex({0: 3, 1: 2}, dual=rng.choice([0, 1])),
        sr.BlockIndex({0: 3, 1: 3}, dual=rng.choice([0, 1])),
        sr.BlockIndex({0: 2, 1: 2}, dual=rng.choice([0, 1])),
        sr.BlockIndex({0: 2, 1: 2}, dual=rng.choice([0, 1])),
    ]
    x = sr.Z2FermionicArray.random(
        indices=ixs,
        dist="uniform",
        seed=rng,
    )
    x.randomize_phases(seed + 1, inplace=True)
    perma = tuple(rng.permutation(x.ndim))
    permb = tuple(rng.permutation(x.ndim))

    pb1 = x.to_pyblock3().transpose(perma).transpose(permb)
    pb2 = x.transpose(perma).transpose(permb).to_pyblock3()

    assert (pb1 - pb2).norm() == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize(
    "method",
    [
        "blockwise",
        "blockwise-switch",
        "fused",
        "fused-switch",
        "fused-explicit",
        "fused-explicit-switch",
    ],
)
def test_tensordot_vs_pyblock3(seed, method):
    pytest.importorskip("pyblock3")

    rng = np.random.default_rng(seed)

    ixs = {
        "a": sr.BlockIndex({0: 2, 1: 3}, dual=rng.choice([1])),
        "b": sr.BlockIndex({0: 3, 1: 4}, dual=rng.choice([1])),
        "c": sr.BlockIndex({0: 4, 1: 5}, dual=rng.choice([1])),
        "d": sr.BlockIndex({0: 2, 1: 1}, dual=rng.choice([1])),
    }
    sx = "".join(rng.permutation(list(ixs)))
    sy = "".join(rng.permutation(list(ixs)))

    axes_x = tuple(sx.index(c) for c in ixs)
    axes_y = tuple(sy.index(c) for c in ixs)

    x = sr.Z2FermionicArray.random(
        indices=[ixs[c] for c in sx],
        dist="uniform",
        seed=rng,
    )

    y = sr.Z2FermionicArray.random(
        indices=[ixs[c].conj() for c in sy],
        dist="uniform",
        seed=rng,
    )

    expected = np.tensordot(x.to_pyblock3(), y.to_pyblock3(), [axes_x, axes_y])

    if "switch" in method:
        x, y = y, x
        axes_x, axes_y = axes_y, axes_x

    if "blockwise" in method:
        z = sr.tensordot(x, y, axes=(axes_x, axes_y), mode="blockwise")

    elif "fused-explicit" in method:
        x = x.fuse(axes_x)
        y = y.fuse(axes_y)
        z = sr.tensordot(x, y, 1)

    elif "fused" in method:
        z = sr.tensordot(x, y, axes=(axes_x, axes_y), mode="fused")

    assert float(z) == pytest.approx(expected)


@pytest.mark.parametrize("symm", all_symmetries)
@pytest.mark.parametrize("seed", range(3))
def test_einsum_vs_tensordot(symm, seed):
    a = sr.utils.rand_index(symm, 4)
    b = sr.utils.rand_index(symm, 5)
    c = sr.utils.rand_index(symm, 6)
    d = sr.utils.rand_index(symm, 3)
    e = sr.utils.rand_index(symm, 2)
    eq = "abcd,baed->ec"
    rng = sr.utils.get_rng(seed)
    x = sr.utils.get_rand(
        symm,
        shape=(a, b, c, d),
        fermionic=True,
        seed=rng,
    )
    x.randomize_phases(rng, inplace=True)
    y = sr.utils.get_rand(
        symm,
        shape=(b.conj(), a.conj(), e, d.conj()),
        fermionic=True,
        seed=rng,
    )
    y.randomize_phases(rng, inplace=True)
    z1 = sr.tensordot(x, y, [(1, 0, 3), (0, 1, 3)]).transpose()
    xy = xy = sr.tensordot(x, y, ((), ()))
    seq = eq.replace(",", "")
    z2 = xy.einsum(seq)
    assert z1.allclose(z2)
