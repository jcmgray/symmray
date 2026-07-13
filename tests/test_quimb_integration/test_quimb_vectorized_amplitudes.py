"""End-to-end tests of fermionic amplitude computation under compiled and
vectorized (batched) evaluation, on jax and torch. These run the full
``pack``/``unpack`` -> ``isel`` (squeeze to dummy modes) -> contraction with
traced configs, as in the batch amplitude example notebooks, but smaller.
"""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import symmray as sr

BATCHSIZE = 8


def _make_amplitude_fn_hotrg(skeleton):
    """Get approximate contraction function returning ``(mantissa, exponent)``,
    numerically exact here since ``max_bond`` is not saturated.
    """

    def amplitude(x, params):
        tn = qtn.unpack(params, skeleton)
        tnx = tn.isel({tn.site_ind(s): x[i] for i, s in enumerate(tn.sites)})
        return tnx.contract_hotrg(max_bond=4, cutoff=0.0, strip_exponent=True)

    return amplitude


def _setup(phys_dim, odd_sites, seed=7):
    """Build a small flat fermionic PEPS, its exact amplitude closure, a
    batch of valid configs, and eager numpy reference amplitudes.
    """
    peps = sr.networks.PEPS_fermionic_rand(
        "Z2",
        2,
        3,
        bond_dim=2,
        phys_dim=phys_dim,
        seed=42,
        flat=True,
        subsizes="equal",
        site_charge=lambda site: int(site in odd_sites),
    )
    charge = sum(x.charge for x in peps.arrays) % 2
    params, skeleton = qtn.pack(peps)

    def amplitude(x, params):
        # config x is per site state indices, possibly traced or batched
        tn = qtn.unpack(params, skeleton)
        tnx = tn.isel({tn.site_ind(s): x[i] for i, s in enumerate(tn.sites)})
        return tnx.contract(all, output_inds=())

    # random configs, fixing the last site so the total parity matches the
    # network charge, the parity of state k being k // (phys_dim // 2) for
    # Z2 with equal subsizes (even states first)
    k = phys_dim // 2
    rng = sr.utils.get_rng(seed)
    xs = rng.integers(0, phys_dim, size=(BATCHSIZE, peps.nsites))
    xs = xs.astype(np.int32)
    need = (charge + (xs[:, :-1] // k).sum(axis=1)) % 2
    xs[:, -1] = k * need + xs[:, -1] % k

    refs = [float(amplitude(x, params)) for x in xs]
    # relative comparisons need non-vanishing amplitudes
    assert all(abs(r) > 1e-12 for r in refs)
    return params, skeleton, amplitude, xs, refs


def _finite_diff_reference(amplitude, x, params, leaf, entry, eps=1e-5):
    """Central finite difference of the eager numpy amplitude with respect
    to a single parameter entry.
    """
    leaves, ref = qu.utils.tree_flatten(params, get_ref=True)
    leaves = [a.copy() for a in leaves]
    leaves[leaf].flat[entry] += eps
    ap = amplitude(x, qu.utils.tree_unflatten(leaves, ref))
    leaves[leaf].flat[entry] -= 2 * eps
    am = amplitude(x, qu.utils.tree_unflatten(leaves, ref))
    return (ap - am) / (2 * eps)


def _largest_grad_entry(gleaves):
    """Locate the (leaf, entry) with the largest gradient magnitude, so the
    finite difference check below is guaranteed meaningful.
    """
    leaf = max(range(len(gleaves)), key=lambda i: np.abs(gleaves[i]).max())
    entry = np.abs(gleaves[leaf]).argmax()
    return leaf, entry


AMPLITUDE_CASES = [
    pytest.param(2, (), id="spinless-even"),
    pytest.param(2, ((0, 0), (1, 1), (0, 2)), id="spinless-odd"),
    pytest.param(4, ((0, 1),), id="spinful-odd"),
]

ODD_CASE = (2, ((0, 0), (1, 1), (0, 2)))


class TestVectorizedFermionicAmplitudes:
    @pytest.mark.parametrize("phys_dim,odd_sites", AMPLITUDE_CASES)
    def test_jax_jit_and_vmap(self, phys_dim, odd_sites):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        params, _, amplitude, xs, refs = _setup(phys_dim, odd_sites)
        jparams = jax.tree.map(jnp.asarray, params)
        jxs = jnp.asarray(xs)

        a0 = jax.jit(amplitude)(jxs[0], jparams)
        assert float(a0) == pytest.approx(refs[0], rel=1e-10)

        av = jax.jit(jax.vmap(amplitude, in_axes=(0, None)))(jxs, jparams)
        assert list(map(float, av)) == pytest.approx(refs, rel=1e-10)

    @pytest.mark.parametrize("phys_dim,odd_sites", AMPLITUDE_CASES)
    def test_torch_vmap(self, phys_dim, odd_sites):
        torch = pytest.importorskip("torch")

        params, _, amplitude, xs, refs = _setup(phys_dim, odd_sites)
        tparams = qu.tree_map(torch.as_tensor, params)
        txs = torch.as_tensor(xs)

        a0 = amplitude(txs[0], tparams)
        assert float(a0) == pytest.approx(refs[0], rel=1e-10)

        av = torch.vmap(amplitude, in_dims=(0, None))(txs, tparams)
        assert list(map(float, av)) == pytest.approx(refs, rel=1e-10)

    def test_jax_hotrg_jit_vmap(self):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        params, skeleton, _, xs, refs = _setup(*ODD_CASE)
        amplitude = _make_amplitude_fn_hotrg(skeleton)
        jparams = jax.tree.map(jnp.asarray, params)
        jxs = jnp.asarray(xs)

        m, e = jax.jit(amplitude)(jxs[0], jparams)
        a0 = float(m) * 10.0 ** float(e)
        assert a0 == pytest.approx(refs[0], rel=1e-10)

        mv, ev = jax.jit(jax.vmap(amplitude, in_axes=(0, None)))(jxs, jparams)
        av = [float(m) * 10.0 ** float(e) for m, e in zip(mv, ev)]
        assert av == pytest.approx(refs, rel=1e-10)

    def test_torch_hotrg_vmap(self):
        torch = pytest.importorskip("torch")

        params, skeleton, _, xs, refs = _setup(*ODD_CASE)
        amplitude = _make_amplitude_fn_hotrg(skeleton)
        tparams = qu.tree_map(torch.as_tensor, params)
        txs = torch.as_tensor(xs)

        mv, ev = torch.vmap(amplitude, in_dims=(0, None))(txs, tparams)
        av = [float(m) * 10.0 ** float(e) for m, e in zip(mv, ev)]
        assert av == pytest.approx(refs, rel=1e-10)

    def test_torch_export_compile(self):
        torch = pytest.importorskip("torch")

        params, _, amplitude, xs, refs = _setup(*ODD_CASE)

        # direct torch.compile of `amplitude` fails, dynamo's numpy interop
        # mangles the static sector arithmetic, instead: wrap in a module,
        # export to trace all python out into a pure computational graph,
        # then compile that with nothing dynamic left
        class TNAmplitudeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                flat, self.pytree = qu.utils.tree_flatten(params, get_ref=True)
                self.params = torch.nn.ParameterList(
                    [torch.as_tensor(p) for p in flat]
                )

                def f(x):
                    tparams = qu.utils.tree_unflatten(self.params, self.pytree)
                    return amplitude(x, tparams)

                self.f = torch.vmap(f)

            def forward(self, x):
                return self.f(x)

        txs = torch.as_tensor(xs)
        model = TNAmplitudeModel()
        model.eval()

        gm = torch.export.export(model, (txs,)).module()
        av = gm(txs)
        assert list(map(float, av.detach())) == pytest.approx(refs, rel=1e-10)

        cm = torch.compile(gm, fullgraph=True)
        av = cm(txs)
        assert list(map(float, av.detach())) == pytest.approx(refs, rel=1e-10)

        # torch can't bake func.grad into the exported graph itself, but
        # autograd flows through the compiled forward, completing the
        # compile + vmap + grad composition as a training loop would
        av.sum().backward()
        grads = [p.grad for p in gm.parameters()]
        assert all(g is not None for g in grads)

        # eager reference: same scalar loss on requires_grad params
        gparams = qu.tree_map(
            lambda a: torch.as_tensor(a).requires_grad_(), params
        )
        torch.stack([amplitude(x, gparams) for x in txs]).sum().backward()
        grefs = [p.grad for p in qu.utils.tree_flatten(gparams)]
        for g, gr in zip(grads, grefs):
            assert g.numpy() == pytest.approx(gr.numpy(), rel=1e-10)

    def test_jax_grad(self):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        params, _, amplitude, xs, _ = _setup(*ODD_CASE)
        jparams = jax.tree.map(jnp.asarray, params)

        grads = jax.grad(amplitude, argnums=1)(jnp.asarray(xs[0]), jparams)
        # flatten with quimb to match the ordering of params
        gleaves = [np.asarray(g) for g in qu.utils.tree_flatten(grads)]
        assert all(np.all(np.isfinite(g)) for g in gleaves)

        leaf, entry = _largest_grad_entry(gleaves)
        g = gleaves[leaf].flat[entry]
        assert g != 0.0
        fd = _finite_diff_reference(amplitude, xs[0], params, leaf, entry)
        assert g == pytest.approx(fd, rel=1e-5)

    def test_jax_jit_vmap_grad(self):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        params, _, amplitude, xs, _ = _setup(*ODD_CASE)
        jparams = jax.tree.map(jnp.asarray, params)
        jxs = jnp.asarray(xs)

        # fully composed per-config batched gradients, as traced by a
        # batched VMC optimizer
        gfn = jax.jit(
            jax.vmap(jax.grad(amplitude, argnums=1), in_axes=(0, None))
        )
        bleaves = qu.utils.tree_flatten(gfn(jxs, jparams))
        assert max(float(jnp.max(jnp.abs(g))) for g in bleaves) > 0

        # each batch row should match the eager unbatched gradient, which
        # test_jax_grad checks against a finite difference
        gfn_single = jax.grad(amplitude, argnums=1)
        for b in range(BATCHSIZE):
            gleaves = qu.utils.tree_flatten(gfn_single(jxs[b], jparams))
            for gb, g in zip(bleaves, gleaves):
                assert np.asarray(gb[b]) == pytest.approx(
                    np.asarray(g), rel=1e-10
                )

    def test_torch_vmap_grad(self):
        torch = pytest.importorskip("torch")

        params, _, amplitude, xs, _ = _setup(*ODD_CASE)
        tparams = qu.tree_map(torch.as_tensor, params)
        txs = torch.as_tensor(xs)

        # fully composed per-config batched gradients, as traced by a
        # batched VMC optimizer
        gfn = torch.vmap(
            torch.func.grad(amplitude, argnums=1), in_dims=(0, None)
        )
        bleaves = qu.utils.tree_flatten(gfn(txs, tparams))
        assert max(float(g.abs().max()) for g in bleaves) > 0

        # each batch row should match the eager unbatched gradient, which
        # test_torch_grad checks against a finite difference
        gfn_single = torch.func.grad(amplitude, argnums=1)
        for b in range(BATCHSIZE):
            gleaves = qu.utils.tree_flatten(gfn_single(txs[b], tparams))
            for gb, g in zip(bleaves, gleaves):
                assert gb[b].numpy() == pytest.approx(g.numpy(), rel=1e-10)

    def test_torch_grad(self):
        torch = pytest.importorskip("torch")

        params, _, amplitude, xs, _ = _setup(*ODD_CASE)
        tparams = qu.tree_map(
            lambda a: torch.as_tensor(a).requires_grad_(), params
        )

        a = amplitude(torch.as_tensor(xs[0]), tparams)
        a.backward()

        gleaves = [
            np.zeros(p.shape) if p.grad is None else p.grad.numpy()
            for p in qu.utils.tree_flatten(tparams)
        ]
        assert all(np.all(np.isfinite(g)) for g in gleaves)

        leaf, entry = _largest_grad_entry(gleaves)
        g = gleaves[leaf].flat[entry]
        assert g != 0.0
        fd = _finite_diff_reference(amplitude, xs[0], params, leaf, entry)
        assert g == pytest.approx(fd, rel=1e-5)
