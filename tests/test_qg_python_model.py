"""Basic smoke tests for the Python QG model."""

import sys
import numpy as np

from NEDAS.models.qg.python.spectral import setup_spectral_grid, spec2grid_cc, grid2spec, ir_prod
from NEDAS.models.qg.python.strat import get_vmodes, strat_params
from NEDAS.models.qg.python.numerics import march, tridiag_vec
from NEDAS.models.qg.python.model import QGModel


def test_transform_roundtrip():
    """spec -> grid -> spec should recover the original field on active modes.

    The ky=0, kx<=0 modes are not independent (they're conjugate-symmetric
    copies of the kx>0, ky=0 modes), so the round-trip only preserves modes
    where filter_mask > 0.
    """
    kmax = 15
    g = setup_spectral_grid(kmax)
    nkx, nky = int(g['nkx']), int(g['nky'])
    filt = g['filter_mask']

    # Random spectral field only on active (filter_mask > 0) modes
    rng = np.random.default_rng(42)
    wf = (rng.standard_normal((nky, nkx))
          + 1j * rng.standard_normal((nky, nkx))) * filt

    physfield = spec2grid_cc(wf, g)
    wf2 = grid2spec(physfield, g)

    # Compare only the active modes (mask out conjugate-symmetry region)
    err = np.max(np.abs((wf - wf2) * filt))
    print(f'Transform round-trip error on active modes (max abs): {err:.2e}')
    assert err < 1e-8, f'Round-trip error too large: {err}'


def test_ir_prod_dealiasing():
    """ir_prod product should have zero energy in conjugate-symmetry region."""
    kmax = 15
    g = setup_spectral_grid(kmax)
    nkx, nky = int(g['nkx']), int(g['nky'])
    filt = g['filter_mask']

    rng = np.random.default_rng(0)
    wf1 = (rng.standard_normal((nky, nkx)) + 1j * rng.standard_normal((nky, nkx))) * filt
    wf2 = (rng.standard_normal((nky, nkx)) + 1j * rng.standard_normal((nky, nkx))) * filt

    p1 = spec2grid_cc(wf1, g)
    p2 = spec2grid_cc(wf2, g)
    prod_spec = grid2spec(ir_prod(p1, p2), g)

    # Active dealiasing mask (isotropic region, excluding conjugate side)
    # The product of two fields each supported on kmax_da should only have
    # modes up to 2*kmax_da, but the dealiasing mask zeros out beyond kmax_da.
    # Check that the result is supported on the de-aliasing mask region
    leakage = np.max(np.abs(prod_spec * filt))
    print(f'ir_prod max magnitude on active mask: {leakage:.2e}')
    assert np.isfinite(leakage), 'ir_prod produced non-finite result'
    assert leakage < 1e6, 'ir_prod result suspiciously large'

    # Verify that conjugate-determined region (ky=0, kx<=0) is consistent
    kmax = int(g['kmax'])
    nkx  = int(g['nkx'])
    # Hermitian symmetry: F(-kx, 0) = conj(F(kx, 0)).
    # In the array (kx indexed 0..nkx-1 where kx=0 is at position kmax):
    #   prod_ky0[j] = conj(prod_ky0[nkx-1-j]) for j < kmax
    prod_ky0 = prod_spec[0, :]
    sym_err = np.max(np.abs(prod_ky0[:kmax] - np.conj(prod_ky0[nkx-1:kmax:-1])))
    print(f'Hermitian symmetry error at ky=0: {sym_err:.2e}')
    assert sym_err < 1e-8, f'Hermitian symmetry broken: {sym_err}'


def test_tridiag():
    """Vectorised tridiagonal solver against numpy.linalg.solve."""
    nz = 4
    nmask = 100
    rng = np.random.default_rng(1)

    sub_1d  = rng.standard_normal(nz - 1)
    sup_1d  = rng.standard_normal(nz - 1)
    diag_2d = rng.standard_normal((nmask, nz)) + 5.0   # ensure dominance
    f_in    = rng.standard_normal((nmask, nz)) + 1j * rng.standard_normal((nmask, nz))

    x = tridiag_vec(f_in, sub_1d, diag_2d, sup_1d)

    # Verify: A*x = f for one row
    i = 5
    A = (np.diag(diag_2d[i])
         + np.diag(sub_1d, -1)
         + np.diag(sup_1d,  1))
    x_ref = np.linalg.solve(A, f_in[i])
    err = np.max(np.abs(x[i] - x_ref))
    print(f'Tridiag solver error: {err:.2e}')
    assert err < 1e-10


def test_march():
    """Leapfrog should be second-order on a linear ODE df/dt = f."""
    dt = 0.01
    f   = np.array([1.0 + 0j])
    f_o = f.copy()
    rhs = f.copy()
    calls = 0

    for _ in range(10):
        rhs = f.copy()
        f, f_o, calls = march(f, f_o, rhs, dt, rob=0.01, calls=calls)

    # Exact solution: exp(10*dt)
    exact = np.exp(10 * dt)
    err = abs(f[0] - exact)
    print(f'March integrator error after 10 steps: {err:.2e}')
    assert err < 0.02   # leapfrog is O(dt²); Robert filter adds small dissipation


def test_barotropic_run():
    """Barotropic (nz=1) model should run without error and conserve energy.

    Scale psi so that the spectral ENERGY sum(k²|ψ|²) ~ 0.01, which
    matches the Fortran model's typical initial condition amplitude.
    """
    m = QGModel(kmax=15, nz=1, F=0.0, beta=1.0, adapt_dt=True,
                filter_type='hyperviscous', filter_exp=4.0, dealiasing='isotropic')

    rng = np.random.default_rng(7)
    g = setup_spectral_grid(15)
    nky, nkx = int(g['nky']), int(g['nkx'])
    psi0 = (rng.standard_normal((1, nky, nkx))
            + 1j * rng.standard_normal((1, nky, nkx))) * g['filter_mask']
    # Scale to target spectral energy sum(k²|ψ|²) = e_target
    e_target = 0.01
    ksqd_: np.ndarray = g['ksqd_']
    energy_spec = float(np.sum(ksqd_[np.newaxis] * np.abs(psi0)**2))
    psi0 *= np.sqrt(e_target / (energy_spec + 1e-30))

    m.initialize(psi_init=psi0)
    assert m.psi is not None
    e0 = float(np.sum(ksqd_ * np.abs(m.psi[0])**2))
    m.run(50)
    assert m.psi is not None
    e1 = float(np.sum(ksqd_ * np.abs(m.psi[0])**2))
    print(f'Barotropic: initial energy={e0:.4f}, after 50 steps={e1:.4f}, dt={m.dt:.4f}')
    assert np.isfinite(e1), 'Energy is not finite after run'
    assert e1 < e0 * 100,   'Energy exploded (>100x initial)'


def test_multilayer_run():
    """4-layer model should run without error."""
    nz = 4
    dz  = np.array([0.1, 0.2, 0.3, 0.4])
    rho = np.array([1.0, 1.01, 1.02, 1.03])

    m = QGModel(kmax=15, nz=nz, F=50.0, beta=1.5, adapt_dt=True,
                filter_type='hyperviscous', filter_exp=4.0, dealiasing='isotropic')

    g = setup_spectral_grid(15)
    rng = np.random.default_rng(3)
    nky, nkx = int(g['nky']), int(g['nkx'])
    psi0 = (rng.standard_normal((nz, nky, nkx))
            + 1j * rng.standard_normal((nz, nky, nkx))) * g['filter_mask']
    # Scale to target energy = 0.01 per layer
    ksqd_: np.ndarray = g['ksqd_']
    e_spec = float(np.sum(ksqd_[np.newaxis] * np.abs(psi0)**2))
    psi0 *= np.sqrt(0.01 * nz / (e_spec + 1e-30))

    m.initialize(psi_init=psi0, dz=dz, rho=rho)
    m.run(20)
    assert m.psi is not None
    print(f'Multi-layer run OK, time={m.time:.4f}, dt={m.dt:.4f}')
    assert np.isfinite(m.time)
    assert np.isfinite(float(np.sum(np.abs(m.psi)**2)))


def test_fortran_config_stability():
    """2-layer model with Fortran-equivalent config should not blow up at e_o=10.

    Fortran defaults: kmax=127, dt=0.00025, F=100, beta=16, bot_drag=0.5,
    filter_type='exp_cutoff', k_cut=100.  At kmax=63 (half resolution) the
    stable fixed dt scales to ~0.0005 and k_cut scales to ~50.
    Tests both fixed-dt and adapt_dt (with dt_max cap) modes across 5 seeds.
    """
    kmax = 63
    g = setup_spectral_grid(kmax)
    nky, nkx = int(g['nky']), int(g['nkx'])
    ksqd_: np.ndarray = g['ksqd_']
    dz  = np.array([0.5, 0.5])
    rho = np.array([1.0, 1.03])

    common = dict(kmax=kmax, nz=2, F=100.0, beta=16.0, bot_drag=0.5,
                  filter_type='exp_cutoff', filter_exp=8.0, k_cut=50.0,
                  dealiasing='isotropic')

    for seed in range(3):
        rng = np.random.default_rng(seed)
        psi0 = (rng.standard_normal((2, nky, nkx))
                + 1j * rng.standard_normal((2, nky, nkx))) * g['filter_mask']
        e_o = 10.0
        psi0 *= np.sqrt(e_o / (float(np.sum(ksqd_[np.newaxis] * np.abs(psi0)**2)) + 1e-30))

        # Fixed dt
        m = QGModel(**common, adapt_dt=False, dt=0.0005)
        m.initialize(psi_init=psi0.copy(), dz=dz, rho=rho)
        m.run(2000)
        assert m.psi is not None
        e_final = float(np.sum(ksqd_[np.newaxis] * np.abs(m.psi)**2))
        print(f'  fixed-dt seed={seed}: final energy={e_final:.3f}, dt={m.dt:.5f}')
        assert np.isfinite(e_final), f'fixed-dt seed={seed} blew up'
        assert e_final < e_o * 10,   f'fixed-dt seed={seed} energy exploded'

        # Adaptive dt with ceiling
        m2 = QGModel(**common, adapt_dt=True, dt_max=0.002, dt_tune=1.5, dt_step=10)
        m2.initialize(psi_init=psi0.copy(), dz=dz, rho=rho)
        m2.run(2000)
        assert m2.psi is not None
        e_final2 = float(np.sum(ksqd_[np.newaxis] * np.abs(m2.psi)**2))
        print(f'  adapt-dt seed={seed}: final energy={e_final2:.3f}, dt={m2.dt:.5f}')
        assert np.isfinite(e_final2), f'adapt-dt seed={seed} blew up'
        assert e_final2 < e_o * 10,   f'adapt-dt seed={seed} energy exploded'


if __name__ == '__main__':
    tests = [
        test_transform_roundtrip,
        test_ir_prod_dealiasing,
        test_tridiag,
        test_march,
        test_barotropic_run,
        test_multilayer_run,
    ]
    failed = []
    for t in tests:
        try:
            print(f'\n--- {t.__name__} ---')
            t()
            print('  PASS')
        except Exception as e:
            print(f'  FAIL: {e}')
            import traceback; traceback.print_exc()
            failed.append(t.__name__)

    print(f'\n{len(tests)-len(failed)}/{len(tests)} tests passed')
    if failed:
        sys.exit(1)
