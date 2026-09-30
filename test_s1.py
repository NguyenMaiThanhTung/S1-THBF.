import numpy as np

from channel import em_configured_channel_direct, steering_vector
from config import StaticTHBFConfig
from em import build_em_codebook, pattern_gain_at
from metrics import gross_spectral_efficiency, sinr
from precoding import rzf_precoder, transmit_power
from rf import build_rf_codebook, rf_precoder
from scheduling import greedy_user_selection


def test_em_rf_steering_norms():
    cfg = StaticTHBFConfig()
    em_codebook, _ = build_em_codebook(cfg)
    rf_codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)
    assert np.allclose(np.sum(em_codebook ** 2, axis=1), cfg.M, atol=1e-10)
    assert np.allclose(np.linalg.norm(rf_codebook, axis=1), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(steering_vector(0.2, cfg.Nt)), 1.0, atol=1e-12)


def test_eq14_matches_kronecker_form():
    # Eq. (6)/(13) appear only in this unit test, as required by S1.
    cfg = StaticTHBFConfig(Nt=4, L=2, M=16, N_EM=2)
    rng = np.random.default_rng(0)
    gains = (rng.normal(size=(1, cfg.L)) + 1j * rng.normal(size=(1, cfg.L))) / np.sqrt(2)
    aods = np.array([[0.13, -0.41]])
    g = build_em_codebook(cfg)[0][0]

    direct = em_configured_channel_direct(gains, aods, g, cfg.Nt)[0]

    # Eq. (13): h_k in C^(M Nt), using e_i(theta).
    h = np.zeros(cfg.M * cfg.Nt, dtype=complex)
    for ell in range(cfg.L):
        theta = float(aods[0, ell])
        # Recover the sampled bin via g(theta); locate same index robustly.
        from em import nearest_angular_index
        idx = nearest_angular_index(theta, cfg.M)
        e = np.zeros(cfg.M)
        e[idx] = 1.0
        h += np.sqrt(cfg.Nt / cfg.L) * gains[0, ell] * np.kron(steering_vector(theta, cfg.Nt), e)

    # Eq. (6): F_EM = I_Nt kron g_q.
    F_EM = np.kron(np.eye(cfg.Nt), g.reshape(-1, 1))
    kron_result = h.conj() @ F_EM
    assert np.allclose(direct, kron_result, atol=1e-12)


def test_rzf_power_and_metrics():
    cfg = StaticTHBFConfig(Nt=8, NRF=2, Ns=2, B=8)
    rng = np.random.default_rng(2)
    H = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / np.sqrt(2)
    codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)
    F_RF = rf_precoder(codebook, [1, 3])
    F_BB, _, _ = rzf_precoder(H, F_RF, cfg.Pmax, cfg.sigma2)
    assert abs(transmit_power(F_RF, F_BB) - cfg.Pmax) / cfg.Pmax < 1e-9
    gamma = sinr(H, F_BB, cfg.sigma2)
    assert np.all(gamma >= 0)
    assert gross_spectral_efficiency(gamma) >= 0


def test_greedy_scheduler_is_deterministic():
    cfg = StaticTHBFConfig(Nt=8, NRF=2, Ns=2, K=4, B=8)
    rng = np.random.default_rng(4)
    H_all = (rng.normal(size=(cfg.K, cfg.NRF)) + 1j * rng.normal(size=(cfg.K, cfg.NRF))) / np.sqrt(2)
    codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)
    F_RF = rf_precoder(codebook, [1, 3])
    a = greedy_user_selection(H_all, F_RF, cfg.Ns, cfg.Pmax, cfg.sigma2)
    b = greedy_user_selection(H_all, F_RF, cfg.Ns, cfg.Pmax, cfg.sigma2)
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert np.array_equal(a[2], b[2])
