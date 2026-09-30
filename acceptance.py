
import numpy as np

from channel import em_configured_channel_direct, steering_vector
from config import StaticTHBFConfig
from em import (
    broadside_pattern,
    build_em_codebook,
    normalized_pattern_peak_db,
    pattern_gain_at,
)
from metrics import gross_spectral_efficiency, sinr
from precoding import rzf_precoder, transmit_power
from rf import build_rf_codebook, rf_precoder


def check_norms_and_phases(cfg):
    em_codebook, _ = build_em_codebook(cfg)
    rf_codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)

    rf_norm_err = float(np.max(np.abs(np.linalg.norm(rf_codebook, axis=1) - 1.0)))
    steering_err = abs(np.linalg.norm(steering_vector(0.37, cfg.Nt)) - 1.0)
    em_energy_err = float(np.max(np.abs(np.sum(em_codebook ** 2, axis=1) - cfg.M)))

    step = 2.0 * np.pi / (2 ** cfg.beta)
    phases = np.angle(rf_codebook * np.sqrt(cfg.Nt))
    phase_grid_err = float(np.max(np.abs(phases / step - np.round(phases / step))))

    return {
        "rf_norm_max_abs_error": rf_norm_err,
        "steering_norm_abs_error": float(steering_err),
        "em_energy_max_abs_error": em_energy_err,
        "phase_grid_max_error": phase_grid_err,
        "allowed_phase_levels": 2 ** cfg.beta,
    }


def check_power_normalization(cfg):
    rng = np.random.default_rng(1)
    H_eff = (rng.normal(size=(cfg.Ns, cfg.NRF)) + 1j * rng.normal(size=(cfg.Ns, cfg.NRF))) / np.sqrt(2)
    codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)
    F_RF = rf_precoder(codebook, np.arange(1, cfg.NRF + 1))
    F_BB, _, _ = rzf_precoder(H_eff, F_RF, cfg.Pmax, cfg.sigma2)
    p = transmit_power(F_RF, F_BB)
    rel = abs(p - cfg.Pmax) / cfg.Pmax
    return {"power": p, "Pmax": cfg.Pmax, "relative_error": float(rel)}


def check_eq14_statistics(cfg, draws=100_000, seed=2):
    rng = np.random.default_rng(seed)
    q = 1
    g = build_em_codebook(cfg)[0][q - 1]
    aods = np.deg2rad(np.array([[-20.0, 3.0, 17.0]]))

    gains = (rng.normal(size=(draws, cfg.L)) + 1j * rng.normal(size=(draws, cfg.L))) / np.sqrt(2.0)
    # Vectorized Eq. (14) for one user's fixed AoDs over many gain draws.
    A = np.stack([steering_vector(float(th), cfg.Nt).conj() for th in aods[0]], axis=0)
    gv = np.array([pattern_gain_at(g, float(th)) for th in aods[0]])
    H = np.sqrt(cfg.Nt / cfg.L) * ((np.conj(gains) * gv[None, :]) @ A)
    empirical = float(np.mean(np.sum(np.abs(H) ** 2, axis=1)))
    theory = float((cfg.Nt / cfg.L) * np.sum(gv ** 2))
    rel = abs(empirical - theory) / theory
    return {"empirical": empirical, "theory": theory, "relative_error": float(rel)}


def check_pattern_peak(cfg):
    g, _, _ = broadside_pattern(cfg.phi_3db, cfg.omega_max_db, cfg.M)
    peak = normalized_pattern_peak_db(g)
    return {"peak_db": peak}


def check_no_em_reproducibility(cfg, seed=3):
    def one_run():
        rng = np.random.default_rng(seed)
        gains = (rng.normal(size=(cfg.K, cfg.L)) + 1j * rng.normal(size=(cfg.K, cfg.L))) / np.sqrt(2.0)
        # Fixed static AoDs generated reproducibly only for this acceptance check.
        aods = rng.uniform(-np.pi / 3, np.pi / 3, size=(cfg.K, cfg.L))
        g0, _, _ = broadside_pattern(cfg.phi_3db, cfg.omega_max_db, cfg.M)
        H = em_configured_channel_direct(gains, aods, g0, cfg.Nt)
        return gains, aods, H

    a = one_run()
    b = one_run()
    return {"bitwise_equal": all(np.array_equal(x, y) for x, y in zip(a, b))}


def main():
    cfg = StaticTHBFConfig()
    results = {
        "norms_and_phases": check_norms_and_phases(cfg),
        "power_normalization": check_power_normalization(cfg),
        "eq14_statistics": check_eq14_statistics(cfg),
        "pattern_peak": check_pattern_peak(cfg),
        "no_em_reproducibility": check_no_em_reproducibility(cfg),
        
    }

    for name, value in results.items():
        print(f"[{name}]")
        print(value)
        print()

    assert results["norms_and_phases"]["rf_norm_max_abs_error"] < 1e-12
    assert results["norms_and_phases"]["steering_norm_abs_error"] < 1e-12
    assert results["norms_and_phases"]["em_energy_max_abs_error"] < 1e-9
    assert results["norms_and_phases"]["phase_grid_max_error"] < 1e-12
    assert results["power_normalization"]["relative_error"] < 1e-9
    assert results["eq14_statistics"]["relative_error"] < 0.01
    assert abs(results["pattern_peak"]["peak_db"] - 7.1) < 0.2
    assert results["no_em_reproducibility"]["bitwise_equal"]


if __name__ == "__main__":
    main()
