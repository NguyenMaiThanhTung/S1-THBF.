from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StaticTHBFConfig:
    """Static S1 parameters. Defaults follow Table II where applicable."""

    # System dimensions
    Nt: int = 64
    NRF: int = 4
    Ns: int = 4
    K: int = 8
    L: int = 3

    # EM layer: Eq. (3)-(5)
    N_EM: int = 8
    psi_max_deg: float = 45.0
    phi_3db_deg: float = 65.0
    omega_max_db: float = 30.0
    M: int = 360

    # RF layer: Eq. (8)
    B: int = 64
    beta: int = 3

    # Eq. (11), (20), (22)
    snr_db: float = 10.0
    # Implementation normalization only: the paper specifies Pmax/sigma^2.
    sigma2: float = 1.0

    @property
    def Pmax(self) -> float:
        return self.sigma2 * 10.0 ** (self.snr_db / 10.0)

    @property
    def psi_max(self) -> float:
        return float(np.deg2rad(self.psi_max_deg))

    @property
    def phi_3db(self) -> float:
        return float(np.deg2rad(self.phi_3db_deg))
