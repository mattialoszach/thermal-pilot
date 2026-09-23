import numpy as np
from scipy.signal import cont2discrete

from thermalpilot.config import BuildingParameters
from thermalpilot.model import RCModel


def test_continuous_matrices_match_energy_balance_coefficients() -> None:
    p = BuildingParameters()
    a_c, b_c, e_c = RCModel(p).continuous_matrices()
    assert np.isclose(a_c[0, 1], 1 / (p.r_air_wall_k_per_w * p.c_air_j_per_k))
    assert np.isclose(a_c[0, 2], 1 / (p.r_air_mass_k_per_w * p.c_air_j_per_k))
    assert np.isclose(b_c[0, 0], p.eta_heater / p.c_air_j_per_k)
    assert np.isclose(e_c[1, 1], p.alpha_wall_m2 / p.c_wall_j_per_k)
    assert np.allclose(b_c[1:], 0.0)


def test_exact_zoh_matches_scipy_reference() -> None:
    model = RCModel()
    a_c, b_c, e_c = model.continuous_matrices()
    input_matrix = np.hstack([b_c, e_c])
    a_ref, be_ref, _, _, _ = cont2discrete(
        (a_c, input_matrix, np.eye(3), np.zeros((3, 3))), 900.0, method="zoh"
    )
    discrete = model.discretize(900.0)
    assert np.allclose(discrete.A, a_ref, atol=1e-12)
    assert np.allclose(np.hstack([discrete.B, discrete.E]), be_ref, atol=1e-12)


def test_uniform_temperature_is_an_equilibrium_without_heat_or_sun() -> None:
    discrete = RCModel().discretize(900.0)
    state = np.full(3, 12.5)
    next_state = discrete.step(state, 0.0, np.array([12.5, 0.0]))
    assert np.allclose(next_state, state, atol=1e-12)


def test_default_discrete_model_is_stable_and_physically_plausible() -> None:
    model = RCModel()
    report = model.plausibility_report()
    assert report["discrete_stable"] is True
    assert 30 < report["heat_loss_coefficient_w_per_k"] < 100
    assert 0.2 < report["specific_heat_loss_w_per_m2k"] < 1.5
