"""
Convert a SAMPLED S_8 into sigma_8 so the CAMB interface can solve for A_s and
compute the (nonlinear) lensing potential spectrum directly.

S_8 == sigma_8 * sqrt(Omega_m / 0.3)   =>   sigma_8 = S_8 / sqrt(Omega_m / 0.3)

Place this module BEFORE camb. The camb interface (camb_interface.py) detects
cosmological_parameters/sigma_8 in the block, runs once with a default A_s,
then recompute_for_sigma8() rescales A_s and recomputes the power spectra
(reusing the transfer function) so the phiphi spectrum is correct for the
target sigma_8. Sampling S_8 (not sigma_8) keeps the prior matched to the
other cosmosis chains.
"""
import numpy as np
from cosmosis.datablock import names

cosmo = names.cosmological_parameters

# Sum(m_nu) [eV] -> omega_nu h^2 (standard, T_nu = (4/11)^1/3 T_cmb)
NEUTRINO_MASS_FAC = 93.14


def setup(options):
    # cosmosis passes this back as `config` to execute(); must be non-None or
    # cosmosis calls execute(block) with one arg.
    return {}


def execute(block, config):
    S8 = block[cosmo, "S_8"]

    # Total matter density today: Omega_m = (omch2 + ombh2 + omnuh2) / h^2.
    # Prefer omega_m already in the block (from the consistency module) so we
    # match cosmosis' own convention; otherwise derive it from the sampled
    # physical densities.
    if block.has_value(cosmo, "omega_m"):
        omega_m = block[cosmo, "omega_m"]
    else:
        h2 = block[cosmo, "h0"] ** 2
        omnuh2 = block.get_double(cosmo, "mnu", 0.06) / NEUTRINO_MASS_FAC
        omega_m = (block[cosmo, "omch2"] + block[cosmo, "ombh2"] + omnuh2) / h2

    block[cosmo, "sigma_8"] = S8 / np.sqrt(omega_m / 0.3)
    return 0


def cleanup(config):
    return 0
