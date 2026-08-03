"""
cosmosis module: pk_limber_pp

Computes C_L^phi-phi via Limber over the (possibly alphak-modified)
matter_power_nl, using pyccl's angular_cl with a custom Pk2D built from
the cosmosis datablock. Overwrites `cmb_cl/pp` so downstream likelihoods
(candl / spt3g_d1_lensing) see the rescaled lensing-potential spectrum.

Replaces the pair (pk_to_cl_cmbkappa + kappa_to_phi) since cosmosis's own
`project_2d` cmbkappa output has a scale-dependent normalization offset
(~5-13x too large over 100<=L<=3000; verified with a CAMB-vs-pyccl-Limber
test script). pyccl's Limber reproduces CAMB's native `cmb_cl/pp` to ~2%
over the same L range, which is the expected Limber-vs-nonLimber accuracy.

Pipeline placement:

    camb -> alphak -> pk_limber_pp -> spt3g_d1_lensing

Options:
    z_source       (default 1089.0)   CMB source redshift
    ell_min        (default 2.0)
    ell_max        (default 5000.0)
    n_ell          (default 1000)     log-spaced L for Limber evaluation
    output_section (default "cmb_cl")
    output_key     (default "pp")
"""

import numpy as np
import pyccl as ccl
from cosmosis.datablock import option_section


def setup(options):
    z_source = options.get_double(option_section, "z_source", default=1089.0)
    ell_min = options.get_double(option_section, "ell_min", default=2.0)
    ell_max = options.get_double(option_section, "ell_max", default=5000.0)
    n_ell = options.get_int(option_section, "n_ell", default=1000)
    output_section = options.get_string(option_section, "output_section", default="cmb_cl")
    output_key = options.get_string(option_section, "output_key", default="pp")
    # Target ell grid to interpolate onto; defaults to "cmb_cl" (candl expects it).
    target_ell_section = options.get_string(option_section, "target_ell_section", default="cmb_cl")

    L_arr = np.unique(np.round(np.logspace(
        np.log10(max(1.0, ell_min)), np.log10(ell_max), n_ell
    )).astype(int)).astype(float)

    return {
        "z_source": z_source,
        "L_arr": L_arr,
        "output_section": output_section,
        "output_key": output_key,
        "target_ell_section": target_ell_section,
        "cosmo": None,
        "tracer": None,
        "h": None,
    }


def _scalar(block, section, key):
    for k in (key, key.lower(), key.upper()):
        if block.has_value(section, k):
            return block[section, k]
    raise KeyError(f"{section}/{key}")


def _build_cosmo(block):
    sec = "cosmological_parameters"
    h = _scalar(block, sec, "h0")
    ombh2 = _scalar(block, sec, "ombh2")
    omch2 = _scalar(block, sec, "omch2")
    n_s = _scalar(block, sec, "n_s")
    try:
        A_s = _scalar(block, sec, "a_s")
    except KeyError:
        log1e10As = _scalar(block, sec, "log1e10as")
        A_s = np.exp(log1e10As) * 1e-10
    try:
        m_nu = float(_scalar(block, sec, "mnu"))
    except KeyError:
        m_nu = 0.06

    cosmo = ccl.Cosmology(
        Omega_c=omch2 / h**2,
        Omega_b=ombh2 / h**2,
        h=h,
        n_s=n_s,
        A_s=A_s,
        m_nu=m_nu,
        matter_power_spectrum="linear",
    )
    return cosmo, h


def _build_pk2d(block, h):
    k_h = np.asarray(block["matter_power_nl", "k_h"])
    z = np.asarray(block["matter_power_nl", "z"])
    P = np.asarray(block["matter_power_nl", "P_k"])
    k_Mpc = k_h * h
    P_Mpc3 = P / h**3

    a_arr = 1.0 / (1.0 + z)
    isort = np.argsort(a_arr)
    a_arr = a_arr[isort]
    pk_sorted = P_Mpc3[:, isort].T
    pk_sorted = np.clip(pk_sorted, 1e-300, None)

    return ccl.Pk2D(
        a_arr=a_arr, lk_arr=np.log(k_Mpc), pk_arr=np.log(pk_sorted),
        is_logp=True, extrap_order_lok=1, extrap_order_hik=2,
    )


def execute(block, config):
    if config["cosmo"] is None:
        config["cosmo"], config["h"] = _build_cosmo(block)
        # force growth splines to be precomputed (angular_cl needs them)
        _ = config["cosmo"].growth_factor(1.0)
        config["tracer"] = ccl.CMBLensingTracer(
            config["cosmo"], z_source=config["z_source"]
        )
        print(f"[pk_limber_pp] pyccl cosmo + CMBLensingTracer built "
              f"(z_source={config['z_source']}, h={config['h']:.4f})", flush=True)

    pk2d = _build_pk2d(block, config["h"])

    L = config["L_arr"]
    C_kk = ccl.angular_cl(
        config["cosmo"], config["tracer"], config["tracer"], L,
        p_of_k_a=pk2d,
    )
    # Raw C_L^phi-phi = 4 C_L^kappa-kappa / [L(L+1)]^2
    with np.errstate(divide="ignore", invalid="ignore"):
        C_pp_raw = 4.0 * C_kk / (L * (L + 1.0))**2

    # cosmosis's camb_interface.py stores cmb_cl/pp as:
    #   cmb_cl/pp = C_pp_raw * L(L+1) / (2 pi)
    # Match that convention so candl reads a consistent quantity.
    C_pp_stored = C_pp_raw * L * (L + 1.0) / (2.0 * np.pi)

    out_sec = config["output_section"]
    out_key = config["output_key"]
    target_sec = config["target_ell_section"]
    ell_camb = np.asarray(block[target_sec, "ell"], dtype=float)

    good = (L > 0) & np.isfinite(C_pp_stored) & (C_pp_stored > 0)
    log_interp = np.interp(
        np.log(np.clip(ell_camb, 1.0, None)),
        np.log(L[good]), np.log(C_pp_stored[good]),
        left=np.log(C_pp_stored[good][0]),
        right=np.log(C_pp_stored[good][-1]),
    )
    C_pp_camb = np.exp(log_interp)
    C_pp_camb[ell_camb < 1.0] = 0.0

    # Make sure the output section has matching ell (useful when writing to a
    # fresh section for diagnostics/comparison).
    if out_sec != target_sec:
        block[out_sec, "ell"] = ell_camb
    block[out_sec, out_key] = C_pp_camb
    return 0


def cleanup(config):
    return 0
