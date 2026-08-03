"""
cosmosis module: kappa_to_phi

Converts the CMB-lensing convergence power spectrum C_L^kk (produced by
pk_to_cl with a cmbkappa-cmbkappa tracer pair) to the lensing-potential
spectrum C_L^phi-phi via

    C_L^phi-phi = 4 C_L^kk / [L(L+1)]^2          (since kappa = L(L+1)/2 * phi)

and overwrites `cmb_cl/pp` on the ell grid that the candl likelihood reads
from the existing CAMB output.

Chain placement:

    camb  ->  alphak  ->  pk_to_cl_cmbkappa  ->  kappa_to_phi  ->  spt3g_d1_lensing

This is the cheap way to propagate alpha(k) into SPT-3G CMB-lensing data:
alphak modifies matter_power_nl, pk_to_cl re-Limbers that into C_L^kk, this
module converts to pp at the ells candl expects.

Options:
    input_section   (default "cmbkappa_cl_cmb")  section written by pk_to_cl
    input_key       (default "bin_1_1")
    output_section  (default "cmb_cl")           section candl reads
    output_key      (default "pp")
"""

import numpy as np
from cosmosis.datablock import option_section


def setup(options):
    return {
        "input_section": options.get_string(option_section, "input_section", default="cmbkappa_cl_cmb"),
        "input_key":     options.get_string(option_section, "input_key",     default="bin_1_1"),
        "output_section": options.get_string(option_section, "output_section", default="cmb_cl"),
        "output_key":    options.get_string(option_section, "output_key",    default="pp"),
    }


def execute(block, config):
    in_sec  = config["input_section"]
    in_key  = config["input_key"]
    out_sec = config["output_section"]
    out_key = config["output_key"]

    ell_kk = np.asarray(block[in_sec, "ell"], dtype=float)
    C_kk   = np.asarray(block[in_sec, in_key], dtype=float)

    # C_pp(L) = 4 C_kk(L) / [L(L+1)]^2
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 4.0 / (ell_kk * (ell_kk + 1.0))**2
    C_pp_kk_grid = np.where(np.isfinite(factor), C_kk * factor, 0.0)

    # Interpolate onto the CAMB ell grid that candl expects, in log space
    # (smoother and monotone) clipped to the pk_to_cl support.
    ell_camb = np.asarray(block[out_sec, "ell"], dtype=float)
    good = (ell_kk > 0) & np.isfinite(C_pp_kk_grid) & (C_pp_kk_grid > 0)
    logL_src = np.log(ell_kk[good])
    logC_src = np.log(C_pp_kk_grid[good])

    logL_tgt = np.log(np.clip(ell_camb, 1.0, None))
    logC_tgt = np.interp(logL_tgt, logL_src, logC_src,
                         left=logC_src[0], right=logC_src[-1])
    C_pp_camb = np.exp(logC_tgt)
    C_pp_camb[ell_camb < 1.0] = 0.0

    block[out_sec, out_key] = C_pp_camb
    return 0


def cleanup(config):
    return 0
