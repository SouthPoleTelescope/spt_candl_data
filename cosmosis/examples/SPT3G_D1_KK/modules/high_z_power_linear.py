"""
cosmosis module: high_z_power_linear

Extends matter_power_{nl,lin} above CAMB's zmax using matter-dominated
linear growth:

    P(k, z > z_break) = P(k, z_break) * [(1+z_break) / (1+z)]^2

This is <1% accurate for z >~ 2 in LCDM where matter domination holds,
and avoids CAMB's mead2020_feedback misbehaviour at high z that shows up
as P(k,z) NOT decaying for z > ~5. For CMB-lensing Limber integrals the
integrand at z > z_break is essentially linear anyway, so using the
matter-dominated growth law for the extension is appropriate.

Run AFTER camb and BEFORE clpp_limber / pk_to_cl_cmbkappa. The k axis is
unchanged; only new z columns are appended.

Options:
    sections  (default "matter_power_nl matter_power_lin")
    zmax      (default 1200.0)   target zmax to extend to
    nz_log    (default 100)      number of log(1+z)-spaced points to append
    verbose   (default F)
"""

import numpy as np
from cosmosis.datablock import option_section


def setup(options):
    return {
        "sections": options.get_string(
            option_section, "sections",
            default="matter_power_nl matter_power_lin").split(),
        "zmax": options.get_double(option_section, "zmax", default=1200.0),
        "nz_log": options.get_int(option_section, "nz_log", default=100),
        "verbose": options.get_bool(option_section, "verbose", default=False),
    }


def _extend(block, section, zmax, nz_log, verbose):
    if not block.has_section(section):
        if verbose:
            print(f"[high_z_power_linear] skip {section}: not in block", flush=True)
        return
    # cosmosis CAMB order: (k_h, z), P shape = (n_k, n_z)
    k, z_cur, P = block.get_grid(section, "k_h", "z", "p_k")
    z_break = float(z_cur.max())
    if zmax <= z_break:
        if verbose:
            print(f"[high_z_power_linear] {section}: zmax={zmax} "
                  f"<= z_break={z_break}, nothing to do", flush=True)
        return

    # log(1+z) spaced from just above z_break up to zmax
    log1p_new = np.linspace(
        np.log1p(z_break) + 1e-4, np.log1p(zmax), nz_log
    )
    z_new = np.expm1(log1p_new)
    z_new = z_new[z_new > z_break]  # safety

    # Matter-dominated growth: D(z)/D(z_break) = (1+z_break)/(1+z)
    P_break = P[:, -1]                              # shape (n_k,)
    factor2 = ((1.0 + z_break) / (1.0 + z_new)) ** 2  # shape (n_new,)
    P_new = P_break[:, None] * factor2[None, :]       # (n_k, n_new)

    z_total = np.concatenate([z_cur, z_new])
    P_total = np.concatenate([P, P_new], axis=1)

    block.replace_grid(section, "k_h", k, "z", z_total, "p_k", P_total)

    if verbose:
        print(f"[high_z_power_linear] extended {section}: "
              f"z_break={z_break:.3f} -> zmax={zmax:.1f}  "
              f"({nz_log} log(1+z)-spaced points; new total n_z={P_total.shape[1]})",
              flush=True)


def execute(block, config):
    for section in config["sections"]:
        _extend(block, section, config["zmax"], config["nz_log"], config["verbose"])
    return 0


def cleanup(config):
    return 0
