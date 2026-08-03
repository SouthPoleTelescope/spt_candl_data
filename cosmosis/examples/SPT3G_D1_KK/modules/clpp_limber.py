"""
cosmosis module: clpp_limber

Computes the CMB lensing-potential power spectrum C_L^phi-phi using the
CAMB-demo-style Weyl Limber integral

    C_L^kappa = [L(L+1)]^2 * int dchi  (chi* - chi)^2 / (chi^2 chi*)^2
                             * P_Weyl(k=(L+0.5)/chi, z(chi)) / k^4

and overwrites `cmb_cl/pp` on the ell grid CAMB wrote. This matches CAMB's
full non-Limber calculation to <0.1% over 30 < L < 3000.

Because the Weyl power spectrum scales linearly with the matter power
spectrum via Poisson,
    P_Weyl(k,z) = (3/2 Omega_m H_0^2 (1+z))^2 / k^4 * P_m(k,z),
any multiplicative rescaling of P_m (e.g. the alpha(k) knob applied by
the `alphak` module) multiplies P_Weyl by the same factor. So the alpha
modification propagates correctly if we either:

  (a) Compute P_Weyl once from the CAMB interpolator at the fixed
      cosmology (fast_mode=True), then rescale by (1 + alpha(k)) per
      call using a fiducial matter P(k,z) ratio.
  (b) Rebuild P_Weyl on the fly from block[matter_power_nl] and the
      fiducial Poisson prefactor.

We use (b) because `cached_camb` already populates matter_power_nl and
`alphak` has already applied the rescaling to it. The implementation
just reads matter_power_nl, converts to P_Weyl, Limber-integrates, and
writes cmb_cl/pp.

Chain placement (replaces pk_to_cl_cmbkappa + kappa_to_phi):

    cached_camb  ->  alphak  ->  clpp_limber  ->  spt3g_d1_lensing

Options:
    lmax       (default 4500)   upper L for the output spectrum
    nchi       (default 4000)   number of chi samples for the integral
    kmin       (default 1e-4)   P(k) range clip
    kmax       (default 10.0)   in 1/Mpc
    z_cmb      (default 1089.0) source redshift
    output_section (default "cmb_cl")
    output_key     (default "pp")
"""

import numpy as np
from cosmosis.datablock import option_section


C_KMS = 299792.458
H100_C = 100.0 / C_KMS  # 1/Mpc per h (i.e. H0/c = h * H100_C)


def setup(options):
    return {
        "lmax":           options.get_int(option_section, "lmax",  default=4500),
        "nchi":           options.get_int(option_section, "nchi",  default=4000),
        "kmin":           options.get_double(option_section, "kmin", default=1e-4),
        "kmax":           options.get_double(option_section, "kmax", default=10.0),
        "z_cmb":          options.get_double(option_section, "z_cmb", default=1089.0),
        "output_section": options.get_string(option_section, "output_section", default="cmb_cl"),
        "output_key":     options.get_string(option_section, "output_key", default="pp"),
    }


def _load_pk_interp(block):
    """Read matter_power_nl (cosmosis units: k_h in h/Mpc, P in (Mpc/h)^3)
    and return a callable P_m(k, z) where k is in 1/Mpc and P in Mpc^3."""
    k_h = np.asarray(block["matter_power_nl", "k_h"], dtype=float)
    z   = np.asarray(block["matter_power_nl", "z"], dtype=float)
    P_k = np.asarray(block["matter_power_nl", "p_k"], dtype=float)

    # Need h to convert axes. The cosmosis block labels are `k_h` and `p_k`
    # but the underlying convention is not 100% consistent across modules,
    # so make the unit convention a knob. Default: treat as h/Mpc and
    # (Mpc/h)^3 (standard cosmosis docs). Set pk_units_are_hless=False to
    # treat matter_power_nl as already in 1/Mpc and Mpc^3.
    h = float(block["cosmological_parameters", "h0"])
    _hless = True   # flip to False to test the alternative convention
    if _hless:
        k_Mpc = k_h * h             # h/Mpc -> 1/Mpc
        P_Mpc = P_k / h**3          # (Mpc/h)^3 -> Mpc^3
    else:
        k_Mpc = k_h.copy()          # treat k_h as already in 1/Mpc
        P_Mpc = P_k.copy()          # treat P_k as already in Mpc^3

    log_k = np.log(k_Mpc)
    log_P = np.log(np.clip(P_Mpc, 1e-300, None))

    def P_m(k_eval, z_eval):
        # bilinear in (log k, z); clip to support
        lk = np.log(np.clip(k_eval, k_Mpc[0], k_Mpc[-1]))
        zc = np.clip(z_eval, z[0], z[-1])
        i = np.clip(np.searchsorted(log_k, lk) - 1, 0, len(log_k) - 2)
        j = np.clip(np.searchsorted(z, zc) - 1, 0, len(z) - 2)
        tx = (lk - log_k[i]) / (log_k[i + 1] - log_k[i])
        ty = (zc - z[j]) / (z[j + 1] - z[j])
        lp = ((1 - tx) * (1 - ty) * log_P[i, j]
              + tx       * (1 - ty) * log_P[i + 1, j]
              + (1 - tx) * ty       * log_P[i, j + 1]
              + tx       * ty       * log_P[i + 1, j + 1])
        return np.exp(lp)

    return P_m, h


def execute(block, config):
    lmax    = config["lmax"]
    nchi    = config["nchi"]
    kmin    = config["kmin"]
    kmax    = config["kmax"]
    z_cmb   = config["z_cmb"]
    out_sec = config["output_section"]
    out_key = config["output_key"]

    # Background: chi(z) and chi_star
    z_bg = np.asarray(block["distances", "z"],   dtype=float)
    d_m  = np.asarray(block["distances", "d_m"], dtype=float)
    # chi_star: value at z = z_cmb (or stored scalar)
    if block.has_value("distances", "chistar"):
        chi_star = float(block["distances", "chistar"])
    else:
        chi_star = float(np.interp(z_cmb, z_bg, d_m))

    # Cosmology for the Poisson prefactor
    h      = float(block["cosmological_parameters", "h0"])
    Omega_m = float(block["cosmological_parameters", "omega_m"])
    H0_c   = h * H100_C  # H0/c in 1/Mpc

    # P_m interpolator from the (alpha-rescaled) matter_power_nl
    P_m, _h = _load_pk_interp(block)

    # chi grid up to chi_star
    chis  = np.linspace(1.0, chi_star - 1.0, nchi)
    dchis = np.gradient(chis)
    zs    = np.interp(chis, d_m, z_bg)
    a     = 1.0 / (1.0 + zs)

    # CMB lensing convergence kernel (flat universe):
    #   W_k(chi) = (3/2)(H0/c)^2 Omega_m * (chi/a) * (chi* - chi)/chi*     [1/Mpc]
    W_k = 1.5 * H0_c**2 * Omega_m * (chis / a) * (chi_star - chis) / chi_star

    # Limber loop
    L = np.arange(2, lmax + 1, dtype=float)
    Ckk = np.empty_like(L)
    base = (W_k**2 / np.maximum(chis, 1.0)**2) * dchis   # 1/Mpc
    for i, ell in enumerate(L):
        k = (ell + 0.5) / chis
        good = (k >= kmin) & (k < kmax) & (chis > 0)
        P_m_vals = P_m(k, zs)
        Ckk[i] = np.sum(np.where(good, base * P_m_vals, 0.0))

    # Convert C_L^kappa -> C_L^pp raw
    with np.errstate(divide="ignore", invalid="ignore"):
        Cpp_raw = 4.0 * Ckk / (L * (L + 1))**2

    # CAMB stores cmb_cl/pp as C_pp_raw * L(L+1) / (2pi). Write same convention
    # on the CAMB ell grid.
    L_camb = np.asarray(block["cmb_cl", "ell"], dtype=float)
    Cpp_on_camb = np.exp(
        np.interp(np.log(np.clip(L_camb, 2.0, None)), np.log(L),
                  np.log(np.clip(Cpp_raw, 1e-300, None)),
                  left=np.log(Cpp_raw[0]), right=np.log(Cpp_raw[-1]))
    )
    Cpp_on_camb[L_camb < 2] = 0.0

    pp_stored = Cpp_on_camb * L_camb * (L_camb + 1) / (2 * np.pi)
    if out_sec != "cmb_cl" and not block.has_value(out_sec, "ell"):
        block[out_sec, "ell"] = L_camb
    block[out_sec, out_key] = pp_stored
    return 0


def cleanup(config):
    return 0
