"""
Standalone Limber C_L^phi-phi vs CAMB native, against an existing cosmosis
test-sampler dump. Tests two unit conventions for matter_power_nl:

  A. "hless"   : k_h in h/Mpc, P in (Mpc/h)^3  (standard cosmosis docs)
  B. "raw"     : k_h already in 1/Mpc, P in Mpc^3 (no h conversion)

Prerequisite: run the test sampler first, e.g.
    cosmosis cosmosis_files/gmv_alphak/gmv_test_camb_vs_limber.ini

Make sure [camb] has zmax >> 3 (matter_power_nl extended to CMB z) or
the high-z kernel contribution will be wrong irrespective of units.
"""

import os
import re
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


C_KMS = 299792.458


def read_values(save_dir, section):
    out = {}
    path = os.path.join(save_dir, section, "values.txt")
    if os.path.isfile(path):
        with open(path) as f:
            for ln in f:
                m = re.match(r"^\s*(\w+)\s*=\s*(\S+)", ln)
                if m:
                    try:
                        out[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
    return out


def load(save_dir, section, key):
    return np.loadtxt(os.path.join(save_dir, section, f"{key}.txt"))


def make_pk_interp(k_axis, z_axis, P_grid):
    """Bilinear in (log k, z) with flat clip at boundaries."""
    log_k = np.log(k_axis)
    log_P = np.log(np.clip(P_grid, 1e-300, None))

    def P_m(k, z):
        lk = np.log(np.clip(k, k_axis[0], k_axis[-1]))
        zc = np.clip(z, z_axis[0], z_axis[-1])
        i = np.clip(np.searchsorted(log_k, lk) - 1, 0, len(log_k) - 2)
        j = np.clip(np.searchsorted(z_axis, zc) - 1, 0, len(z_axis) - 2)
        tx = (lk - log_k[i]) / (log_k[i + 1] - log_k[i])
        ty = (zc - z_axis[j]) / (z_axis[j + 1] - z_axis[j])
        return np.exp(
            (1 - tx) * (1 - ty) * log_P[i, j]
            + tx * (1 - ty) * log_P[i + 1, j]
            + (1 - tx) * ty * log_P[i, j + 1]
            + tx * ty * log_P[i + 1, j + 1]
        )

    return P_m


def limber_cpp(L, P_m, chi_arr, z_arr, chi_star, Omega_m, h,
               k_in_1_over_Mpc=True):
    """
    C_L^kappa-kappa = int_0^{chi*} dchi W_k^2 / chi^2 * P_m(k=(L+0.5)/chi, z)
    W_k = (3/2) (H0/c)^2 Omega_m * chi/a * (chi* - chi)/chi*.

    If k_in_1_over_Mpc: P_m expects k in 1/Mpc and returns P in Mpc^3.
    Else: P_m expects k in h/Mpc and returns P in (Mpc/h)^3 — we then
    rescale via h^3 so C_L comes out in dimensionless units.
    """
    H0_c = h * 100.0 / C_KMS
    prefac = 1.5 * H0_c ** 2 * Omega_m
    a = 1.0 / (1.0 + z_arr)
    W_k = prefac * chi_arr / a * (chi_star - chi_arr) / chi_star
    dchi = np.gradient(chi_arr)
    base = (W_k ** 2 / np.maximum(chi_arr, 1.0) ** 2) * dchi

    Ckk = np.empty_like(L, dtype=float)
    for i, Li in enumerate(L):
        if k_in_1_over_Mpc:
            k = (Li + 0.5) / chi_arr
        else:
            k = (Li + 0.5) / chi_arr / h    # chi physical, want k in h/Mpc
        P = P_m(k, z_arr)
        if not k_in_1_over_Mpc:
            P = P / h ** 3                  # (Mpc/h)^3 -> Mpc^3
        Ckk[i] = np.sum(base * P)

    with np.errstate(divide="ignore", invalid="ignore"):
        Cpp = 4.0 * Ckk / (L * (L + 1)) ** 2
    return Cpp, Ckk


def main():
    dir_chain = os.environ.get(
        "DIR_CHAIN",
        "/lcrc/project/SPT3G/users/ac.yomori/repo/cosmosis-standard-library/output",
    )
    save_dir = os.path.join(dir_chain, "gmv_alphak_camb_vs_limber", "test")
    out_png = os.path.join(
        dir_chain, "gmv_alphak_camb_vs_limber", "limber_conventions.png"
    )
    if not os.path.isdir(save_dir):
        sys.exit(f"Test save dir not found: {save_dir}. "
                 f"Run the test sampler first.")

    cos = read_values(save_dir, "cosmological_parameters")
    dists_scalars = read_values(save_dir, "distances")
    h = cos["h0"]
    Omega_m = cos["omega_m"]

    k_h = load(save_dir, "matter_power_nl", "k_h")
    z_pk = load(save_dir, "matter_power_nl", "z")
    P_k = load(save_dir, "matter_power_nl", "p_k")
    print(f"matter_power_nl: k_h in [{k_h.min():.2e}, {k_h.max():.2e}],  "
          f"z in [{z_pk.min():.3f}, {z_pk.max():.3f}]")

    z_bg = load(save_dir, "distances", "z")
    d_m = load(save_dir, "distances", "d_m")
    chi_star = float(dists_scalars.get("chistar",
                                       np.interp(1089.0, z_bg, d_m)))
    print(f"chi_star = {chi_star:.1f} Mpc,  h = {h},  Omega_m = {Omega_m}")

    L_camb = load(save_dir, "cmb_cl", "ell")
    pp_stored = load(save_dir, "cmb_cl", "pp")
    Cpp_camb = pp_stored * 2 * np.pi / (L_camb * (L_camb + 1))

    chi_arr = np.linspace(1.0, chi_star - 1.0, 4000)
    z_arr = np.interp(chi_arr, d_m, z_bg)

    L = np.logspace(np.log10(2.0), np.log10(4000), 200)

    # Convention A: "hless"   (k_h in h/Mpc, P in (Mpc/h)^3)
    k_axis_A = k_h * h
    P_grid_A = P_k / h ** 3
    P_m_A = make_pk_interp(k_axis_A, z_pk, P_grid_A)
    Cpp_A, _ = limber_cpp(L, P_m_A, chi_arr, z_arr, chi_star, Omega_m, h,
                          k_in_1_over_Mpc=True)

    # Convention B: "raw"     (k_h is already 1/Mpc, P is Mpc^3)
    P_m_B = make_pk_interp(k_h, z_pk, P_k)
    Cpp_B, _ = limber_cpp(L, P_m_B, chi_arr, z_arr, chi_star, Omega_m, h,
                          k_in_1_over_Mpc=True)

    def ratio_on_camb(L_src, Cpp_src):
        gp = (L_src > 0) & np.isfinite(Cpp_src) & (Cpp_src > 0)
        return np.exp(
            np.interp(np.log(L_camb.clip(1, None)),
                      np.log(L_src[gp]), np.log(Cpp_src[gp]))
        ) / Cpp_camb

    rA = ratio_on_camb(L, Cpp_A)
    rB = ratio_on_camb(L, Cpp_B)

    # --- report ---
    print(f"\n{'L':>5} {'A (h-convert)':>15} {'B (no h)':>12}")
    for Ltg in (50, 100, 200, 500, 1000, 2000, 3000):
        i = np.argmin(abs(L_camb - Ltg))
        print(f"{L_camb[i]:5.0f} {rA[i]:15.3f} {rB[i]:12.3f}")

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06},
    )

    def dl(L, C):
        return (L * (L + 1)) ** 2 * C / (2 * np.pi)

    ax1.plot(L_camb, dl(L_camb, Cpp_camb), lw=1.6, label="CAMB native")
    ax1.plot(L, dl(L, Cpp_A), lw=1.3, ls="--",
             label="Conv A: k_h=h/Mpc, P=(Mpc/h)^3")
    ax1.plot(L, dl(L, Cpp_B), lw=1.3, ls=":",
             label="Conv B: k_h=1/Mpc, P=Mpc^3")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(2, 4000)
    ax1.set_ylabel(r"$[L(L+1)]^2\,C_L^{\phi\phi}/2\pi$")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(L_camb, rA, lw=1.3, ls="--", label="A / CAMB")
    ax2.plot(L_camb, rB, lw=1.3, ls=":",  label="B / CAMB")
    ax2.axhline(1.0, color="k", lw=0.7, ls="--")
    ax2.axhspan(0.98, 1.02, color="gray", alpha=0.2, label="±2%")
    ax2.set_xscale("log"); ax2.set_xlim(2, 4000)
    ax2.set_xlabel("$L$"); ax2.set_ylabel("ratio to CAMB")
    ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
