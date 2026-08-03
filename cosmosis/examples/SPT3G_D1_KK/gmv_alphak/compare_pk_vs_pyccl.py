"""
Compare cosmosis's matter_power_nl (from the test-sampler dump) to pyccl's
matter_power at the same cosmology, at a few representative redshifts.

Reads the dump at $DIR_CHAIN/gmv_alphak_camb_vs_limber/test/.
Produces pk_vs_pyccl.png with per-z ratio panels.
"""

import os
import re
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyccl as ccl


def read_values(save_dir, section):
    o = {}
    path = os.path.join(save_dir, section, "values.txt")
    if not os.path.isfile(path):
        return o
    with open(path) as f:
        for ln in f:
            m = re.match(r"^\s*(\w+)\s*=\s*(\S+)", ln)
            if m:
                try:
                    o[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return o


def load(save_dir, section, key):
    return np.loadtxt(os.path.join(save_dir, section, f"{key}.txt"))


def main():
    dir_chain = os.environ.get(
        "DIR_CHAIN",
        "/lcrc/project/SPT3G/users/ac.yomori/repo/cosmosis-standard-library/output",
    )
    save_dir = os.path.join(dir_chain, "gmv_alphak_camb_vs_limber", "test")
    out_png = os.path.join(
        dir_chain, "gmv_alphak_camb_vs_limber", "pk_vs_pyccl.png"
    )
    if not os.path.isdir(save_dir):
        sys.exit(f"Missing: {save_dir}")

    cos = read_values(save_dir, "cosmological_parameters")
    h = cos["h0"]
    Omega_c = cos["omch2"] / h ** 2
    Omega_b = cos["ombh2"] / h ** 2
    n_s = cos["n_s"]
    A_s = cos["a_s"]
    print(f"cosmology: h={h}, Omega_c={Omega_c:.4f}, Omega_b={Omega_b:.4f}, "
          f"n_s={n_s:.4f}, A_s={A_s:.4e}")

    # cosmosis matter_power_nl
    k_h = load(save_dir, "matter_power_nl", "k_h")       # [h/Mpc]
    z_pk = load(save_dir, "matter_power_nl", "z")
    P_kz = load(save_dir, "matter_power_nl", "p_k")      # shape (n_k, n_z), (Mpc/h)^3
    print(f"matter_power_nl: k_h in [{k_h.min():.2e}, {k_h.max():.2e}]  "
          f"z in [{z_pk.min():.3f}, {z_pk.max():.3f}]  shape={P_kz.shape}")

    # pyccl at same cosmology, same matter power prescription
    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=n_s, A_s=A_s,
        matter_power_spectrum="camb",
        extra_parameters={
            "camb": {"kmax": 10.0, "halofit_version": "mead2020_feedback"}
        },
    )

    z_test = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1000.0]
    z_test = [z for z in z_test if z <= z_pk.max() + 0.01]

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )

    for i, zt in enumerate(z_test):
        j = int(np.argmin(abs(z_pk - zt)))
        P_cs = P_kz[:, j]                                  # (Mpc/h)^3
        # pyccl: nonlin_matter_power takes k in 1/Mpc, returns Mpc^3
        k_Mpc = k_h * h
        P_cc = ccl.nonlin_matter_power(cosmo, k_Mpc, 1.0 / (1 + z_pk[j]))
        P_cc_hless = P_cc * h ** 3                         # back to (Mpc/h)^3

        lab = f"z={z_pk[j]:.2f}"
        line, = axes[0].loglog(k_h, k_h ** 3 * P_cs, lw=1.3, label=f"cosmosis  {lab}")
        axes[0].loglog(k_h, k_h ** 3 * P_cc_hless, lw=1.0, ls="--",
                       color=line.get_color(), label=f"pyccl     {lab}")
        axes[1].semilogx(k_h, P_cs / P_cc_hless, lw=1.3, color=line.get_color(),
                         label=lab)

    axes[0].set_ylabel(r"$k^3 P(k)$  $[(h^{-1}\mathrm{Mpc})^0]$")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(1.0, color="k", lw=0.5, ls=":")
    axes[1].axhspan(0.98, 1.02, color="gray", alpha=0.2)
    axes[1].set_xlabel(r"$k$  [$h/\mathrm{Mpc}$]")
    axes[1].set_ylabel("cosmosis / pyccl")
    axes[1].set_xlim(k_h.min(), k_h.max())
    axes[1].set_ylim(0.5, 1.5)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_png}")

    # Numeric summary
    for zt in z_test:
        j = int(np.argmin(abs(z_pk - zt)))
        k_Mpc = k_h * h
        P_cs = P_kz[:, j]
        P_cc = ccl.nonlin_matter_power(cosmo, k_Mpc, 1.0 / (1 + z_pk[j])) * h ** 3
        r = P_cs / P_cc
        mask = (k_h > 1e-3) & (k_h < 10.0)
        rr = r[mask]
        print(f"  z={z_pk[j]:.2f}: cosmosis/pyccl on 1e-3<k_h<10 "
              f"median={np.median(rr):.4f}  min={rr.min():.4f}  max={rr.max():.4f}")


if __name__ == "__main__":
    main()
