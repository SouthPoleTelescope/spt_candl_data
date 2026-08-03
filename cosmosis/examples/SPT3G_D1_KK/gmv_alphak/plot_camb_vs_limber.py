"""
Compare CAMB native C_L^phi-phi against pk_limber_pp (pyccl Limber over
matter_power_nl). Both are stored in cosmosis convention
  cmb_cl/pp = C_pp_raw * L(L+1) / (2 pi).

Reads the test-sampler dump at $DIR_CHAIN/gmv_alphak_camb_vs_limber/test/.
Produces camb_vs_limber.png.
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_arr(save_dir, section, key):
    return np.loadtxt(os.path.join(save_dir, section, f"{key}.txt"))


def main():
    dir_chain = os.environ.get(
        "DIR_CHAIN",
        "/lcrc/project/SPT3G/users/ac.yomori/repo/cosmosis-standard-library/output",
    )
    save_dir = os.path.join(dir_chain, "gmv_alphak_camb_vs_limber", "test")
    out_png = os.path.join(dir_chain, "gmv_alphak_camb_vs_limber", "camb_vs_limber.png")
    if not os.path.isdir(save_dir):
        sys.exit(f"Test save dir not found: {save_dir}")

    # Both in cosmosis convention: stored = C_pp_raw * L(L+1) / (2pi)
    L_camb = load_arr(save_dir, "cmb_cl", "ell")
    pp_camb_stored = load_arr(save_dir, "cmb_cl", "pp")

    L_lim = load_arr(save_dir, "cmb_cl_limber", "ell")
    pp_lim_stored = load_arr(save_dir, "cmb_cl_limber", "pp")

    # Convert cosmosis-stored to the physics plot variable [L(L+1)]^2 C_pp / 2pi
    # Stored = C_pp * L(L+1)/(2π), so  Dl = Stored * L(L+1)
    Dl_camb = pp_camb_stored * L_camb * (L_camb + 1)
    Dl_lim  = pp_lim_stored  * L_lim  * (L_lim  + 1)

    # Plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06},
    )

    ax1.plot(L_camb, Dl_camb, lw=1.6, label="CAMB native")
    ax1.plot(L_lim,  Dl_lim,  lw=1.4, ls="--",
             label="clpp_limber (cosmosis matter_power_nl)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(2, 4000)
    ax1.set_ylabel(r"$[L(L+1)]^2\,C_L^{\phi\phi}/2\pi$")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Interpolate pk_limber_pp onto CAMB grid for ratio
    good = (L_lim > 0) & np.isfinite(pp_lim_stored) & (pp_lim_stored > 0)
    log_lim_on_camb = np.interp(
        np.log(np.clip(L_camb, 1.0, None)),
        np.log(L_lim[good]), np.log(pp_lim_stored[good])
    )
    pp_lim_on_camb = np.exp(log_lim_on_camb)
    ratio = pp_lim_on_camb / pp_camb_stored

    ax2.plot(L_camb, ratio, lw=1.4, color="C2")
    ax2.axhline(1.0, color="k", lw=0.7, ls="--")
    ax2.axhspan(0.98, 1.02, color="gray", alpha=0.2, label="±2%")
    ax2.set_xscale("log"); ax2.set_xlim(2, 4000)
    ax2.set_ylim(0.85, 1.15)
    ax2.set_xlabel("$L$")
    ax2.set_ylabel("clpp_limber / CAMB")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_png}")

    mask = (L_camb >= 100) & (L_camb <= 3000)
    r = ratio[mask]
    r = r[np.isfinite(r)]
    if r.size:
        print(
            f"clpp_limber / CAMB over 100<=L<=3000: "
            f"min={r.min():.4f}  max={r.max():.4f}  median={np.median(r):.4f}"
        )


if __name__ == "__main__":
    main()
