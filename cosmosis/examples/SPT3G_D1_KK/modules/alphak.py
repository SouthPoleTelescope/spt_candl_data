"""
cosmosis module: alphak

Rescales the matter power spectrum P(k,z) by a piecewise-linear-in-lnk
modulation (1 + alpha(k)), following Doux & Karwal (2025). The alpha values
at nk log-spaced k-bin centres are sampled as free parameters; they are
read from the values file in section `param_section` as
`{param_prefix}0`, `{param_prefix}1`, ..., `{param_prefix}{nk-1}`.

Downstream modules that re-read `matter_power_nl` / `matter_power_lin`
(e.g. project_2d / pk_to_cl) will automatically see the rescaled P(k,z).

If `save_alpha_lin = T`, the module also writes a derived alpha_lin(k),
defined as the data-preferred rescaling relative to the LINEAR fiducial
at z_pivot:
    alpha_lin_i = (1 + alpha_i) * P_nl_fid(k_i, z_pivot) / P_lin_fid(k_i, z_pivot) - 1
It's captured at each sample BEFORE matter_power_nl is overwritten. You
can then plot both "deviation vs nonlinear" and "deviation vs linear"
from a single chain without rerunning.

Caveat for CMB lensing: CAMB writes `cmb_cl/pp` directly from its own
internal P(k,z) calculation. Modifying `matter_power_nl` here does NOT
update `cmb_cl/pp`. If you want SPT-3G / Planck CMB-lensing likelihoods
to respond to alpha(k), recompute Cl^phi-phi via Limber from the modified
matter_power_nl in a follow-up module (not included here).
"""

import numpy as np
from cosmosis.datablock import option_section, names


def setup(options):
    nk = options.get_int(option_section, "nk", default=24)
    k_min = options.get_double(option_section, "k_min")
    k_max = options.get_double(option_section, "k_max")

    log_edges = np.linspace(np.log(k_min), np.log(k_max), nk + 1)
    log_k_bins = 0.5 * (log_edges[1:] + log_edges[:-1])
    k_bins = np.exp(log_k_bins)
    dlnk = log_k_bins[1] - log_k_bins[0]

    config = {
        "nk": nk,
        "k_bins": k_bins,
        "log_k_bins": log_k_bins,
        "dlnk": dlnk,
        "param_section": options.get_string(
            option_section, "param_section", default="pk_alpha_parameters"
        ),
        "param_prefix": options.get_string(
            option_section, "param_prefix", default="alpha_"
        ),
        "section_nl": options.get_string(
            option_section, "section_nl", default="matter_power_nl"
        ),
        "apply_to_nl": options.get_bool(option_section, "apply_to_nl", default=True),
        "apply_to_lin": options.get_bool(option_section, "apply_to_lin", default=False),
        "section_lin": options.get_string(
            option_section, "section_lin", default="matter_power_lin"
        ),
        "extrap": options.get_string(option_section, "extrap", default="clip"),
        "smoothness_sigma": options.get_double(
            option_section, "smoothness_sigma", default=0.3 * dlnk
        ),
        "smoothness_like_name": options.get_string(
            option_section, "smoothness_like_name", default="alphak_smoothness"
        ),
        "save_alpha_lin": options.get_bool(
            option_section, "save_alpha_lin", default=False
        ),
        "z_pivot": options.get_double(option_section, "z_pivot", default=0.0),
        "derived_section": options.get_string(
            option_section, "derived_section", default="pk_alpha_derived"
        ),
    }

    if config["extrap"] not in ("clip", "zero"):
        raise ValueError(f"alphak: extrap must be 'clip' or 'zero', got {config['extrap']}")

    return config


def _read_alpha(block, config):
    section = config["param_section"]
    prefix = config["param_prefix"]
    return np.array(
        [block[section, f"{prefix}{i}"] for i in range(config["nk"])]
    )


def _alpha_on_grid(k_grid, config, alpha):
    log_k = np.log(k_grid)
    if config["extrap"] == "zero":
        return np.interp(log_k, config["log_k_bins"], alpha, left=0.0, right=0.0)
    return np.interp(log_k, config["log_k_bins"], alpha)


def _rescale_section(block, section, alpha_k):
    k, z, P = block.get_grid(section, "k_h", "z", "P_k")
    P_new = P * (1.0 + alpha_k)[:, None]
    block.replace_grid(section, "k_h", k, "z", z, "P_k", P_new)
    return k


def _P_at_bins(block, section, log_k_bins, z_pivot):
    k, z, P = block.get_grid(section, "k_h", "z", "P_k")
    iz = int(np.argmin(np.abs(z - z_pivot)))
    return np.interp(log_k_bins, np.log(k), P[:, iz])


def execute(block, config):
    alpha = _read_alpha(block, config)

    if config["save_alpha_lin"]:
        p_nl = _P_at_bins(block, config["section_nl"], config["log_k_bins"], config["z_pivot"])
        p_lin = _P_at_bins(block, config["section_lin"], config["log_k_bins"], config["z_pivot"])
        alpha_lin = (1.0 + alpha) * (p_nl / p_lin) - 1.0
        for i, v in enumerate(alpha_lin):
            block[config["derived_section"], f"alpha_lin_{i}"] = float(v)

    if config["apply_to_nl"]:
        _rescale_section(
            block, config["section_nl"],
            _alpha_on_grid(block[config["section_nl"], "k_h"], config, alpha),
        )
    if config["apply_to_lin"]:
        _rescale_section(
            block, config["section_lin"],
            _alpha_on_grid(block[config["section_lin"], "k_h"], config, alpha),
        )

    sigma = config["smoothness_sigma"]
    if sigma > 0:
        diffs = np.diff(alpha)
        log_prior = -0.5 * np.sum((diffs / sigma) ** 2)
        block[names.likelihoods, f"{config['smoothness_like_name']}_like"] = log_prior

    return 0


def cleanup(config):
    return 0
