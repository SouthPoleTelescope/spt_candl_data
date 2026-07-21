"""
Expose log1e10As from the A_s that CAMB solved for (when sampling S_8/sigma_8).

When the amplitude is sampled as S_8 (-> sigma_8 -> CAMB solves A_s), the block
has A_s but not log1e10As. The candl interface requires log1e10As
(model_dict["logA"] = model_dict["log1e10as"]) for its lensing systematics
emulator, so provide it here from the solved A_s. Place this module AFTER camb
and BEFORE the candl likelihood.
"""
import numpy as np
from cosmosis.datablock import names

cosmo = names.cosmological_parameters


def setup(options):
    return {}


def execute(block, config):
    A_s = block[cosmo, "A_s"]
    block[cosmo, "log1e10As"] = float(np.log(A_s * 1.0e10))
    return 0


def cleanup(config):
    return 0
