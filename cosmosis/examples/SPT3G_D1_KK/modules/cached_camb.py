"""
cosmosis module: cached_camb

Drop-in replacement for cosmosis's `camb` module when cosmology is held
fixed. On the first execute() call it runs the real CAMB and snapshots
every datablock entry CAMB wrote; on subsequent calls it replays the
snapshot. Saves ~2-3 s per likelihood evaluation for samplers that can't
cache across calls (e.g. nautilus, emcee with fast_slow=F).

Usage:

    [camb]
    file = cosmosis_files/modules/cached_camb/cached_camb.py
    camb_file = boltzmann/camb/camb_interface.py
    ; ... everything else you'd normally put under [camb] ...
    mode = all
    lmax = 2500
    halofit_version = mead2020_feedback
    ...

Keep the pipeline module name as `camb` in [pipeline] modules = ... camb ...
so downstream consumers find the same datablock sections they expected.

Caveats:
  * Safe ONLY when cosmological parameters are fixed. If CAMB's inputs
    change between calls the cached output is stale and results are wrong.
  * If you switch to a sampler that varies cosmology, swap this back to
    the real camb module.
"""

import importlib.util
import os
from cosmosis.datablock import option_section


def _load_camb_interface(path):
    if not os.path.isabs(path):
        cosmosis_src = os.environ.get("COSMOSIS_SRC_DIR")
        if cosmosis_src:
            candidate = os.path.join(cosmosis_src, path)
            if os.path.isfile(candidate):
                path = candidate
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"cached_camb: could not locate camb_interface at {path}. "
            "Provide an absolute path in `camb_file`, or set COSMOSIS_SRC_DIR."
        )
    spec = importlib.util.spec_from_file_location("real_camb_interface", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iter_all_keys(block):
    """Yield (section, key) for every entry currently in the datablock."""
    try:
        for item in block.keys():
            if isinstance(item, tuple) and len(item) == 2:
                yield item
            else:
                raise TypeError
    except TypeError:
        for section in block.sections():
            for key in block.keys(section):
                yield (section, key)


def _snapshot(block, skip):
    """Capture (section, key, value) for entries not in `skip`."""
    snap = []
    for section, key in _iter_all_keys(block):
        if (section, key) in skip:
            continue
        try:
            snap.append((section, key, block[section, key]))
        except Exception:
            pass
    return snap


def _restore(block, snap):
    """Write cached entries into block, skipping ones already present."""
    for section, key, val in snap:
        if not block.has_value(section, key):
            block[section, key] = val


def setup(options):
    camb_file = options.get_string(option_section, "camb_file")
    camb_iface = _load_camb_interface(camb_file)
    camb_config = camb_iface.setup(options)
    return {
        "camb_iface": camb_iface,
        "camb_config": camb_config,
        "snapshot": None,
    }


def execute(block, config):
    if config["snapshot"] is None:
        pre_existing = set(_iter_all_keys(block))
        status = config["camb_iface"].execute(block, config["camb_config"])
        if status != 0:
            return status
        config["snapshot"] = _snapshot(block, skip=pre_existing)
        print(
            f"[cached_camb] CAMB ran once; cached "
            f"{len(config['snapshot'])} datablock entries. "
            "Subsequent calls will replay from cache.",
            flush=True,
        )
    else:
        _restore(block, config["snapshot"])
    return 0


def cleanup(config):
    iface = config.get("camb_iface")
    if iface is not None and hasattr(iface, "cleanup"):
        try:
            iface.cleanup(config["camb_config"])
        except Exception:
            pass
    return 0
