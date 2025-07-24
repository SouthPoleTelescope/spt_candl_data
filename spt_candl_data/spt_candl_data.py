"""
Short cuts to ``.yaml`` files for released data sets.

"""

# Grab Data Path
import os

data_path = os.path.dirname(os.path.realpath(__file__))

# --------------------------------------#
# DATA SETS
# --------------------------------------#

# SPT-3G D1 BB
SPT3G_D1_BB_folder = "SPT3G_D1_BB_v0"
SPT3G_D1_BB = f"{data_path}/{SPT3G_D1_BB_folder}/SPT3G_D1_BB_BK_HL.yaml"


# SPT-3G D1 TnE
SPT3G_D1_TnE_folder = "SPT3G_D1_TnE_v0"

# Key files
SPT3G_D1_TnE = f"{data_path}/{SPT3G_D1_TnE_folder}/SPT3G_D1_TnE_index.yaml"
SPT3G_D1_TnE_multifreq = f"{data_path}/{SPT3G_D1_TnE_folder}/SPT3G_D1_TnE.yaml"
SPT3G_D1_TnE_lite = f"{data_path}/{SPT3G_D1_TnE_folder}/lite/SPT3G_D1_TnE_lite.yaml"

# Individual spectra
SPT3G_D1_TT = f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TT.yaml"
SPT3G_D1_TE = f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TE.yaml"
SPT3G_D1_EE = f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_EE.yaml"
SPT3G_D1_TEEE = f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TEEE.yaml"
SPT3G_D1_TTTE = f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TTTE.yaml"

# Individual frequencies
SPT3G_D1_TnE_90x90 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TnE_90x90.yaml"
)
SPT3G_D1_TnE_90x150 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TnE_90x150.yaml"
)
SPT3G_D1_TnE_90x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TnE_90x220.yaml"
)
SPT3G_D1_TnE_150x150 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TnE_150x150.yaml"
)
SPT3G_D1_TnE_150x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TnE_150x220.yaml"
)
SPT3G_D1_TnE_220x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TnE_220x220.yaml"
)

SPT3G_D1_TT_90x90 = f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TT_90x90.yaml"
SPT3G_D1_TT_90x150 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TT_90x150.yaml"
)
SPT3G_D1_TT_90x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TT_90x220.yaml"
)
SPT3G_D1_TT_150x150 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TT_150x150.yaml"
)
SPT3G_D1_TT_150x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TT_150x220.yaml"
)
SPT3G_D1_TT_220x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TT_220x220.yaml"
)

SPT3G_D1_TE_90x90 = f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TE_90x90.yaml"
SPT3G_D1_TE_90x150 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TE_90x150.yaml"
)
SPT3G_D1_TE_90x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TE_90x220.yaml"
)
SPT3G_D1_TE_150x150 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TE_150x150.yaml"
)
SPT3G_D1_TE_150x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TE_150x220.yaml"
)
SPT3G_D1_TE_220x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_TE_220x220.yaml"
)

SPT3G_D1_EE_90x90 = f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_EE_90x90.yaml"
SPT3G_D1_EE_90x150 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_EE_90x150.yaml"
)
SPT3G_D1_EE_90x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_EE_90x220.yaml"
)
SPT3G_D1_EE_150x150 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_EE_150x150.yaml"
)
SPT3G_D1_EE_150x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_EE_150x220.yaml"
)
SPT3G_D1_EE_220x220 = (
    f"{data_path}/{SPT3G_D1_TnE_folder}/subsets/SPT3G_D1_EE_220x220.yaml"
)

# --------------------------------------#
# PREPARE SHORCUTS AND INFO
# --------------------------------------#


# Define Shortcuts
shortcuts = {
    "SPT-3G D1 TnE": {
        "index": "SPT3G_D1_TnE",
        "multifreq": "SPT3G_D1_TnE_multifreq",
        "lite": "SPT3G_D1_TnE_lite",
        "TT": "SPT3G_D1_TT",
        "TE": "SPT3G_D1_TE",
        "EE": "SPT3G_D1_EE",
        "TEEE": "SPT3G_D1_TEEE",
        "TTTE": "SPT3G_D1_TTTE",
        "90x90": "SPT3G_D1_TnE_90x90",
        "90x150": "SPT3G_D1_TnE_90x150",
        "90x220": "SPT3G_D1_TnE_90x220",
        "150x150": "SPT3G_D1_TnE_150x150",
        "150x220": "SPT3G_D1_TnE_150x220",
        "220x220": "SPT3G_D1_TnE_220x220",
        "TT_90x90": "SPT3G_D1_TT_90x90",
        "TT_90x150": "SPT3G_D1_TT_90x150",
        "TT_90x220": "SPT3G_D1_TT_90x220",
        "TT_150x150": "SPT3G_D1_TT_150x150",
        "TT_150x220": "SPT3G_D1_TT_150x220",
        "TT_220x220": "SPT3G_D1_TT_220x220",
        "TE_90x90": "SPT3G_D1_TE_90x90",
        "TE_90x150": "SPT3G_D1_TE_90x150",
        "TE_90x220": "SPT3G_D1_TE_90x220",
        "TE_150x150": "SPT3G_D1_TE_150x150",
        "TE_150x220": "SPT3G_D1_TE_150x220",
        "TE_220x220": "SPT3G_D1_TE_220x220",
        "EE_90x90": "SPT3G_D1_EE_90x90",
        "EE_90x150": "SPT3G_D1_EE_90x150",
        "EE_90x220": "SPT3G_D1_EE_90x220",
        "EE_150x150": "SPT3G_D1_EE_150x150",
        "EE_150x220": "SPT3G_D1_EE_150x220",
        "EE_220x220": "SPT3G_D1_EE_220x220",
    },
    "SPT-3G D1 BB": "SPT3G_D1_BB",
}


# Additional Information
info = {
    "SPT-3G D1 TnE": {
        "index": None,
        "multifreq": "Complete multi-frequency likelihood.",
        "lite": "SPTlite, CMB-only (lite) likelihood.",
        "TT": "TT-only multi-frequency likelihood.",
        "TE": "TE-only multi-frequency likelihood.",
        "EE": "EE-only multi-frequency likelihood.",
        "TEEE": "TE/EE multi-frequency likelihood.",
        "TTTE": "TT/TE multi-frequency likelihood.",
        "90x90": "TT/TE/EE likelihood using only 90x90 spectra.",
        "90x150": "TT/TE/EE likelihood using only 90x150 spectra.",
        "90x220": "TT/TE/EE likelihood using only 90x220 spectra.",
        "150x150": "TT/TE/EE likelihood using only 150x150 spectra.",
        "150x220": "TT/TE/EE likelihood using only 150x220 spectra.",
        "220x220": "TT/TE/EE likelihood using only 220x220 spectra.",
        "TT_90x90": "TT likelihood using only the 90x90 spectrum.",
        "TT_90x150": "TT likelihood using only the 90x150 spectrum.",
        "TT_90x220": "TT likelihood using only the 90x220 spectrum.",
        "TT_150x150": "TT likelihood using only the 150x150 spectrum.",
        "TT_150x220": "TT likelihood using only the 150x220 spectrum.",
        "TT_220x220": "TT likelihood using only the 220x220 spectrum.",
        "TE_90x90": "TE likelihood using only the 90x90 spectrum.",
        "TE_90x150": "TE likelihood using only the 90x150 spectrum.",
        "TE_90x220": "TE likelihood using only the 90x220 spectrum.",
        "TE_150x150": "TE likelihood using only the 150x150 spectrum.",
        "TE_150x220": "TE likelihood using only the 150x220 spectrum.",
        "TE_220x220": "TE likelihood using only the 220x220 spectrum.",
        "EE_90x90": "EE likelihood using only the 90x90 spectrum.",
        "EE_90x150": "EE likelihood using only the 90x150 spectrum.",
        "EE_90x220": "EE likelihood using only the 90x220 spectrum.",
        "EE_150x150": "EE likelihood using only the 150x220 spectrum.",
        "EE_150x220": "EE likelihood using only the 150x150 spectrum.",
        "EE_220x220": "EE likelihood using only the 220x220 spectrum.",
    },
}

# default data sets
default_data_sets = {
    "SPT-3G D1 TnE": "multifreq",
}

# --------------------------------------#
# PRINT SHORTCUTS
# --------------------------------------#


def print_all_shortcuts():
    """
    Prints all available shortcuts to data sets.
    """
    for ky in list(shortcuts.keys()):
        # Check if we are dealing with an index file
        if isinstance(shortcuts[ky], dict):
            print(
                f"{ky}: 'spt_candl_data.{shortcuts[ky]['index']}' (index file)\n  Multiple variants available, access with index file (above) and 'variant' keyword or with specific shortcut for desired variant:"
            )
            for subky in list(shortcuts[ky].keys()):
                if subky == "index":
                    continue
                info_str = ""
                if info[ky][subky]:
                    info_str = f"{info[ky][subky]}"
                print(
                    f"  * 'variant = {subky}' or 'spt_candl_data.{shortcuts[ky][subky]}'  (data files located at: {globals()[shortcuts[ky][subky]]})\n    {info_str}"
                )
                if default_data_sets[ky] == subky:
                    print(
                        f"    Default variant. Using spt_candl_data.{shortcuts[ky]['index']} will initialise this variant."
                    )
        else:
            print(
                f"{ky}: 'spt_candl_data.{shortcuts[ky]}'\n(data files located at: {globals()[shortcuts[ky]]})"
            )


# --------------------------------------#
# TESTS
# --------------------------------------#


def run_all_tests():
    """
    Runs all tests.
    See candl.test.run_test().
    """

    from candl.tests import run_test

    # Grab all test yaml files

    all_test_yaml_files = [
        f for f in os.listdir(f"{data_path}/tests/") if f.endswith(".yaml")
    ]

    # Loop over all tests
    for test_file in all_test_yaml_files:
        run_test(f"{data_path}/tests/{test_file}")
