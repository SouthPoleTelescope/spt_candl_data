import numpy as np

from cobaya.theories.camb.camb import CAMB


class a2ptarec(CAMB):

    params = {
        "Alens": None,
        "Arec": None,
    }

    def calculate(self, state, want_derived=True, **params_values_dict):
        Arec = float(params_values_dict.pop("Arec", 1.0))
        Alens = float(params_values_dict.pop("Alens", 1.0))
        
        params_values_dict["Alens"] = Alens

        ok = super().calculate(state, want_derived=want_derived, **params_values_dict)
        if ok is False:
            return False

        if Alens == 0.0:
            return False

        r = Arec / Alens
        r_sqrt = np.sqrt(r)
        
        for key in ("Cl", "lensed_scal_Cl", "unlensed_Cl"):
            cl = state.get(key)
            #import pdb;pdb.set_trace()
            if not isinstance(cl, dict):
                continue

            if "lens_potential" in cl:
                cl["lens_potential"][:, 0] *= r        # pp
                cl["lens_potential"][:, 1] *= r_sqrt   # tp
                cl["lens_potential"][:, 2] *= r_sqrt   # ep

            #if "lens_potential" in cl:
            #    lp = cl["lens_potential"]
            #    lp[:, 0] *= r          # pp
            #    lp[:, 1] *= r_sqrt     # tp
            #    lp[:, 2] *= r_sqrt     # ep
            '''
            if "pp" in cl:
                cl["pp"] *= r

            # If any likelihood requests cross spectra with phi, keep them consistent:
            if "tp" in cl:
                cl["tp"] *= r_sqrt
            if "ep" in cl:
                cl["ep"] *= r_sqrt
            '''
        return True

