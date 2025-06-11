"""
Beams module containing the Beams class. 

Note:
---------------
Warning: this is NOT a comprehensive beam model library. 

Overview:
-----------------
...

"""

from candl.lib import *
import candl.transformations.abstract_base
import candl.constants

# --------------------------------------#
# BEAMS
# --------------------------------------#


class BeamModes(candl.transformations.abstract_base.Transformation):
    """ 
    rc2 beam model
    """

    def __init__(
        self,
        ells,
        data_set_dict,
        spec_freqs,
        beam_eigenmodes,
        modes_params,
        descriptor="Beams",
        operation_hint="multiplicative",
        mode_index=None,
        **kwargs,
    ):
        """
        Initialise a new instance of the Beams class.
        """

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=modes_params,
            operation_hint=operation_hint,
        )
        self.spec_freqs = spec_freqs
        load = np.load(f"{data_set_dict['data_set_path']}{beam_eigenmodes}")
        self.beam_eigenmodes_array = load["modes"]
        self.beam_ells = load["ell"]
        del load
        indices = np.isin(self.beam_ells, self.ells)
        self.beam_ells = self.beam_ells[indices]
        self.freqs = np.unique(spec_freqs)
        indices = np.tile(indices, 3)
        self.beam_eigenmodes_array = self.beam_eigenmodes_array[indices]

        # select only the first N eigenmodes, N = len(spec_param_dict)
        if mode_index is None:
            mode_index = list(range(len(modes_params)))
        else:
            assert len(mode_index) == len(
                modes_params
            ), "mode_index must have the same length as modes_params"
        self.beam_eigenmodes = {}
        for ind_param, param in enumerate(modes_params):
            self.beam_eigenmodes[param] = self.beam_eigenmodes_array[
                :, mode_index[ind_param]
            ]

        # expand per frequency
        freqs = {"90": 0, "150": 1, "220": 2}
        self.beam_eigenmodes_freq = {
            param: {
                freq: val[
                    freqs[freq] * len(self.ells) : (freqs[freq] + 1) * len(self.ells)
                ]
                for freq in self.freqs
            }
            for param, val in self.beam_eigenmodes.items()
        }

        # the eigenmodes are ordered 090GHz, then 150GHz, then 220GHz
        self.beam_long_dv = {
            param: [
                np.zeros(len(self.ells) * len(self.spec_freqs)),
                np.zeros(len(self.ells) * len(self.spec_freqs)),
            ]
            for param in self.param_names
        }
        for param, val in self.beam_eigenmodes_freq.items():
            for ind, tuple_freq in enumerate(self.spec_freqs):
                self.beam_long_dv[param][0][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] += val[tuple_freq[0]]
                self.beam_long_dv[param][1][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] += val[tuple_freq[1]]

    def output_left_right(self, sample_params):
        left = jnp.ones(len(self.ells) * len(self.spec_freqs))
        right = jnp.ones(len(self.ells) * len(self.spec_freqs))
        for param in self.param_names:
            left += self.beam_long_dv[param][0] * sample_params[param]
            right += self.beam_long_dv[param][1] * sample_params[param]
        return left, right

    def output(self, sample_params):
        left, right = self.output_left_right(sample_params)
        return 1./jnp.multiply(left, right)

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        return Dls / self.output(sample_params)


class ModesPolarizedBeamRC3(candl.transformations.abstract_base.Transformation):
    """
    rc3 polarized beam model
    """

    def __init__(
        self,
        ells,
        long_ells,
        data_set_dict,
        spec_freqs,
        spec_types,
        beam_eigenmodes,
        beam_main_temperature,
        modes_params,
        pol_params,
        descriptor="Polarized Beam Model (RC3)",
        operation_hint="multiplicative",
        mode_index=None,
        applies_to=['E'],
        alpha=0.,
        beam_renorm=True,
        **kwargs,
    ):
        """
        Initialise a new instance of the PolarizedBeamRC3 class.
        """

        if isinstance(alpha, float) or isinstance(alpha, int):
            super().__init__(
                ells=ells,
                descriptor=descriptor,
                param_names=modes_params + pol_params,
                operation_hint=operation_hint,
            )
            self.alpha = alpha
        elif isinstance(alpha, str):
            super().__init__(
                ells=ells,
                descriptor=descriptor,
                param_names=modes_params + pol_params + [alpha],
                operation_hint=operation_hint,
            )
            self.alpha = alpha
        else:
            raise ValueError("alpha must be a float or an int or a string")
            
        self.long_ells = long_ells
        self.modes_params = modes_params
        self.pol_params = pol_params
        self.spec_freqs = spec_freqs
        self.spec_types = spec_types
        self.freqs = np.unique(spec_freqs)

        # first load the beam eigenmodes
        load = np.load(f"{data_set_dict['data_set_path']}{beam_eigenmodes}")
        self.beam_eigenmodes_array = load["modes"]
        self.beam_ells = load["ell"]
        del load
        indices = np.isin(self.beam_ells, self.ells)
        self.beam_ells = self.beam_ells[indices]
        indices = np.tile(indices, 3)
        self.beam_eigenmodes_array = self.beam_eigenmodes_array[indices]
        # select only the first N eigenmodes, N = len(spec_param_dict)
        if mode_index is None:
            mode_index = list(range(len(modes_params)))
        else:
            assert len(mode_index) == len(
                modes_params
            ), "mode_index must have the same length as modes_params"
        self.beam_eigenmodes = {}
        for ind_param, param in enumerate(modes_params):
            self.beam_eigenmodes[param] = self.beam_eigenmodes_array[
                :, mode_index[ind_param]
            ]
        # expand per frequency
        freqs = {"90": 0, "150": 1, "220": 2}
        self.beam_eigenmodes_freq = {
            param: {
                freq: val[
                    freqs[freq] * len(self.ells) : (freqs[freq] + 1) * len(self.ells)
                ]
                for freq in self.freqs
            }
            for param, val in self.beam_eigenmodes.items()
        }
        # the eigenmodes are ordered 090GHz, then 150GHz, then 220GHz
        self.beam_long_dv = {
            param: [
                np.zeros(len(self.ells) * len(self.spec_freqs)),
                np.zeros(len(self.ells) * len(self.spec_freqs)),
            ]
            for param in self.modes_params
        }
        for param, val in self.beam_eigenmodes_freq.items():
            for ind, tuple_freq in enumerate(self.spec_freqs):
                self.beam_long_dv[param][0][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] += val[tuple_freq[0]]
                self.beam_long_dv[param][1][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] += val[tuple_freq[1]]

        # then load the beam main temperature
        load = np.load(f"{data_set_dict['data_set_path']}{beam_main_temperature}")
        self.beam_ells = load["ell"]
        indices = np.isin(self.beam_ells, self.ells)
        self.beam_main_temperature = {}
        for freq in self.freqs:
            self.beam_main_temperature[freq] = load[freq][indices]
        del load
        self.pol_beam_long_dv = {}
        self.pol_beam_long_dv_mask = {}
        self.pol_beam_long_dv_total_mask = [
            np.zeros(len(self.ells) * len(self.spec_freqs)),
            np.zeros(len(self.ells) * len(self.spec_freqs)),
        ]
        self.freq_to_par = {}
        for freq in self.freqs:
            for indpar, par in enumerate(self.pol_params):
                if freq in par:
                    self.freq_to_par[freq] = indpar
                    self.pol_beam_long_dv[par] = [
                        np.zeros(len(self.ells) * len(self.spec_freqs)),
                        np.zeros(len(self.ells) * len(self.spec_freqs)),
                    ]
                    self.pol_beam_long_dv_mask[par] = [
                        np.zeros(len(self.ells) * len(self.spec_freqs)),
                        np.zeros(len(self.ells) * len(self.spec_freqs)),
                    ]

        for ind, tuple_freq in enumerate(self.spec_freqs):
            l, r = self.spec_types[ind][0], self.spec_types[ind][1]
            par_l, par_r = (
                self.pol_params[self.freq_to_par[tuple_freq[0]]],
                self.pol_params[self.freq_to_par[tuple_freq[1]]],
            )

            if l in applies_to:
                self.pol_beam_long_dv[par_l][0][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] += self.beam_main_temperature[tuple_freq[0]]
                self.pol_beam_long_dv_mask[par_l][0][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] = 1.0
                self.pol_beam_long_dv_total_mask[0][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] = 1.0
            if r in applies_to:
                self.pol_beam_long_dv[par_r][1][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] += self.beam_main_temperature[tuple_freq[1]]
                self.pol_beam_long_dv_mask[par_r][1][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] = 1.0
                self.pol_beam_long_dv_total_mask[1][
                    ind * len(self.ells) : (ind + 1) * len(self.ells)
                ] = 1.0

        to_remove = []
        for par in self.pol_params:
            if not np.any(par in list(self.pol_beam_long_dv_mask.keys())):
                to_remove.append(par)
        for par in to_remove:
            self.pol_params.remove(par)
        
        if beam_renorm:
            self.beam_main_800 = {}
            for freq in self.freqs:
                for indpar, par in enumerate(self.pol_params):
                    if freq in par:
                            self.beam_main_800[par] = self.beam_main_temperature[freq][ells == 800]
        else:
            self.beam_main_800 = {
                param: 1. 
                for param in self.pol_params
            }
        

    def output_modes(self, sample_params):
        left = jnp.ones(len(self.ells) * len(self.spec_freqs))
        right = jnp.ones(len(self.ells) * len(self.spec_freqs))
        for param in self.modes_params:
            left += self.beam_long_dv[param][0] * sample_params[param]
            right += self.beam_long_dv[param][1] * sample_params[param]
        return left, right
    
    def output(self, sample_params):
        alpha = sample_params[self.alpha] if isinstance(self.alpha, str) else self.alpha
        left = jnp.zeros(len(self.ells) * len(self.spec_freqs))
        right = jnp.zeros(len(self.ells) * len(self.spec_freqs))
        left_modes, right_modes = self.output_modes(sample_params)
        for param in self.pol_params:
            # left += ((self.pol_beam_long_dv[param][0] + sample_params[param] *
            #         (self.long_ells / 4000) ** alpha *
            #         (left_modes - self.pol_beam_long_dv[param][0])) * self.pol_beam_long_dv_mask[param][0]) / (
            #             self.beam_main_800[param] + sample_params[param] *
            #             (self.long_ells / 4000) ** alpha *
            #             (1. - self.beam_main_800[param]) * self.pol_beam_long_dv_mask[param][0]
            #         )
            # right += ((self.pol_beam_long_dv[param][1] + sample_params[param] * 
            #         (self.long_ells / 4000) ** alpha *
            #         (right_modes - self.pol_beam_long_dv[param][1])) * self.pol_beam_long_dv_mask[param][1]) / (
            #             self.beam_main_800[param] + sample_params[param] *
            #             (self.long_ells / 4000) ** alpha *
            #             (1. - self.beam_main_800[param]) * self.pol_beam_long_dv_mask[param][1]
            #         )

            left += ((self.pol_beam_long_dv[param][0] + sample_params[param] *
                    (alpha * self.long_ells/4000 + 1 - alpha) *
                    (left_modes - self.pol_beam_long_dv[param][0])) * self.pol_beam_long_dv_mask[param][0]) / (
                        self.beam_main_800[param] + sample_params[param] *
                        (alpha * self.long_ells/4000 + 1 - alpha) *
                        (1. - self.beam_main_800[param]) * self.pol_beam_long_dv_mask[param][0]
                    )
            right += ((self.pol_beam_long_dv[param][1] + sample_params[param] * 
                    (alpha * self.long_ells/4000 + 1 - alpha) *
                    (right_modes - self.pol_beam_long_dv[param][1])) * self.pol_beam_long_dv_mask[param][1]) / (
                        self.beam_main_800[param] + sample_params[param] *
                        (alpha * self.long_ells/4000 + 1 - alpha) *
                        (1. - self.beam_main_800[param]) * self.pol_beam_long_dv_mask[param][1]
                    )
        return 1./jnp.multiply(
            left + left_modes * (1 - self.pol_beam_long_dv_total_mask[0]),
            right + right_modes * (1 - self.pol_beam_long_dv_total_mask[1]),
        )

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        return Dls / self.output(sample_params)
