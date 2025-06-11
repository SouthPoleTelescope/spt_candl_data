import candl.transformations
from functools import partial
from copy import deepcopy
from candl.lib import jax_optional_set_element, jit, np, jnp, custom_jvp, jacfwd

class PoissonPowerCross(candl.transformations.abstract_base.Foreground):
    def __init__(
        self, ells, spec_order, spec_param_dict, spec_cross_dict, ell_ref, descriptor="Poisson Power"
    ):
        # param_names = list(
        #     np.unique(
        #         [
        #             cal_par
        #             for spec_cal_pars in spec_param_dict.values()
        #             for cal_par in spec_cal_pars
        #         ]
        #     )
        # )

        super().__init__(
            ells=ells,
            ell_ref=ell_ref,
            descriptor=descriptor,
            param_names=list(spec_param_dict.values()),
        )

        self.spec_param_dict = spec_param_dict
        self.spec_cross_dict = spec_cross_dict
        self.affected_specs = list(self.spec_param_dict.keys())

        self.spec_order = spec_order
        self.N_spec = len(self.spec_order)

        # Generate boolean mask of affected specs
        self.spec_mask = np.zeros(
            len(spec_order)
        )  # Generate as np array for easier item assignment
        for i, spec in enumerate(self.spec_order):
            if spec in self.affected_specs:
                self.spec_mask[i] = 1
                # self.spec_mask = self.spec_mask.at[i].set(1)
        self.spec_mask = self.spec_mask == 1
        self.spec_mask = jnp.array(self.spec_mask)
        self.affected_specs_ix = [ix[0] for ix in jnp.argwhere(self.spec_mask)]

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part
        amp_vals = jnp.zeros(len(self.spec_order))
        for ix in self.affected_specs_ix:
            this_amp_val = sample_params[self.spec_param_dict[self.spec_order[ix]]]
            if self.spec_order[ix] in self.spec_cross_dict:
                this_sum = jnp.prod(jnp.array([sample_params[par] for par in self.spec_cross_dict[self.spec_order[ix]]]))
                this_amp_val *= jnp.sqrt(this_sum)
            amp_vals = jax_optional_set_element(
                amp_vals, ix, this_amp_val
            )
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** 2
        tiled_ell_dependence = jnp.tile(ell_dependence, self.N_spec)

        # Complete foreground contribution
        fg_pow = tiled_amp_vals * tiled_ell_dependence
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sample_params : dict
            A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
            The transformed spectrum in Dl.
        """

        return Dls + self.output(sample_params)

class SPT3GCIBClusteringCross(candl.transformations.abstract_base.Foreground):

    def __init__(
        self,
        ells,
        spec_order,
        spec_param_dict,
        spec_cross_dict,
        ell_ref,
        alpha,
        descriptor="CIB Clustering Multi Amp",
    ):

        super().__init__(
            ells=ells,
            ell_ref=ell_ref,
            descriptor=descriptor,
            param_names= list(spec_param_dict.values()) + [alpha],
        )

        self.spec_param_dict = spec_param_dict
        self.spec_cross_dict = spec_cross_dict
        self.affected_specs = list(self.spec_param_dict.keys())
        self.alpha = alpha

        self.spec_order = spec_order
        self.N_spec = len(self.spec_order)

        # Generate boolean mask of affected specs
        self.spec_mask = jnp.zeros(
            len(spec_order),
            dtype=bool,
        )  # Generate as jnp array for easier item assignment
        for i, spec in enumerate(self.spec_order):
            if spec in self.affected_specs:
                self.spec_mask = jax_optional_set_element(self.spec_mask, i, True)

        self.full_mask = jnp.asarray(
            jnp.repeat(self.spec_mask, len(self.ells)), dtype=float
        )

        self.affected_specs_ix = [ix[0] for ix in jnp.argwhere(self.spec_mask)]

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part
        amp_vals = jnp.zeros(len(self.spec_order))
        for ix in self.affected_specs_ix:
            this_amp_val = sample_params[self.spec_param_dict[self.spec_order[ix]]]
            if self.spec_order[ix] in self.spec_cross_dict:
                this_sum = jnp.prod(jnp.array([sample_params[par] for par in self.spec_cross_dict[self.spec_order[ix]]]))
                this_amp_val *= jnp.sqrt(this_sum)
            amp_vals = jax_optional_set_element(
                amp_vals, ix, this_amp_val
            )
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # alpha
        alpha = sample_params[self.alpha]

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** alpha
        tiled_ell_dependence = jnp.tile(ell_dependence, self.N_spec)

        # Complete foreground contribution
        fg_pow = self.full_mask * tiled_amp_vals * tiled_ell_dependence
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the ijnput spectrum.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sample_params : dict
            A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
            The transformed spectrum in Dl.
        """

        return Dls + self.output(sample_params)

class SPT3GCIBClustering(candl.transformations.abstract_base.Foreground):

    def __init__(
        self,
        ells,
        spec_order,
        spec_param_dict,
        ell_ref,
        alpha,
        descriptor="CIB Clustering Multi Amp",
    ):

        super().__init__(
            ells=ells,
            ell_ref=ell_ref,
            descriptor=descriptor,
            param_names=list(spec_param_dict.values()) + [alpha],
        )

        self.spec_param_dict = spec_param_dict
        self.alpha = alpha
        self.spec_order = spec_order
        self.generate_mask()
    
    def generate_mask(self):
        self.affected_specs = list(self.spec_param_dict.keys())
        
        self.N_spec = len(self.spec_order)

        # Generate boolean mask of affected specs
        self.spec_mask = jnp.zeros(
            len(self.spec_order),
            dtype=bool,
        )  # Generate as jnp array for easier item assignment

        for i, spec in enumerate(self.spec_order):
            if spec in self.affected_specs:
                self.spec_mask = jax_optional_set_element(self.spec_mask, i, True)

        self.full_mask = jnp.asarray(
            jnp.repeat(self.spec_mask, len(self.ells)), dtype=float
        )
        self.affected_specs_ix = [ix[0] for ix in jnp.argwhere(self.spec_mask)]
        return 

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return foreground spectrum.

        Arguments
        --------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        --------------
        array, float
            Foreground spectrum.
        """

        # amplitude part
        amp_vals = jnp.zeros(len(self.spec_order))
        for ix in self.affected_specs_ix:
            amp_vals = jax_optional_set_element(
                amp_vals, ix, sample_params[self.spec_param_dict[self.spec_order[ix]]]
            )
        tiled_amp_vals = jnp.repeat(amp_vals, len(self.ells))

        # alpha
        alpha = sample_params[self.alpha]

        # ell part
        ell_dependence = (self.ells / self.ell_ref) ** alpha
        tiled_ell_dependence = jnp.tile(ell_dependence, self.N_spec)

        # Complete foreground contribution
        fg_pow = self.full_mask * tiled_amp_vals * tiled_ell_dependence
        return fg_pow

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the ijnput spectrum.

        Arguments
        --------------
        Dls : array (float)
            The spectrum to transform in Dl.
        sample_params : dict
            A dictionary of parameters that are used in the transformation

        Returns
        --------------
        array : float
            The transformed spectrum in Dl.
        """

        return Dls + self.output(sample_params)


# CIB-tSZ correlation functions defined outside of class to guarantee differentiability
@partial(custom_jvp, nondiff_argnums=(0,))
def _SPT3GCIBtSZCorrelationGeometricMean_output(fg_class_instance, sample_params):
    # CIB
    CIB_nu_1 = fg_class_instance.CIB[0].output(sample_params)
    CIB_nu_2 = fg_class_instance.CIB[1].output(sample_params)

    # tSZ
    tSZ_nu_1 = fg_class_instance.tSZ[0].output(sample_params)
    tSZ_nu_2 = fg_class_instance.tSZ[1].output(sample_params)

    # CIB x tSZ
    CIB_x_tSZ = jnp.sqrt(CIB_nu_1 * tSZ_nu_2) + jnp.sqrt(CIB_nu_2 * tSZ_nu_1)

    # Complete foreground contribution and mask down
    fg_pow = (
        -1.0
        * fg_class_instance.full_mask
        * sample_params[fg_class_instance.amp_param]
        * CIB_x_tSZ
    )
    return fg_pow


@_SPT3GCIBtSZCorrelationGeometricMean_output.defjvp
def _SPT3GCIBtSZCorrelationGeometricMean_output_jvp(
    fg_class_instance, primals, tangents
):
    """
    Hand-defined derivative of CIB-tSZ correlation term output function.
    See also: candl.transformations.common._SPT3GCIBtSZCorrelationGeometricMean_output, jax.custom_jvp
    """
    # Process ijnput into regular dictionary
    (full_pars,) = primals
    (pars_dot,) = tangents

    # Don't pass on Dl array - it's unnecessary
    pars = deepcopy(full_pars)
    if "Dl" in pars:
        del pars["Dl"]

    # Pass to original function for values
    ans = fg_class_instance.output(pars)

    # Calculate derivatives

    # xi
    xi_deriv = ans / pars["TT_tSZ_CIB_Corr_Amp"]

    # A_tSZ
    tSZ_amp_deriv_term1 = (
        0.5
        * jacfwd(fg_class_instance.tSZ[1].output)(pars)["TT_tSZ_Amp"]
        * jnp.sqrt(
            fg_class_instance.CIB[0].output(pars)
            / fg_class_instance.tSZ[1].output(pars)
        )
    )
    tSZ_amp_deriv_term2 = (
        0.5
        * jacfwd(fg_class_instance.tSZ[0].output)(pars)["TT_tSZ_Amp"]
        * jnp.sqrt(
            fg_class_instance.CIB[1].output(pars)
            / fg_class_instance.tSZ[0].output(pars)
        )
    )
    tSZ_amp_deriv_term1 = jnp.where(
            jnp.invert(jnp.asarray(fg_class_instance.CIB[0].full_mask, dtype=bool)),
            0.0,
            tSZ_amp_deriv_term1,
        )
    tSZ_amp_deriv_term2 = jnp.where(
            jnp.invert(jnp.asarray(fg_class_instance.CIB[1].full_mask, dtype=bool)),
            0.0,
            tSZ_amp_deriv_term2,
        )

    tSZ_amp_deriv = -pars["TT_tSZ_CIB_Corr_Amp"] * (
        tSZ_amp_deriv_term1 + tSZ_amp_deriv_term2
    )
    tSZ_amp_deriv = jnp.where(
        jnp.invert(jnp.asarray(fg_class_instance.full_mask, dtype=bool)),
        0.0,
        tSZ_amp_deriv,
    )

    # A_CIB
    CIB_amp_deriv = {}
    amplitudes = np.unique(
        list(fg_class_instance.CIB[0].spec_param_dict.values())
        + list(fg_class_instance.CIB[1].spec_param_dict.values())
    )
    for amp in amplitudes:
        CIB_amp_deriv_term1 = (
            0.5
            * jacfwd(fg_class_instance.CIB[1].output)(pars)[amp]
            * jnp.sqrt(
                fg_class_instance.tSZ[0].output(pars)
                / fg_class_instance.CIB[1].output(pars)
            )
        )
        CIB_amp_deriv_term2 = (
            0.5
            * jacfwd(fg_class_instance.CIB[0].output)(pars)[amp]
            * jnp.sqrt(
                fg_class_instance.tSZ[1].output(pars)
                / fg_class_instance.CIB[0].output(pars)
            )
        )

        CIB_amp_deriv_term1 = jnp.where(
            jnp.invert(jnp.asarray(fg_class_instance.CIB[1].full_mask, dtype=bool)),
            0.0,
            CIB_amp_deriv_term1,
        )
        CIB_amp_deriv_term2 = jnp.where(
            jnp.invert(jnp.asarray(fg_class_instance.CIB[0].full_mask, dtype=bool)),
            0.0,
            CIB_amp_deriv_term2,
        )

        CIB_amp_deriv[amp] = -pars["TT_tSZ_CIB_Corr_Amp"] * (
            CIB_amp_deriv_term1 + CIB_amp_deriv_term2
        )

    # alpha
    alpha_deriv_term1 = (
        0.5
        * jacfwd(fg_class_instance.CIB[1].output)(pars)["TT_CIBClustering_Alpha"]
        * jnp.sqrt(
            fg_class_instance.tSZ[0].output(pars)
            / fg_class_instance.CIB[1].output(pars)
        )
    )
    alpha_deriv_term2 = (
        0.5
        * jacfwd(fg_class_instance.CIB[0].output)(pars)["TT_CIBClustering_Alpha"]
        * jnp.sqrt(
            fg_class_instance.tSZ[1].output(pars)
            / fg_class_instance.CIB[0].output(pars)
        )
    )

    alpha_deriv_term1 = jnp.where(
        jnp.invert(jnp.asarray(fg_class_instance.CIB[1].full_mask, dtype=bool)),
        0.0,
        alpha_deriv_term1,
    )
    alpha_deriv_term2 = jnp.where(
        jnp.invert(jnp.asarray(fg_class_instance.CIB[0].full_mask, dtype=bool)),
        0.0,
        alpha_deriv_term2,
    )

    alpha_deriv = -pars["TT_tSZ_CIB_Corr_Amp"] * (alpha_deriv_term1 + alpha_deriv_term2)

    ans_dot = (
        tSZ_amp_deriv * pars_dot["TT_tSZ_Amp"] # this is the line posing problem
        + alpha_deriv * pars_dot["TT_CIBClustering_Alpha"]
    )
    if fg_class_instance.value is None:
        ans_dot += xi_deriv * pars_dot["TT_tSZ_CIB_Corr_Amp"]
    for amp in amplitudes:
        ans_dot += CIB_amp_deriv[amp] * pars_dot[amp]

    return ans, ans_dot


class SPT3GCIBtSZCorrelationGeometricMean(
    candl.transformations.abstract_base.Foreground
):
    def __init__(
        self,
        ells,
        spec_order,
        affected_specs,
        amp_param,
        link_transformation_module_CIB,
        link_transformation_module_tSZ,
        descriptor="CIB-tSZ correlation",
    ):
        if isinstance(amp_param, str):
            param_names = [amp_param]
            self.amp_param = amp_param
            self.value = None
        else:
            param_names = []
            self.amp_param = 'TT_tSZ_CIB_Corr_Amp'
            self.value = amp_param

        super().__init__(
            ells=ells,
            ell_ref=0,  # reference ell not required
            descriptor=descriptor,
            param_names=param_names,
        )

        self.spec_order = spec_order
        self.affected_specs = affected_specs
        self.spec_mask = jnp.asarray(
            [spec in self.affected_specs for spec in self.spec_order]
        )
        self.N_spec = len(self.spec_mask)

        # Turn spectrum mask into a full mask
        self.full_mask = jnp.asarray(
            jnp.repeat(self.spec_mask, len(self.ells)), dtype=float
        )

        # Make 2 copies of the CIB and tSZ classes and modify their effective frequencies
        # (need to have nu_1-only and nu_2-only versions of each).
        self.CIB = [
            deepcopy(link_transformation_module_CIB),
            deepcopy(link_transformation_module_CIB),
        ]
        self.tSZ = [
            deepcopy(link_transformation_module_tSZ),
            deepcopy(link_transformation_module_tSZ),
        ]

        for i, tSZ_freq_pair in enumerate(link_transformation_module_tSZ.freq_info):
            self.tSZ[0].freq_info[i] = [tSZ_freq_pair[0], tSZ_freq_pair[0]]
            self.tSZ[1].freq_info[i] = [tSZ_freq_pair[1], tSZ_freq_pair[1]]

        for affected_spec in self.affected_specs:
            st, fr = affected_spec.split(" ")
            f1, f2 = fr.split("x")
            if f"{st} {f1}x{f1}" in link_transformation_module_CIB.spec_param_dict:
                self.CIB[0].spec_param_dict[affected_spec] = (
                    link_transformation_module_CIB.spec_param_dict[f"{st} {f1}x{f1}"]
                )
                self.CIB[0].generate_mask()
            if f"{st} {f2}x{f2}" in link_transformation_module_CIB.spec_param_dict:
                self.CIB[1].spec_param_dict[affected_spec] = (
                    link_transformation_module_CIB.spec_param_dict[f"{st} {f2}x{f2}"]
                )
                self.CIB[1].generate_mask()

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        if self.value is None:
            return _SPT3GCIBtSZCorrelationGeometricMean_output(self, sample_params)
        else:
            copy_pars = deepcopy(sample_params)
            copy_pars[self.amp_param] = self.value
            return _SPT3GCIBtSZCorrelationGeometricMean_output(self, copy_pars)

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        return Dls + self.output(sample_params)
