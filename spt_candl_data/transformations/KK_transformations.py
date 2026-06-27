# --------------------------------------#
# IMPORTS
# --------------------------------------#
from candl.lib import *
import candl.transformations.abstract_base
import candl.transformations.common_lensing

# --------------------------------------#
# LENSING TRANSFORMATIONS
# --------------------------------------#

class ATTTEEE(candl.transformations.abstract_base.Transformation):
    """
    Class to scale ATT, ATE, AEE per bin relative to some fiducial ClTT, ClTE, ClEE
    """
    def __init__(
        self,
        ells,
        crop_mask,
        window_functions,
        spec_types,
        N_bins,
        descriptor=""):

        A_params = ["A%s%i"%(spec,idx+1) for nbins, spec in zip(N_bins, spec_types) for idx in range(nbins)]

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=A_params,
            operation_hint="multiplicative",
        )
        self.A_params = A_params
        self.spec_types = spec_types

        #arrays of ones corresponding to delta_ell=50 bins 
        bpwf = {}
        for i, spec in enumerate(spec_types):
            bpwf[spec] = window_functions[i].copy()

        for spec in spec_types:
            ells, bins = np.where(bpwf[spec] > 0.01)
            bpwf[spec] = bpwf[spec].at[(ells,bins)].set(1) #setting to 1 all bpwf > 0.01
            ells, bins = np.where(bpwf[spec] <= 0.01)
            bpwf[spec] = bpwf[spec].at[(ells,bins)].set(0) #setting to 0 all bpwf <= 0.01
        
        self.bpwf = bpwf

    def output(self, sample_params):

        arr_scale = self.bpwf.copy()

        for par in self.A_params:
            spec = par[1:3]
            bini = int(par[3:])-1
            arr_scale[spec] = arr_scale[spec].at[:, bini].set( arr_scale[spec][:,bini] * sample_params[par]   )

        return jnp.hstack([ jnp.sum( arr_scale[spec], axis=1) for spec in self.spec_types  ])

    def transform(self, Dls, sample_params):

        return Dls * self.output(sample_params)



class Aphiphi(candl.transformations.abstract_base.Transformation):
    """
    Class to scale Aphiphi per bin relative to some fiducial Clphiphi spectrum

    """
    def __init__(
        self, 
        ells, 
        crop_mask, 
        Aphiphi_params=["App1","App2","App3"], #number of bandpowers 
        descriptor=""):

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=Aphiphi_params,
            operation_hint="multiplicative",
        )
        self.Aphiphi_params = Aphiphi_params

    def output(self, sample_params):
        """
        Return an array of Aphiphi scalings to multiply to the Dls
        """
        aphiphi_arr = []
        for i, aphiphi_par in enumerate(self.Aphiphi_params):
            aphiphi_arr.append( sample_params[aphiphi_par] ) 

        return jnp.array( aphiphi_arr )

    def transform(self, Dls, sample_params):

        return Dls * self.output(sample_params)


class Aphiphi_recon(candl.transformations.abstract_base.Transformation):

    def __init__(self,
        ells,
        crop_mask,
        descriptor=""
    ):
        super().__init__(
            ells,
            descriptor=descriptor,
            param_names=["Arec", "Alens"],
            operation_hint="multiplicative",
        )
    
    def output(self, sample_params):
        return (sample_params['Arec']/sample_params['Alens'])

    def transform(self, Dls, sample_params):
        #import pdb;pdb.set_trace()
        jax.debug.print('AAAAAA {x}',x=self.output(sample_params))
        return Dls * self.output(sample_params)


        
class MllCalBeamBaseClass(
        candl.transformations.common_lensing.ResponseFunctionM
):
    """
    Base Class to be inherited by LensOnlyResponseCorrCMBliteBP

    def output() returns 0
    """
    def __init__(
        self,
        ells,
        M_matrices,
        fiducial_correction,
        crop_mask,
        mainbeam_template_arr,
        Tbeam_template_arr,
        Tbeameigmodes_template_arr,
        ilcTweight_template_arr,
        ilcEweight_template_arr,
        overwrite_ell_max={},
        fix_cal=False,
        fix_beam=False,
        tcal_param="Tcal",
        pcal_param="Pcal",
        teigenbeam_params=["beam1", "beam2", "beam3", "beam4", "beam5", "beam6"],
        polbeam_params=["beta_pol_90", "beta_pol_150", "beta_pol_220"],
        descriptor="",
    ):

        super().__init__(
            ells=ells,
            M_matrices=M_matrices,
            fiducial_correction=fiducial_correction,
            crop_mask=crop_mask,
            descriptor=descriptor,
            overwrite_ell_max=overwrite_ell_max,
        )
        self.lmaxCMB = {
            st: self.required_spectra["Dl"][st] - 1
            for st in list(self.required_spectra["Dl"].keys())
        }

        self.param_names = [tcal_param, pcal_param] + polbeam_params + teigenbeam_params

        self.fix_cal = fix_cal
        self.fix_beam = fix_beam

        self.tcal_param = tcal_param
        self.pcal_param = pcal_param
        self.teigenbeam_params = teigenbeam_params
        self.polbeam_params = polbeam_params

        # load physical (main) beam and T beam and T beam err eigenmodes
        self.blm = {
            ky: jnp.array(np.array(mainbeam_template_arr[ky]), dtype=float)
            for ky in mainbeam_template_arr.files
        }
        self.blT = {
            ky: jnp.array(np.array(Tbeam_template_arr[ky]), dtype=float)
            for ky in Tbeam_template_arr.files
        }
        self.blerrT = jnp.array(Tbeameigmodes_template_arr)

        # load ILC pol weights
        self.ilct = jnp.array(ilcTweight_template_arr)
        self.ilce = jnp.array(ilcEweight_template_arr)

        # load applied/ILC pol T beam
        self.BlPilc = self.compositeAppliedPolBeam()
        self.BlTilc = self.TbeamILC()

    @partial(jit, static_argnums=(0,))
    def compositeRelTBeam(self, sample_params):
        """
        Bl = Bl^fid + dBl

        Dl --> Dl * (1 + dBl/Bl^fid)^2  (i.e. the data bandpowers were not debiased sufficiently by this factor)

        Returns ILCTweights ( \sum_i beta_i * BlT_eigenmode_i )

        """
        lmax = self.lmaxCMB["TT"] + 2

        blT_pert = self.perturbedTBeam(sample_params)

        blT_eff = (
            self.ilct[0, :lmax] * blT_pert["90"][:lmax]
            + self.ilct[1, :lmax] * blT_pert["150"][:lmax]
            + self.ilct[2, :lmax] * blT_pert["220"][:lmax]
        )
        blT_out = (blT_eff / self.BlTilc)[2:]
        blT_out = jnp.where(blT_out == 0, 1.0, blT_out)

        return blT_out

    @partial(jit, static_argnums=(0,))
    def TbeamILC(self):

        lmax = self.lmaxCMB["TT"] + 2

        blT_ilc = (
            self.ilct[0, :lmax] * self.blT["90"][:lmax]
            + self.ilct[1, :lmax] * self.blT["150"][:lmax]
            + self.ilct[2, :lmax] * self.blT["220"][:lmax]
        )

        # catch zeros
        blT_ilc = jnp.where(blT_ilc == 0, 1.0, blT_ilc)

        return blT_ilc

    @partial(jit, static_argnums=(0,))
    def perturbedTBeam(self, sample_params):
        """
        returns per-freq BlT + dBlT
        """
        total_dBl = 0
        for i, beam_param in enumerate(self.teigenbeam_params):
            total_dBl += sample_params[beam_param] * self.blerrT[i]

        lmax = 3500
        lmin = 350
        lrange = lmax - lmin + 1

        dBl = {}
        dBl["90"] = total_dBl[:lrange]
        dBl["150"] = total_dBl[lrange : 2 * lrange]
        dBl["220"] = total_dBl[2 * lrange : 3 * lrange]

        # lmax TT
        lmaxTT = self.lmaxCMB["TT"] + 2  # 3501 for fiducial
        lmax0 = lmax if lmax > lmaxTT else lmaxTT

        blT_pert = {}
        for freq in dBl.keys():
            tmp = self.blT[freq].copy()[:lmax0]
            tmp = jax_optional_set_element(
                tmp, np.arange(lmin, lmax0), tmp[lmin:lmax0] + dBl[freq][: lmax0 - lmin]
            )
            # tmp[lmin:lmax0] += dBl[freq][: lmax0 - lmin]
            blT_pert[freq] = tmp

        return blT_pert

    @partial(jit, static_argnums=(0,))
    def compositeAppliedPolBeam(self):
        """
        Return ILC polarization beam applied in Cinv filter
        (Correct Pol beam normalization)
        """
        lmax = self.lmaxCMB['EE']+2 #+2 because these files start at ell=0

        #from 'best_fit_20250531_all_rc4_unblind.npz'
        betapol_bestfit = {90:0.53562725 , 150:0.68532188 , 220:0.6580058}

        blP = {}
        for freq in [90, 150, 220]:
            blmmain = self.blm['B_ell_%i'%freq][:lmax]
            blP[freq] = ( blmmain + betapol_bestfit[freq] * (self.blT['%i'%freq][:lmax] - blmmain) )
            blP[freq] /= blP[freq][800]  #normed

        blP_eff = ( self.ilce[0, :lmax]*blP[90][:lmax]
                  + self.ilce[1, :lmax]*blP[150][:lmax]
                  + self.ilce[2, :lmax]*blP[220][:lmax] )

        #catch zeros
        blP_eff = jnp.where(blP_eff == 0, 1.0, blP_eff)

        return blP_eff

    @partial(jit, static_argnums=(0,))
    def compositeRelPolBeam(self, sample_params):
        """
        Returns the sampled polarization beam given ILC weights relative to the assumed polarization beams
        in the Cinv filter. (Correct pol beam normalization)

        """
        lmax = self.lmaxCMB['EE']+2 #+2 because these files start at ell=0

        blT_pert = self.perturbedTBeam(sample_params)

        blP = {}
        for ii, freq in enumerate([90, 150, 220]):
            blmmain = self.blm['B_ell_%i'%freq][:lmax]
            blP[freq] = ( blmmain + sample_params[self.polbeam_params[ii]]*(blT_pert['%i'%freq][:lmax]-blmmain) )
            blP[freq] /= blP[freq][800]

        blP_eff = ( self.ilce[0, :lmax]*blP[90][:lmax]
                  + self.ilce[1, :lmax]*blP[150][:lmax]
                  + self.ilce[2, :lmax]*blP[220][:lmax] )
        
        #( blP_eff/self.BlPilc )[2:] #map-space quantity; return ellmin=2
        out_blP = ( blP_eff/self.BlPilc )[2:]
        out_blP = jnp.where( out_blP == 0, 1.0, out_blP)

        return out_blP


    def output(self, sample_params):
        # to be defined in daughter class
        return

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        return Dls + self.output(sample_params)


class Mll(candl.transformations.common_lensing.ResponseFunctionM):


    def __init__(
        self,
        ells,
        M_matrices,
        fiducial_correction,
        crop_mask,
        overwrite_ell_max={},
        descriptor="",
    ):
        super().__init__(
            ells=ells,
            M_matrices=M_matrices,
            fiducial_correction=fiducial_correction,
            crop_mask=crop_mask,
            descriptor=descriptor,
            overwrite_ell_max=overwrite_ell_max,
        )

        lmaxCMB = {
            st: self.required_spectra["Dl"][st] - 1
            for st in list(self.required_spectra["Dl"].keys())
        }

        # add lmax for 'pp' and 'kk' to dictionary
        if (M_matrices["kk"].shape[0] < len(self.ells)) or (
            M_matrices["pp"].shape[0] < len(self.ells)
        ):
            lmaxCMB["pp"] = M_matrices["kk"].shape[0]
            lmaxCMB["kk"] = M_matrices["kk"].shape[0]
        else:
            lmaxCMB["pp"] = len(self.ells)
            lmaxCMB["kk"] = len(self.ells)

        self.lmaxCMBpk = lmaxCMB

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return the correction.
        Needs to use unbinned (from params) spectra.

        Attributes
        ----------------
        sample_params dict
            Contains the unbinned (pp, TT/TE/EE/BB) theory spectra to be used in calculating (M * C).

        Returns
        ----------------
        array, float
            Correction to the model Dlphiphi/Clkk given changes in the response and the N1
            at the sampled TT/TE/EE and phiphi/kk, respectively.
        """

        # M * (Cth) - M * (Cfid)

        # get required modes and fiducial correction
        M_modes = self.M_matrices.keys()
        M_correction = -self.fiducial_correction

        # multiply arrays according to length of theory spectra
        for mode in M_modes:
            #if mode=="kk" and ("App1" in sample_params), then scale the M_corr with Aphiphi
            if mode=="kk" and ("App1" in sample_params):
                print("in kk + App1 in sample_params")

                nbins = len(self.M_matrices[mode].transpose())
                app_arr = []
                for bini in range(nbins):
                    app_arr.append( sample_params["App%i"%(bini+1)] )

                M_correction += jnp.dot(
                    np.transpose(self.M_matrices[mode][: self.lmaxCMBpk[mode]]),
                    jnp.block(sample_params["Dl"][mode][: self.lmaxCMBpk[mode]]),
                )*jnp.array(app_arr)

            else:
                M_correction += jnp.dot(
                    np.transpose(self.M_matrices[mode][: self.lmaxCMBpk[mode]]),
                    jnp.block(sample_params["Dl"][mode][: self.lmaxCMBpk[mode]]),
                )

        return M_correction

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform the input spectrum.

        Attributes
        ----------------
        Dictionary of Dls : dict of array (float)
            The binned spectra (pp or kk) to add to (M * C).
        sample_params dict
            Contains the unbinned (pp, TT/TE/EE) theory spectra to be used in calculating (M * C).

        Returns
        ----------------
        array : float
            Linear-correction-applied Dlphiphi/Clkk theory model at sampled cosmology (represented by the Dls in sample_params)
        """

        return Dls + self.output(sample_params)

class LensOnlyResponseCorrCMBliteBP(MllCalBeamBaseClass):

    def __init__(
        self,
        ells,
        M_matrices,
        fiducial_correction,
        crop_mask,
        Dl_data_template_arr,
        mainbeam_template_arr,
        Tbeam_template_arr,
        Tbeameigmodes_template_arr,
        ilcTweight_template_arr,
        ilcEweight_template_arr,
        overwrite_ell_max={},
        fix_cal=False,
        fix_beam=False,
        tcal_param="Tcal",
        pcal_param="Pcal",
        teigenbeam_params=["beam1", "beam2", "beam3", "beam4", "beam5", "beam6"],
        polbeam_params=["beta_pol_90", "beta_pol_150", "beta_pol_220"],
        descriptor="",
    ):
        super().__init__(
            ells,
            M_matrices,
            fiducial_correction,
            crop_mask,
            mainbeam_template_arr,
            Tbeam_template_arr,
            Tbeameigmodes_template_arr,
            ilcTweight_template_arr,
            ilcEweight_template_arr,
            overwrite_ell_max=overwrite_ell_max,
            fix_cal=fix_cal,
            fix_beam=fix_beam,
            tcal_param=tcal_param,
            pcal_param=pcal_param,
            teigenbeam_params=teigenbeam_params,
            polbeam_params=polbeam_params,
            descriptor=descriptor,
        )

        # Dl_data_file format: ell, TT, TE, EE
        spec_ix = {
            "TT": 1,
            "TE": 2,
            "EE": 3,
        }
        self.Dl_data = dict()
        for s in list(self.M_matrices.keys()):
            if s in ["TT", "TE", "EE"]:
                self.Dl_data[s] = Dl_data_template_arr[:, spec_ix[s]]

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):

        M_modes = self.M_matrices.keys()
        M_correction = -self.fiducial_correction

        if self.fix_cal:
            cal_fac = {"TT": 1.0, "TE": 1.0, "EE": 1.0}
        else:
            cal_fac = {
                "TT": sample_params[self.tcal_param] ** 2,
                "TE": sample_params[self.tcal_param] ** 2
                * sample_params[self.pcal_param],
                "EE": sample_params[self.tcal_param] ** 2
                * sample_params[self.pcal_param] ** 2,
            }
        if self.fix_beam:
            beam_fac = {"TT": 1.0, "TE": 1.0, "EE": 1.0}
        else:
            relBlP = self.compositeRelPolBeam(sample_params)
            relBlT = self.compositeRelTBeam(sample_params)
            beam_fac = {
                "TT": relBlT**2,
                "TE": relBlT[: self.lmaxCMB["TE"]] * relBlP[: self.lmaxCMB["TE"]],
                "EE": relBlP**2,
            }
        # multiply cal_fac because Dl is data (and hasn't had calibration factors marginalized)
        # multiply beam_fac (as would be done if Dl_data were theory spectra) because beam variations were
        # marginalized and these are "unbiased spectra"(modulo calibration)
        # (^ divide beam_fac because data should be treated as opposite to theory; even if data is perfectly calibrated
        #    with correct beam etc, these parameters account for increased uncertainties to clphiphi)
        # in this case probably makes sense to impose beta_pol posterior given TT/TE/EE chains
        for mode in M_modes:
            M_correction += jnp.dot(
                np.transpose(self.M_matrices[mode][: self.lmaxCMB[mode]]),
                jnp.block(
                    self.Dl_data[mode][: self.lmaxCMB[mode]]
                    * cal_fac[mode]
                    / beam_fac[mode]
                ),
            )
            '''
            M_correction += jnp.dot(
                    self.N0_Mmat[mode][:, 2:self.lmaxCMB[mode]+2],
                    jnp.block(
                        self.Dl_data[mode][: self.lmaxCMB[mode]]
                        * cal_fac[mode]
                        * beam_fac[mode] )
            )
            '''
        self.data_correction = M_correction

        return self.data_correction


class LensingSystematicsEmu(candl.transformations.abstract_base.Transformation):
    def __init__(
        self, ells, emu_file, 
        crop_mask, 
        emu_par_names=['Tcal', 'Pcal', 'beam1', 'beam2', 'beam3', 'beam4',
                    'betapol090', 'betapol150','betapol220',
                    'Atsz', 'Acib150', 'Acib220', 'Arad090', 'Arad150'], 
        data_set_dict=None,
        descriptor=""
    ):
        """
        emu_file: path to emulator .npz file 
        emu_par_names: list of str indicating the parameter names to be sampled through MCMC
                       it must be in the kwarg order; names can be different from those in the kwargs
                       but must match what's in the cobaya yaml
        """
        # Resolve emu_file the same way Mll resources are resolved: relative to data_set_path
        if data_set_dict is not None and not os.path.isabs(emu_file):
            base = data_set_dict.get("data_set_path", "")
            if base:
                emu_file = os.path.join(base, emu_file)

        from . import cosEmu
        self.emu = cosEmu.LensingSysEmulator_gpjax.load(emu_file)




        #dir_emul = os.path.dirname(os.path.realpath(__file__))+'/../lensysemul/'
        #self.emu = cosEmu.LensingSysEmulator_gpjax.load(emu_file)

        #self.emu = cosEmu.LensingSysEmulator_gpjax.load(dir_emul+emu_file)

        self.emu_par_names = emu_par_names
        self.crop_mask = crop_mask

        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=self.emu_par_names,
            operation_hint="multiplicative",
        )

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):
        """
        Return multiplicative factor.

        Attributes
        ----------------
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        ----------------
        float
            Multiplicative factor.
        """

        return self.emu.predict(
            jnp.array([sample_params[p] for p in self.emu_par_names]).reshape(1, -1)
        )[0][self.crop_mask]

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        """
        Transform spectrum by multiplying by overall constant factor (result of output method).

        Attributes
        ----------------
        Dls : array
            Dls to transform.
        sampled_params : dict
            Dictionary of nuisance parameter values.

        Returns
        ----------------
        array, float
            Transformed spectrum.
        """
        #When fg set to =1.3
        #[0.86044338 0.88248423 0.88666586 0.89716475 0.89472556 0.89582996
        # 0.89598905 0.90007022 0.90385511 0.91220925 0.9257862  0.93239806
        # 0.93636935 0.92771882 0.90107449 0.87877037 0.88487001]
        #np.save('fgs.py',self.output(sample_params))
        #print('BBBBBBBBB')
        #print(self.output(sample_params))
        #jax.debug.print("{x}", x=self.output(sample_params))
        return Dls * self.output(sample_params)


class LensingSystematicsEmuFG(LensingSystematicsEmu):

    def __init__(self,
        ells, emu_file,
        crop_mask,
        emu_par_names=['Atsz', 'Acib150', 'Acib220', 'Arad090', 'Arad150'],
    ):
        super().__init__(
            ells, emu_file,
            crop_mask,
            emu_par_names=emu_par_names,
        )
        self.emu_par_names = emu_par_names

        self.inst_sys_fid = [
                9.999590272289629045e-01, #tcal
                1.008417558060605401, #pcal
                0.0, 0.0, 0.0, 0.0, #beam1,2,3,4 
                5.356272497452095882e-01, #betapol90
                6.853218816169579508e-01, #betapol150
                6.580057999530474211e-01 ] #betapol220

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):

        #arr to pass to emulator
        arr = jnp.array(self.inst_sys_fid + [sample_params[p] for p in self.emu_par_names]).reshape(1, -1)

        return self.emu.predict(arr)[0][self.crop_mask]

class LensingSystematicsEmuInst(LensingSystematicsEmu):
    """
    When freeing just instrument systematics, assume there to be no foregrounds
    """

    def __init__(self,
        ells, emu_file,
        crop_mask,
        emu_par_names=[
            'Tcal', 'Pcal', 'beam1', 'beam2', 'beam3', 'beam4',
                    'betapol090', 'betapol150','betapol220'
            ],
    ):
        super().__init__(
            ells, emu_file,
            crop_mask,
            emu_par_names=emu_par_names,
        )
        self.emu_par_names = emu_par_names

        self.fg_fid = [ 0.0, 0.0, 0.0, 0.0, 0.0 ]

    @partial(jit, static_argnums=(0,))
    def output(self, sample_params):

        #arr to pass to emulator
        arr = jnp.array([sample_params[p] for p in self.emu_par_names] + self.fg_fid).reshape(1, -1)

        return self.emu.predict(arr)[0][self.crop_mask]



