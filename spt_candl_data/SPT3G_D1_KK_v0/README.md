# SPT-3G D1 CMB lensing (kk) likelihood data

Data for the SPT-3G 2019–2020 ("D1") CMB lensing power spectrum likelihood
(Omori et al. 2026, SPT-3G Collaboration). Curved-sky quadratic-estimator
reconstruction; CMB multipole ranges lmin = 500 (TT/TE/EE),
lmax = 3500 (TT) / 3000 (TE/EE); 17 bandpower bins in L.

## Variants

Select via `candl.Like(spt_candl_data.SPT3G_D1_kk, variant="...")`:

| Variant | File | Use case |
|---|---|---|
| `GMV` (default) | `GMV/SPT3G_D1_KK_GMV.yaml` | Lensing-only chains, GMV estimator |
| `GMV_joint` | `GMV/SPT3G_D1_KK_GMV_joint.yaml` | Joint primary CMB + lensing chains |
| `PP` | `PP/SPT3G_D1_KK_PP.yaml` | Lensing-only, polarization-only (SQE) estimator |
| `PP_joint` | `PP/SPT3G_D1_KK_PP_joint.yaml` | Joint runs, polarization-only estimator |
| `GMV_agora_50XX` | `agoraGMV/...` | Agora simulation realizations (XX = 01–10) |
| `PP_agora_5001` | `agoraPP/...` | Agora realization, PP estimator |

### Lensing-only vs joint: which one do I use?

The two flavors differ in covariance and in how the lensing response is
corrected for the dependence on the primary CMB spectra:

- **Lensing-only** (`GMV`, `PP`): uses the **CMB-marginalized covariance**
  (`*_cov_CMBmarg.txt`; primary-CMB uncertainty propagated using
  Planck/ACT/SPT data). The response correction is data-driven
  (`LensOnlyResponseCorrCMBliteBP`): measured "lite" TT/TE/EE bandpowers are
  projected through the precomputed response matrices M, with only the kk
  (N1) piece remaining theory-dependent. No primary CMB theory is evaluated.
  **Do not combine this variant with a primary CMB likelihood in the same
  chain** — the response correction is built from measured CMB bandpowers,
  so the primary data would be double-counted.

- **Joint** (`GMV_joint`, `PP_joint`): uses the **non-marginalized
  covariance** (`*_cov_noCMBmarg.txt`) and a theory-driven response
  correction (`Mll` with modes kk, TT, TE, EE): the sampled theory TT/TE/EE
  are propagated through M at every step. **Only valid when run together
  with a primary CMB theory block** and intended for combination with a
  primary CMB likelihood; standalone it underestimates the error budget.

The Agora variants follow the lensing-only setup (CMB-marginalized
covariance, Mll mode kk only); the data-driven CMB correction is not needed
because the Agora skies have the fiducial primary CMB. They share the
covariance, window functions, emulator, and linear corrections of their
parent estimator directory — only the bandpowers differ per realization.

## Data model: the transformation modules

candl bins the unbinned theory C_L^kk through the bandpower window
functions first, then applies the `data_model` modules sequentially to the
binned 17-vector, in yaml order. For the lensing-only configs:

```
model = emu_ratio(nuisance params) * [W^T . C_L^kk,th]            (LensingSystematicsEmu, multiplicative)
        + M_kk^T . C_L^kk,th - fid_kk                             (Mll, additive)
        + sum_{TT,TE,EE} M_s^T . Dl_s^data - fid_TT+TE+EE         (LensOnlyResponseCorrCMBliteBP, additive)
```

The joint configs drop the third term and instead give `Mll` the modes
[kk, TT, EE, TE]. Note the ordering: the emulator ratio multiplies only the
binned theory, not the additive corrections. (candl's `Dl: kk` theory key
is C_L^kappakappa = [L(L+1)]^2 C_L^phiphi / 4, despite the "Dl" name.)

### `LensingSystematicsEmu` — multiplicative systematics/foreground correction

A Gaussian-process emulator (one GP per bandpower bin, Matern-5/2 kernel,
trained on Agora sky simulations) predicting the ratio
Clkk_binned(params) / Clkk_binned(fiducial, zero foregrounds), i.e. the
multiplicative bias of the reconstruction from instrumental systematics and
extragalactic foreground residuals.

- **Inputs (init):** `emu_file` (npz with the per-bin GP training data and
  scalers); `emu_par_names`, the ordered list of 14 sampler parameter
  names. **The order is positional and must match the emulator training
  order** (Tcal, Pcal, beam1-4, betapol 90/150/220, Atsz, Acib150, Acib220,
  Arad90, Arad150) — reordering the list silently mis-assigns parameters.
- **Inputs (per evaluation):** the 14 sampled nuisance values
  (Tcal_lens, Pcal_lens, beam1-4, beta_pol_90/150/220, Atsz, Acib150,
  Acib220, Arad90, Arad150). Never reads theory spectra.
- **Output:** a length-17 ratio vector multiplying the binned theory
  bandpowers (values ~0.86-0.94 near the training center, i.e. a 6-14%
  suppression).

### `Mll` — theory-driven linear correction

- **Inputs (init):** `M_matrices_folder` — the per-bin response files
  `linear_corrections/window_{0..16}.txt` (columns: ell, TT, TE, EE, BB,
  kk; candl reads one column per entry in `Mmodes`);
  `fiducial_correction_file` — candl sums only the columns named in
  `Mmodes` into a length-17 vector.
- **Inputs (per evaluation):** the *unbinned* theory C_L^kk
  (all configs), plus unbinned theory Dl^TT/TE/EE in muK^2 truncated to
  ell <= 3500/3000/3000 via `overwrite_ell_max` (joint configs only).
- **Output:** a length-17 additive correction,
  sum_modes M_mode^T . S_mode^theory - (same at fiducial); exactly zero at
  the fiducial cosmology by construction.

The `Mmodes` list specifies which spectra are used to compute linear
response corrections to the CMB lensing bandpowers:

- `kk` accounts for the self-dependence of the lensing spectrum on its own
  amplitude (i.e., N1).
- `TT`, `TE`, and `EE` propagate changes in the primary CMB power spectra
  to the lensing response, since the lensing reconstruction efficiency
  depends on the shape and amplitude of these spectra.

The `fiducial_correction` term is a precomputed offset that removes the
constant part of this linear expansion so that when the model is evaluated
at the fiducial cosmology, the correction equals zero. In other words, it
anchors the linearized response model to exactly reproduce the fiducial
lensing theory at the fiducial parameters.

(Primary + lensing) vs. lensing-only:

- **Joint run:** use theory-driven corrections (M acting on theory TT/TE/EE
  and kk at each step). This avoids double counting the data and ensures
  that changes in primary spectra consistently update the lensing response.
- **Lensing-only (no primary theory block):** use the data-driven shortcut
  below, which projects *measured* TT/TE/EE bandpowers (optionally
  reweighted by cal/beam) through M, while keeping only the kk piece
  theory-dependent. Doing that in a joint run would double-count the
  primary data and also tie the response to a specific noise realization.

### `LensOnlyResponseCorrCMBliteBP` — data-driven correction (lensing-only)

Builds a *data-bandpower-based* correction to the CMB-lensing response
using "lite" TT/TE/EE bandpowers instead of theory TT/TE/EE.

- **Inputs (init):** the same M matrices and fiducial-correction file as
  `Mll` (here with `Mmodes: [TT, TE, EE]`, i.e. the TT+TE+EE columns of
  the fiducial file); `Dl_data_template_file` — the measured lite TT/TE/EE
  bandpowers boxcar-expanded to per-ell Dl (columns: ell, TT, TE, EE); the
  beam templates (`beam/`) and ILC weights (`ilcweights/`) used to build
  the optional cal/beam reweighting.
- **Inputs (per evaluation):** **none as shipped.** With
  `fix_cal: True` and `fix_beam: True` the module reads no sampled
  parameters and no theory spectra — its output is a constant vector (with
  zero gradient). With the fix flags set to False it additionally reads
  Tcal_lens, Pcal_lens, beam1-4, and beta_pol_90/150/220 to reweight the
  data bandpowers. It never reads theory spectra in any configuration.
- **Output:** a length-17 additive correction,
  sum_{TT,TE,EE} M_s^T . (Dl_s^data * cal_s / beam_s) - fid_TT+TE+EE.

Operations:

1. Load fixed TT/TE/EE bandpowers D_ell^data (`Dl_data_template_file`).
2. Reweight those bandpowers by the current sample's calibration and beam
   parameters:
   - TT: multiply by Tcal^2 and divide by B_T^2
   - TE: multiply by Tcal^2 \* Pcal and divide by (B_T \* B_P)
   - EE: multiply by Tcal^2 \* Pcal^2 and divide by B_P^2

   (This propagates cal/beam uncertainty into the lensing response without
   fitting a primary theory.)
3. Project the reweighted bandpowers through the precomputed linear
   response matrices M^TT, M^TE, M^EE to obtain a lensing-response
   correction vector in L-space.
4. Subtract a precomputed fiducial correction so the net correction is zero
   at the fiducial point.
5. Return this correction; the lensing-only likelihood adds it to the
   fiducial lensing model.

What it does *not* do:

- Does not evaluate primary CMB theory spectra at each step (no CAMB/CLASS
  call).
- Does not include a primary TT/TE/EE likelihood term; the bandpowers are
  inputs, not data to fit.

When to use:

- In *lensing-only* analyses that do not run a primary theory block. It
  efficiently conditions the lensing response on the observed CMB spectra
  while letting cal/beam parameters vary.

When NOT to use:

- In *joint* CMB-primary + CMB-lensing runs. There, you should use the
  theory-driven linearization (`Mll` with modes [kk, TT, TE, EE]) to avoid
  double-counting primary data and to keep the response consistent with the
  theory spectra evaluated at each sample.

Set `fix_cal: True` and `fix_beam: True` when using lite TT/TE/EE
bandpowers, since those products already have calibration and beam
uncertainties marginalized out and you do not want to double-count them.
For analytic marginalization you want to turn the emulator off and set
these to True.

### Implementation notes

- In the shipped lensing-only configs, the nine cal/beam parameters
  (Tcal_lens, Pcal_lens, beam1-4, beta_pol_*) affect the likelihood only
  through the emulator; `LensOnlyResponseCorrCMBliteBP` declares them but
  does not use them (fix flags).
- `Mll` requires `kk` to be present in `Mmodes` (its initialization
  unconditionally accesses the kk M matrix); a TT/TE/EE-only configuration
  will fail.
- The lensing-only likelihood still *requests* theory Dl^TT/TE/EE up to
  ell 3500/3000/3000 from the theory code (a side effect of the response
  module's base class) even though it never uses them; only C_L^kk up to
  L = 4000 actually enters the result.
- The chi-square uses a Hartlap-corrected covariance with `N_sims: 498`
  (factor (N_sims - N_bins - 2)/(N_sims - 1), folded into the Cholesky
  decomposition by candl).

## Files (per estimator directory)

| File | Contents |
|---|---|
| `SPT3G_D1_KK_*_bdp.txt` | 17 Clkk bandpowers (single column) |
| `SPT3G_D1_KK_*_cov_CMBmarg.txt` | 17×17 covariance, primary-CMB-marginalized (lensing-only variants) |
| `SPT3G_D1_KK_*_cov_noCMBmarg.txt` | 17×17 covariance without CMB marginalization (joint variants) |
| `windows/kk_window_functions.txt` | Bandpower window functions; column 0 = theory L, columns 1–17 = one per bin |
| `emulator/emul_deb_*.npz` | GP emulator returning multiplicative bandpower corrections for instrumental/foreground systematics; parameters: Tcal_lens, Pcal_lens, beam1–4, beta_pol_90/150/220, Atsz, Acib150, Acib220, Arad90, Arad150 |
| `linear_corrections/window_{0..16}.txt` | Linear response matrices M, one file per bin; column 0 = L, then TT, TE, EE, BB, kk. Used for both the theory-driven (joint) and data-driven (lensing-only) response corrections. The number of `window_*.txt` files defines the bin count — do not add files matching this pattern. |
| `linear_corrections/fiducial_correction.txt` | Precomputed fiducial offset (17 rows × 5 columns: TT, TE, EE, BB, kk) anchoring the linearized response so the correction vanishes at the fiducial cosmology |
| `linear_corrections/boxcarr_lite_Dlbp_v41_fidfill.txt` | "Lite" (foreground/systematics-marginalized) TT/TE/EE data bandpowers (columns: ell, TT, TE, EE); input to the data-driven response correction (lensing-only variants) |
| `beam/B_ell_main_beam.npz` | Main (polarized) beam per frequency (keys: 90, 150, 220) |
| `beam/B_ell_rc4.npz` | Temperature beam per frequency |
| `beam/beam_eigenmodes.txt` | Temperature beam error eigenmodes (beam1–4) |
| `ilcweights/ilcweights1d_{TT,EE}.npy` | 1D ILC weights (rows: 90, 150, 220 GHz) used to build composite beams |

Notes:

- The `beam/` and `ilcweights/` files are required to initialize the
  lensing-only variants but do not affect the likelihood value as shipped
  (`fix_cal: True`, `fix_beam: True`): the lite bandpowers already have
  calibration and beam uncertainty marginalized. Setting these to `False`
  propagates cal/beam uncertainty through the response instead — only do
  this when supplying non-lite CMB bandpowers.

## Nuisance parameters

The emulator parameters (Tcal_lens, Pcal_lens, beam1–4, beta_pol_*, Atsz,
Acib*, Arad*) require external priors; see the sampler configuration files
shipped with the release.
