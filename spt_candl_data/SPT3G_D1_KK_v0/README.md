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

## Data model: response corrections in detail

Both flavors share the multiplicative systematics emulator
(`LensingSystematicsEmu`) and a linear response correction built from the
precomputed M matrices in `linear_corrections/`. The response correction
comes in two forms, one theory-driven and one data-driven.

### `Mll` — theory-driven linear correction

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

- The covariance carries a Hartlap correction with `N_sims: 498` (set in the
  yaml, applied by candl).
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
