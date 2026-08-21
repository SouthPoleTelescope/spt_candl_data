# validation/a2ptarec_gmvlite

Self-contained mirror of the cobaya_files2 run

    /eagle/fieldlevel/repo/healqest/pipeline/spt3g_20192020/likelihood/cobaya_files2/data/a2ptarec_lenscmbbao/gmvspafglite_spalitetteete_desidr2_modcamb.yaml

in the SPT3G_D1_KK tree: a2ptarec (A_2pt/A_rec) joint lensing+CMB+BAO fit using
the **lite** (fg/sys pre-marginalized) GMV bandpowers instead of the released
GMV_withcmb variant. Nothing is registered in spt_candl_data.py or
SPT3G_D1_KK_index.yaml -- the data set yaml, likelihood yaml, and data files all
live in this directory (same approach as validation/sim_shift and
validation/sysvariants).

## Files

- `gmvlite_spalitetteete_desidr2.yaml` -- main cobaya config (submit via
  data_spalitetteete.submit as `validation/a2ptarec_gmvlite/gmvlite_spalitetteete_desidr2.yaml`).
  Theory/params reuse the shared `configs/theory/withcmb/camb_a2ptarec` and
  `configs/params/withcmb/cosmo_a2ptarec`. Proposal covmat seeded from the
  a2ptarec_lenscmbbao gmv chain (extra gmv nuisance columns are dropped by
  cobaya; the source config had no covmat).
- `likelihood/gmvlite.yaml` -- CandlCobayaLikelihood block, no nuisance params
  ("nocalprior": systematics pre-marginalized in the lite bandpowers),
  data_set_file -> local data set yaml (absolute path).
- `data/GMV/SPT3G_D1_KK_GMVlite_withcmb.yaml` -- candl data set: lite bdp+cov,
  Hartlap N_sims 498, Mll(kk,TT,EE,TE) with overwrite_ell_max TT3500/TE3000/EE3000,
  no LensingSystematicsEmu module.

## Data provenance (copied 2026-08-20)

Library root: /eagle/fieldlevel/repo/crux/candl_likelihoods_lens1920/spt3g_candl_library/data/1920_PP_v6/1920_PP_v6_GMV_data
Package root: /eagle/fieldlevel/repo/crux/spt_candl_data/spt_candl_data/SPT3G_D1_KK_v0/GMV

- `SPT3G_D1_KK_GMVlite_{bdp,cov}.txt` <- library `data/GMV_fid_gmv_v4_lite/lenslite_{bdp,cov}.txt`
  (17 bins; library full bdp in `data/GMV_fid_gmv_v4/` is byte-identical to the
  packaged SPT3G_D1_KK_GMV_bdp.txt, i.e. same gmv052425/v4 generation).
- `windows/`, `linear_corrections/` <- package (verified byte-identical to the
  library copies the source config points at).
- `linear_corrections/fiducial_correction.txt` is the packaged one, verified
  byte-identical to `linear_corrections_matchedlmax/fiducial_correction.txt` in
  the library GMV_sim/GMV_validation trees. Note the source config references
  `1920_PP_v6_GMV_data/linear_corrections_matchedlmax/`, which does NOT exist on
  crux (only under the lcrc DIR_CANDLDATA) -- the packaged/matchedlmax file used
  here is the correct one.
- "modcamb" theory: `configs/theory/withcmb/camb_a2ptarec` uses
  spt_candl_data.transformations.a2ptarec.a2ptarec, byte-identical to the
  library transformations/a2ptarec.py referenced by camb_a2ptarec_modcamb.yaml
  (scales lens_potential pp by Arec/Alens, tp/ep by sqrt, after CAMB Alens).
