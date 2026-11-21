# Coherent-scatter evidence audit, 1 March 2026

This separate study tests the thermal-ISR interpretation of the enhanced northward
60-degree beam. It does not redefine the cavity inversion or claim a measured
coherent fraction. All scientific products are HDF5 and stay outside Git.

## Performed

- All eleven beams reduced from existing unit-energy matched-filter pulse powers
  with arithmetic means, 60s by sixteen 0.3747405725km cells. Negative estimates
  retained, pulse medians and empirical SEM stored separately.
- Supplied LP L2 audited without an upper density cutoff; all original values,
  errors, state flags, beam directions and geometry preserved.
- L2 timestamp registered to original LP filenames and internal pulse clocks.
- IRI2020 modeled temperatures evaluated at WGS84 scattering positions. Library
  units of cm^-3 explicitly converted to m^-3 for its density reference only.
  Package version and solar/geomagnetic-index file hashes retained.
- Quiet zenith L2 anchors at 11:02:45 and11:19:46UTC,300–450km, are matched to the
  LFM mean. Median calibration coefficient protects against isolated L2 outliers;
  no upper density cutoff is used. Relative coefficients from other matched beams
  test the interbeam scale. Beam9 gain transfer remains an explicit assumption.
- Apparent thermal-ISR density, factor10 gain stress scenarios, plasma cutoff
  comparison, persistence, raw-voltage extrema, wrong-code decoding and far-noise
  diagnostics retained.
- Existing finite-beam magnetized-electron BGK thermal pilot integrated over
  ±50kHz at measured local B/aspect, varied collision rate/width and9/33angles.
  Pilot RF430.2MHz differs from actual432MHz. It is not a fitted measured spectrum,
  full LFM ambiguity model or Coulomb-accurate collision operator.
- Local ionosonde SAO directory was empty. Public GIRO SA418 year list contained
  no2026 data. This is a recorded availability check, not proof no local ionogram
  exists elsewhere.

## Reproduction

Use base conda Python locally. Source input/output paths are command arguments;
never show these code paths or build details in scientific figures or captions.
On Sanya, submit the provided audit, mean-power, raw-check, clock and thermal-null
Slurm scripts. MPI reduction uses the existing py311 environment because remote
base lacks mpi4py. Limit total concurrent task allocation to16cores and coordinate
with the parent cavity reconstruction. No node pinning is required.

The main data root is
`/gpfs/pub/data_other/juha/lfm_epb/20260301/coherent_scatter_evidence`.

Local sequence after transfer of the Slurm products:

```sh
conda run -n base python syisr/lfm_epb/coherent_evidence/iri_reference.py GEOMETRY ROOT/iri_assumption.h5
conda run -n base python syisr/lfm_epb/coherent_evidence/calibrate_power.py ROOT GEOMETRY
conda run -n base python syisr/lfm_epb/coherent_evidence/test_calibration.py
conda run -n base python syisr/lfm_epb/coherent_evidence/export_metrics.py ROOT ROOT/metrics.tex
# Build lfm_coherent_evidence from syisr/plots, then render ROOT ROOT/figures.
# From ROOT, run pdflatex twice on evidence_note.tex, render all pages and inspect.
```

## Interpretation limits

The supplied L2 absolute-calibration history remains unverified. The calibration
scatter is not an uncertainty interval. Ideal projected-aperture transfer and
factor10 stress scenarios are not measured beam9 gain confidence limits. Apparent
high density rejects the tested thermal/gain assumptions; waveform, noise and
aspect checks strengthen a coherent-scatter interpretation but do not establish
source location or exclude delayed leakage and sidelobes. The next discriminant
is a finite-beam magnetized thermal-plus-coherent spectral/voltage comparison,
with independent pulses withheld, and a contemporaneous ionosonde/plasma-line
anchor or documented L2 calibration.
