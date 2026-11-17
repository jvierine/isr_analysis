# SAMI2 and GCPM comparison references

Numerical reference sampling is implemented in the Rust `sami2_reference` and `gcpm_reference` binaries; third-party model solvers remain external Fortran. `model_compare` renders their matched samples alongside the accepted fitted fractions and direct IRI comparison. External model sources, coefficients, raw runs and HDF5 products stay outside Git.

## SAMI2

SAMI2-1.00 from the sami2py repository runs six meridian planes intercepted at geographic latitude18.3492 degrees, altitude300km, longitudes100,105,109.6222,110,115,120degrees. The native grid is101 points along each of98 field lines, with apex heights150--4000km. Each simulation has24hours of spinup and26hours of retained output, approximately every15minutes. All seven species are retained. The sampled reference interpolates the seven densities first, then normalizes by their sum: H+,He+,O+,N+,NO+,O2+,N2+.

The seasonal day is2025-10-06, F10.7=133.0sfu,81day mean154.3sfu, dailyAp=4 from the historical index file. Neutral winds are HWM14; electric drifts use Fejer-Scherliess. These inputs are held constant as a diurnal reference. Compare the post-spinup UT phase to each observing-window midpoint. This is not a reconstruction of the actual October5--7 driver history. The same seasonal-day reference repeats across the plotted midnight boundary.

Interpolation uses barycentric latitude/altitude coordinates within each curvilinear meridian grid, then the bracketing actual plane longitudes and model times. It interpolates densities, preserves positive composition and includes all species in the denominator. Points outside spatial/time coverage are marked missing, not extrapolated. A small time-endpoint tolerance of0.02hour covers integer-second output timestamps. The sampler rejects incomplete outputs and fails if more than5percent of requested points are outside its grid. The current run covers all13000 requested points. A linear-field barycentric test and record-size/trailer, finite-density and species-closure checks validate the sampling mechanics.

## GCPM

The external legacyGCPM v2.4 code is taken from the local PyGCPM model distribution. Its core science routines are unchanged. The batch interface calls its own GEO-to-SM transformation and then `gcpm_v24`. Historical three-hourAp is mapped to its discrete Kp value for each query; the bundled IRI2007 uses refreshed Ap/F10.7 and IG/Rz tables for2025. The IG/Rz table starts in1980 to fit the legacy722-element input array without changing any requested historical values. Legacy Ap direct-access records are39bytes and begin1960-01-01. This refresh avoids quietly using expired2006/2009 driver tables.

GCPM retains its old IRI and IGRF routines, including magnetic-field extrapolation beyond the original epoch. Label it as a legacy exploratory reference, not a current independently calibrated2025 model. It is also not independent of IRI because it contains an IRI ionospheric background. Its H/He/O composition prescription differs substantially from the direct TBT-2015 IRI reference.

The Rust preparer creates48 independent external-model batches because the Fortran COMMON state is not thread-safe. The collector validates all13000 output records and electron/ion charge closure. Stored density units are cm^-3; plotted fractions are dimensionless and include the complete species population supported by each model. Model I/O text is an external Fortran protocol; delivered numerical data are HDF5.

## Reproduction

Compile `operations/gcpm_compare_driver.f90` against external GCPM sources. Its model-data path argument must end in `/`. On Linux include uppercase `.FOR` as well as lowercase `.for` source extensions. Prepare with `gcpm_reference prepare IRI_COMPARISON_HDF APF107_DAT CHUNK_DIR`; run each chunk in a separate process and collect with `gcpm_reference collect IRI_COMPARISON_HDF CHUNK_DIR OUTPUT_HDF`.

Compile and run SAMI2 with the external sami2py inputs. Supply the required10-row zero Fourier-coefficient `exb.inp` file even when the standard Fejer drift option is selected. Then run `sami2_reference IRI_COMPARISON_HDF SAMI2_RUN_ROOT OUTPUT_HDF`.

Render with `model_compare IRI_COMPARISON_HDF MODEL_HDF MODEL_LABEL OUTPUT`. Only the previously accepted numerical fit profiles enter the fit-minus-model plots; reference results remain available at every matched gate/time. This comparison does not calculate uncertainty or rerun the ISR inversion.
