# May 2024 storm on the SYISR cluster

User-authorized host: `ssh -p11033 SYISR_SSH_HOST` (`symaster`).
Source checkout: `/home/juha/src/isr_analysis`, main branch only.
Initial campaign: `/gpfs/pub/data/L0/2024/20240510`, `20240511`, `20240512`.
These contain approximately179GB. Outputs belong under
`/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-12-20261120`.
Never copy raw recordings into Git or run production reductions on the login host.

The first surveyed header confirms480us pulse,16000us IPP,100kHz sampling,
1324signed-int32 Q/I samples,1024-byte header,767us front and430.2MHz frequency.
Indexing validates every frame and rejects unsupported layouts or truncation.
The raw clocks are China local time, not UTC: across all62file starts in the
20240511directory, subtracting28800seconds puts every sampled beam in its
matching L2 dwell (all within30s; median13.01535s from L2 timestamp).
Neither zero nor positive8h offset aligns any sample within30s. Use
`SYISR_HEADER_UTC_OFFSET_S=28800` before forming UTC windows and blocks.
Store the offset in the manifest. Earlier November2023 products that assumed
UTC have not been retrospectively revalidated by this storm-specific check.

Matching Sanya L2 files are under
`/gpfs/pub/data/L2/SYISR_results/h5/L2/2024` with names
`OSAYA_TISR01_IFIT_L2_DAY_20240511000000_V01.00_A_LP_30sec.h5` (and10/12May).
Beamdir, Time, GEO, Fittedparameter, Error and Fitstate use the verified Sanya
schema. Only positive Ne with Fitstate0 and relative density error<=0.5 enters
the empirical calibration; match beam and30minUTC window. Median Ne profiles
are used for one fixed gain per beam over the campaign, with complete finite
pulse support. No time-dependent density profile is forced to equal L2.
IRI2020 supplies initial temperatures, heavy-ion ratios and the scale-height
coordinate, not storm composition truth. Gain is empirical, not absolute.

Use Slurm from `/opt/slurm/slurm-22.05.11/bin`, partition `node`.
At inspection node01was idle with128physical cores and257000MB RAM; node02had
another user's16-core job. Use node01for this campaign; the node03pin in the
home instructions applies to the separate YAMI3simulation.
`lp480_slurm.sh` requests64tasks x2cores,180GB, stages indexing/screening,
quiet-reference subtraction, compact products, L2 calibration and temporal
shared-composition fitting. Slurm supplies process rank; no OpenMPI runtime
is required for these independent file-parallel workers. The same solver and
objective are retained. Full10sblock products stay available for uncertainty
work. Fits use atomic checkpoints and per-run copied binary/commit metadata.

Use the established100–1400kmadaptive profile, shared skewed Hsigmoid,
He=0, beam-specificNe/Ti/Te/LOSvi, weak robust temporal priors and soft topside
temperature-slope priors with multistart branch checks. The ordinary spectral
model retains its3degperpendicular-aspect guard; two near-perpendicular looks
must not be silently claimed as fitted. Lower-hybrid full-profile support and
joint uncertainty validation remain separate unfinished work. Provisional
weak-temperature opacity is not an uncertainty interval.

Do not label an sbatch receipt as active processing: inspect squeue, stage logs,
the built binary, indexed coverage, product counts and running worker processes.

## Extension through16May

The user requested several further days to examine the negative phase/recovery.
The identified paper is Chen,Aa,Yue etal.(2025), doi10.1093/nsr/nwaf307,
https://pmc.ncbi.nlm.nih.gov/articles/PMC12587155/ . Its Fig3contains SYISR
10–13May and uses14May as the Ne reference because8–9May lack full-day coverage.
An extension through16May includes the paper's comparison day and later recovery.
Treat14May as that study's reference, not proof of completely quiet conditions.

Use a separate combined-campaign root
`/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-16-20261120` with
`BASE_RUN` pointing to the initial10–12May run. `syisr_lp480 extend` merges
new raw days, deduplicating identical pulse segments and rejecting partial
overlaps. Reuse immutable screened products/accepted initial fits only when
the full indexed observations are unchanged; recompute windows spanning the
new boundary. Keep the initial receiver-background reference and fixed gains
to avoid introducing a processing-induced density step. New times receive
date-matched IRI reference shapes. Run temporal refinement over the combined
record. Preserve the original job/results and leave node03available for YAMI3.
At extension inspection node02had112unallocated cores and~191GB free memory;
48tasks x2cores (96cores),160GB fits alongside the existing16-core other-user job.
Use sbatch CLI overrides for node02/task count/memory rather than modifying the
initial-run defaults. Full10sblocks remain accessible through product links.

Compare H fraction together with Ne and derived H density=fH*Ne; a fraction
change alone does not establish a change in the number of H ions. Retain the
existing composition and low-SNR/uncertainty qualifications.

## Higher-altitude recovery run

The user then requested Storm palettes, all-beam plots and heights above1400km.
Use `SYISR_LP480_TOP_KM=1800` for the next combined10–16May run, at
`/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-16-top1800-20261120`.
The same recording reaches~2100km slant range, not altitude. Each beam gets a
geometrically feasible top; retain valid short lags at the outer edge and omit
unavailable long lags explicitly. Spline/source domains, output and temporal
stability checks extend with the configured top. Recompute gate products when
the gate layout changes. Old1400km fits only seed higher profiles by projecting
log-profile coefficients onto the new spline; they are not completed1800km fits.
Keep the original background/gain, and store per-beam range/noise-gate support
in `range-support.h5`. Upper-range plasma in the noise estimate remains a
background sensitivity issue; support flags are not calibrated uncertainties.

The first extension job6020stopped safely on truncated source file
`20240513/SY_2024_05_09_22_06_13.dat1-1-269`:498073600bytes,
42878complete11616-byte frames plus2752bytes of an incomplete final frame.
The last complete and partial headers have valid sentinels and successive
clocks. With explicit `SYISR_ALLOW_PARTIAL_FINAL_FRAME=1`, retain all validated
complete frames and record the excluded EOF tail in manifest.quarantined_segments.
Internal corrupt complete frames still fail unless explicitly enabling the
cadence-validated recovery described below; changed experiment layouts fail.
Original raw files remain untouched; actual timing gaps and exposure loss are
not interpolated away. Tests cover strict rejection, explicit terminal-tail
quarantine and continued rejection of corrupt complete frames.

Ne/Ti/Te/H maps now use native Rustplotlib `ColorMap::storm()` pinned at
b8ea92da0d68e08ec035704d4698f50cb41d8961. LOS velocity retains its muted
diverging map and+/-100m/s. Full-resolution maps and a contact sheet cover all
11recorded looks; two near-perpendicular looks remain explicitly unfitted.

An endpoint survey of all252added-day files found three damaged sources:
269(2752-byte partial EOF),271(5440-byte partial EOF), and308(size3072bytes
short of its nominal frame multiple, with broken internal alignment). The first
higher run6026stopped on308rather than fitting misaligned IQ.
With explicit `SYISR_QUARANTINE_CORRUPT_FRAMES=1`, resynchronize only after
invalid framing, require two valid adjacent headers with16ms cadence, and
conservatively withdraw the preceding pulse because its payload may straddle
the lost bytes. Preserve every exclusion's byte range. An injected3072-byte
deletion test verifies three correct pulses survive from a four-pulse stream,
while the damaged pulse is excluded and the following frames retain exact
offsets. Default indexing stays strict. No raw data are edited.

## Active combined campaign, 20 November 2026

Job6033 uses112Slurm tasks x2Rayon threads (224cores) on node01andnode02.
It replaces the canceled base job6011; node03and other users' jobs remain alone.
The top1800root above is the active run, not the superseded extension root.
Prepared inputs contain338contiguous half-hour windows,3718beam windows,
and37647318unique pulses, reaching the end of16May. Exact byte comparison
identified seven acquisition files copied across daily folders (87,148,210,
272,334,395,396). Removing their611674duplicate pulses changed12windows;
only affected products were rebuilt. Exclusions are recorded in HDF5.
Latest-first task ordering prioritizes the newly added recovery days without
changing the objective. Initial fits are followed by complete temporal sweeps;
initial snapshots are progress products, not a completed temporal solution.
The May16acquisition restart still requires background/gain consistency checks.

## Altitude ringing correction

The user requested removal of altitude ringing using the existing fits as warm
starts. `SYISR_LP480_PROFILE_CURVATURE=1` replaces the weak coefficient-difference
penalty with curvature of evaluated logNe/logTi/logTe/vi profiles, integrated
on a5kmphysical altitude grid. Its length is max(45,72sin(el))+0.06max(z-400,0)km;
coordinate scales are0.35,0.15,0.20,0.025(the last corresponds to25m/s).
Pseudo-Huber delta30keeps oscillations approximately quadratic over the relevant
range. Native/adaptive knots, all lag data, H shape and strict convergence stay
unchanged. This prior is not an observational resolution or uncertainty claim.

Job6035pilot at epochs1715421600,1715731200,1715859000converged strictly in
39,26,50iterations. Velocity second-difference RMS fell85–93%; weighted data
cost rose15–18%. Retain this bias/residual tradeoff in quality reporting.
Use `lp480_refit_slurm.sh` with the old top1800root as BASE_RUN. Reuse current
accepted states; missing accepted states may use checkpoint nodes only.
Do not import damping/history under a changed objective. All raw products,
backgrounds, gains and references are linked unchanged, and outputs are separate.
New root: `/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-16-curvature-v2-20261120`.

## H temporal continuity pilot

After approving the altitude correction, the user identified isolated near-zero
H columns surrounded by substantial H, and requested stronger time continuity.
Job6038 tests epochs1715371200and1715522400against current curvature-v2states,
at `/gpfs/pub/data_other/juha/syisr-storm-h-time-pilot-20261120`.
`SYISR_LP480_H_TIME_SIGMA=0.1` increases the local quadratic temporal penalty
2.25fold relative to0.15, retaining pseudo-Huber delta1.5. Other parameters'
time priors are unchanged. The exact scale is saved per window, and global audit
rejects mixed adjacent objectives rather than silently using a hard-coded scale.
`SYISR_LP480_H_NEIGHBOR_START=1` tests the current solution and a neighboring
composition start only for isolated H dips>0.15with two adjacent measured fits.
Each candidate optimizes all parameters for100iterations, and only the lower
full penalized objective seeds the continuing solve. This addresses saturated
sigmoid gradients without imposing a neighboring H answer or filling data gaps.
Inspect pilot convergence, H values and lag costs before deploying campaign-wide.
Job6036continues meanwhile; do not duplicate or restart reductions.

Pilot6038finished:1715522400recovered a broad nonzero H transition with a0.044%
lower data cost;1715371200still favored negligible H from both starts. Do not
force that remaining interval or claim a confirmed detection from continuity.
All338curvature-v2initial windows strictly converged. Job6036was then superseded
by6040(224cores), root
`/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-16-time-v3-20261120`.
It uses sigma0.1and the neighboring-start safeguard, current completed initial
states as priors/warm starts, and unchanged reduced observations/calibration.
Two pilot outputs are valid initial v3states. Final temporal sweeps and joint
uncertainty/residual sensitivity checks remain required.

## Display support and pending columns

All five map panels now support alpha0.5for preliminary weak-data/residual/bound
flags. Ne requires short-lag detection>=5assigned sigma; vi additionally uses
the stored lag/temperature support and a fit-bound check. Shared H requires two
supported beams at that height and interior transition coordinates. These are
necessary heuristic support flags, not validated marginal parameter errors or
confidence intervals. H below500km remains the prescribed opaque zero. Missing
values remain missing. Per-beam `-quality.h5` files store both half-opacity and
finite-fit-value masks; raw fitted values are unchanged. Future information or
covariance diagnostics should refine these provisional checks.

The five white columns in progress-1925were pending fits at UTC11May16:30/22:00,
12May08:00,14May00:30/06:00. All are present in the338strict curvature-v2initial
fits. Full local exports are under local/syisr-storm-202405/complete-curvature;
these are altitude-corrected initial fits, not the completed stronger-temporal
solution. Do not describe them as final v3results.

Preserve the user's bottomside-structure validation request: compare matched
before/after altitude regularization, including12h/24hamplitudes/phases and
vertical scales, without identifying tides from appearance alone. Where
practical use finite-pulse lag injection/recovery with measured noise to measure
attenuation. Regularization sensitivity and robust joint uncertainty remain
unfinished acceptance checks. On20November the previous heartbeat automation
was found manually deleted; do not silently recreate it. Slurm processing is
independent and continues, but do not claim that automated monitoring is active.

## Correct H opacity interpretation and investigate solar noise

The user challenged the extensive H fading. The original whole-column rule
was too conservative: hitting a sigmoid height/width bound does not establish
large uncertainty in the fraction at every height. Remove that blanket fade;
store boundary diagnostics separately. Current H alpha only indicates insufficient
short-lag signal support across two beams or poor fit residuals, NOT parameter
errors. Corrected complete-curvature/support-only-plots exports supersede the
earlier alpha-plots display for this interpretation. None of these flags are
validated marginal uncertainty intervals.

Job6047at `/gpfs/pub/data_other/juha/syisr-h-profile-20261120` runs57conditional
profile-grid cases on16otherwise free node02cores, leaving production6040alone.
Epochs1715364000,1715522400,1715488200sample high H, a former collapsed window,
and the solar-transit interval. Each trial fixes a target H(1400km) via a
bounded transition height at widths20/100/300km and refits every beam's Ne/Ti/Te/vi.
Temporal priors are removed; current altitude regularization is retained. Exact
fixed-coordinate Schur steps were checked against reduced dense solves. Finite
width-grid results are exploratory upper bounds on profiled cost, not continuous
profile-likelihood confidence limits. Correlated residual/reference calibration
and joint uncertainty work remain necessary. Do not infer insignificance from
a small percentage change in a large chi-square objective.

The user's solar-noise suspicion has strong geometric and receiver support:
Astropy7vacuum Sun positions at18.3492N,109.6222E put daily zenith culmination
at04:37:50UTC, separations0.603,0.346,0.095,0.153,0.394,0.630,0.861degrees
on10–16May. Measured zenith background_scale peaks in04:30–05UTC at7.99,24.29,
21.71,30.48,18.99,4.53,2.92times the fixed quiet reference. On11–14May these
are14–20times adjacent03:30background. Scalar checks are in the original
top1800root/solar-noise-check.h5. This strongly supports solar pickup, not hot
ions inferred from bad fits. The 30minmedian noise scaling can dilute brief
transits: inspect10snoise amplitudes and subtraction/weights before declaring
solar-contaminated fits corrected. Do not silently mask or restart the data.

Profile job6047completed57outputs:51conditional fits strictly converged and6
target/width combinations were outside the physical transition bounds. Results
are local/syisr-storm-202405/h-profile/results (HDF5), with matching native
archive results-final.tar.gz. Do not interpret the three-width envelope as
continuous profiling: the10May18UTCnominal width is184km, between the100/300km
trial widths, and retaining that width sharply lowers the cost near its fitted
H fraction. Width refinement and correlated-likelihood calibration are needed
before confidence limits. Removing time priors also shifts the12May14UTCbest
sampled fraction from the nominal~0.37towards0.1, demonstrating prior sensitivity.
The user approved the corrected support-only full-map appearance; preserve it.
