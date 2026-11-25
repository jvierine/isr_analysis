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

## Completed nominal campaign and exact H/noise checks (21 November)

The time-v3 campaign is COMPLETE: final points to sweep-20-odd, all338windows
strictly converged, no unresolved windows. Global temporal audit passed after20
sweeps (max H change0; max predicted-data change norm0.0000453144 assignedsigma).
Final local snapshot/archive and inspected all-beam maps are in
`local/syisr-storm-202405/final-sweep20`. Do not restart nominal processing.

Job6075 at `/gpfs/pub/data_other/juha/syisr-h-profile-exact-20261121` completed18
exact fixed-H(1400km) cases, all strictly converged, zero unresolved width trials.
The sigmoid inversion retains multiple height roots, and widths are searched
in log width inside the best coarse bracket. All beam plasma profiles refit;
time priors are removed, altitude priors retained. These conditional working
costs do not establish confidence coverage under correlated residuals. Ten May
18UTC has competing low/high-H solutions; the two12May checks favor low H
without temporal priors. Inspect exact-profile.png and retain that qualification.
Soft-constraint cases from job6062 were invalid near saturated sigmoid endpoints
and are superseded; never use them as profile confidence points.

The four-day11–14May solar-noise regression now removes a separate intercept
and linear time drift in each30minwindow, weighted by pulse count/noise-power²,
with errors clustered by contiguous beam dwell. Real10us slope is-0.034204
+/-0.001006 conditional regression SE (106dwells), versus quiet normalized ACF
-0.023701. Day-specific estimates agree. The quiet ACF contains a component
that does not scale like solar-correlated noise; physical attribution remains
unresolved. No new receiver subtraction has been adopted. Native noise-refine
and noise-sensitivity preserve production inputs. The latter compares existing
median scaling against pair-weighted scaling and a quiet-anchored variable-noise
response on actual gate/lag products, retaining signs and original diagonal
weights. This is a sensitivity diagnostic, not revised uncertainties or a
validated receiver filter. Inspect the rendered noise-response-refined plot.

Read-only job6089 completed44beam-windows (four30minperiods, all11looks).
The quiet-anchored variable-response alternative shifts zenith shortlags by up
to4.13existing assignedsigma at12May04:30UTC, versus0.14at14UTC. Those are
correlated per-product shifts, not independent significance tests.

Warm-started job6092 at `/gpfs/pub/data_other/juha/syisr-noise-pilot-20261121`
completed eight isolated fits (mean and quiet-anchored alternatives, fourtimes),
all strictly converged. It retained the nominal gain, gating, working weights,
altitude priors and neighboring-time priors. At12May04:30, zenith Ti(800km)
changed1676->2023K and Te3006->2438K in the anchored case; at05UTC, Ti(1200km)
changed3528->4719K, Te4179->2958K. These substantial sensitivities do not certify
the alternative subtraction. H stayed on its saturated near-zero sigmoid in
these two windows; do not interpret unchanged H as demonstrated robustness.
The two quieter cases changed H(1400km) by about0.03and0.024. Local inspected
plots and compact outputs are in `local/syisr-storm-202405/noise-pilot`.

Raw-voltage coherence job6097 is the next read-only diagnostic at
`/gpfs/pub/data_other/juha/syisr-noise-coherence-20261121`, pinned source06a347c,
two cores onnode01. It repeats the exact full-window satellite screen, splits
retained pulses into disjoint alternating groups in each10sblock, and computes
symmetrized cross-products of their mean complex far-range profiles. This avoids
the positive variance bias of squaring one noisy mean when incoherent pulse
fields are independent. Test injection passed. No raw or fitted measurements
are altered. Inspect COMPLETE/FAILED and measured coherent ACF before attributing
the quiet floor to instrument DC, RFI, or plasma, and before changing subtraction.
The previous heartbeat remains deleted; no new monitor was created.

Raw coherence job6097 stopped on a fully rejected block; fixed handling records
such blocks as excluded instead of fabricating an estimate. Replacement6099 at
`/gpfs/pub/data_other/juha/syisr-noise-coherence-v2-20261121` completed59blocks,
four exclusions. Longlag (100–470us) cross-pulse coherent ACF / quietN0 is
0.005923,0.008706,0.008433,0.007035 at the four diagnostic times, versus total
far-range longlag0.005948,0.007621,0.008288,0.007054. Quiet-reference longlag
mean is0.006569. Thus a pulse-coherent field explains nearly all of the quiet
far-range longlag floor. The field's physical origin is not established, but
scaling its quiet contribution by total solar noise gain (up to22.4 here) is
clearly inappropriate. Native compact diagnostic is local solar-noise-dwells/
noise-coherence.h5. Do not mistake the conditional cross-mean estimate for a
validated system-noise calibration.

An isolated raw covariance pilot is implemented in lp480_covariance_pilot_slurm.sh.
SYISR_LP480_PULSE_COVARIANCE=1 produces a distinct immutable decoded schema.
For each screened10sblock, subtract the across-pulse complex mean separately
at every fast-time sample and multiply covariance by n/(n-1). This preserves
independent incoherent-pulse covariance in expectation and removes a constant
coherent field; numerical injection passed. It is not fast-time DC notching.
Same satellite mask and lag/pulse gating; blocks with fewer than two surviving
pulses supply no covariance measurements. Gains/references stay fixed and fits
warm start from the nominal final states with unchanged priors/bounds/convergence.
The pilot quiet background is re-estimated from its restricted four-window
set; that restriction, pulse independence and residual changing coherent signals
must be assessed before whole-campaign adoption. Do not present pilot outputs
as replacement campaign results or calibrated confidence intervals.

The covariance pilot is job6107,32cores (4Slurmranks x8Rayonthreads),node01,
root `/gpfs/pub/data_other/juha/syisr-covariance-pilot-20261121`, frozen source
e4401d0. All18LPnumerical/I/O tests passed before pushing/deployment. Check
logs/build.log, tests.log, prepare.log, fit.log and COMPLETE/FAILED. It re-reduces
only the44selected beam-windows (fourtimes), then fits all9supported looks with
shared H from the nominal final warm states. Inspect signed lag residuals,
far-range coherent floor, new quiet response, temperatures and H-basin sensitivity.
Near-zero H warm starts can remain sigmoid-saturated; unchanged H does not
establish identification. Full conditional profile checks and joint correlated
uncertainties remain necessary before campaign-wide adoption or confidence
claims. Do not start duplicate controllers or overwrite the completed time-v3
campaign; current nominal all-beam exports remain in final-sweep20 locally.

## Full May2024, sequential64core daily jobs

The user expanded the requested next-six-day continuation to the full calendar
month,1–31May2024. Preserve the completed10–16May nominal fits. Process17–31May
forward first, then9–1May backward from the measured10May boundary. At most one
daily job runs:32Slurmworkers x2Rayonthreads=64cores,node01, with afterok chaining.
All code stays on main and must be pushed before deploying a frozen source copy.
Month root: `/gpfs/pub/data_other/juha/syisr-month-lp480-202405-20261121`.

`lp480_month.py` and `lp480_daily_slurm.sh` implement the queue. Validated native
per-acquisition-folder indices are cached with source metadata signatures;
neighboring folders are merged before UTC half-open daily selection, with exact
byte-for-byte repeated-file deduplication. Every frame is still layout/clock
validated and corrupt segments remain quarantined. Recompute complete midnight
windows; do not copy earlier partial boundary products. A file-start survey of
all1875Mayfiles found the expected480us/16ms/1324sample layout. It does not
replace native full-frame validation.

All1488May30minIRI reference tables were generated offline with the existing
official Fortran driver. The new reference's scale-height coordinate agrees
with the previous same-epoch reference to6e-14km. Month references are transported
once; daily jobs have no laptop/network dependency. Fixed nominal quiet background
and supplied-L2 empirical gains remain unchanged for comparability. The separate
raw covariance pilot is not automatically promoted into this month campaign.

The closest actual fitted state seeds each new window (controls projected only
if necessary). Timestamp provenance is retained; no previous state is relabelled
as a measurement. Fixed context contributes a temporal prior only at the adjacent
measured UTC-day boundary; it cannot fill a missing time within the new day or
bridge an unmeasured gap. Later within-day sweeps use their own measured fits.
Audits count fixed boundary edge cost once and keep strict/stable/unresolved
statuses distinct. Tests cover daily UTC limits, merged boundary segments,
nearest measured seed selection and exclusion of old in-day context. Sequential
daily conditional convergence is not global monthly temporal convergence.

`jobs.h5` records Slurm IDs, dependencies and resources atomically; an existing
submission manifest prevents accidental duplicate queues. If a day fails, later
days stay dependency-blocked until the actual failure is resolved. Each completed
day exports a compact snapshot and updates month `profiles/` and `coverage.h5`.
Actual missing observations, pending days and unsupported near-perpendicular
looks remain explicit. No interpolated measurements or confidence claims.

Submitted24daily jobs6117–6140 from frozen main056b708 on21November. Job6117is
17May;6118–6131cover18–31May;6132–6140backfill9–1May. Each requests32tasks x2
threads=64cores,120GB,node01, and afterok on the preceding job. Scientific boundary
context for9May is the completed base10–16May campaign, not31May. The24-job
submission/dependency/resource plan and duplicate-submission guard passed a
mocked-scheduler test; all21LPnumerical/I/O tests passed. The first-file header
survey is `/gpfs/pub/data_other/juha/syisr-month-202405-survey.h5` (1875files,
31folders, no unsupported first headers). All1488offline references were
losslessly transported and SHA256verified before any submission. Do not create
a second queue: inspect month jobs.h5 and actual Slurm state/logs. The standalone
background-correction pilot remains separate; nominal monthly continuation
retains the completed base's fixed gain/background and physical model.

## Nighttime H collapse audit,22November

The user correctly challenged the apparent disappearance of H in all new days.
Read-only checks at00/12/18/21/23UTC on17/18/20/25/31May show many exact
center=2000km,width=20km solutions, essentially zero H through1400km, and no
recorded alternative-start trials. A stabilized inherited fit bypassed the
general multistart block; the neighbor-start safeguard cannot detect a whole
sequence trapped in the same low-H basin. Local gradient convergence alone
does not establish a global preference for absent H. Nominal results remain
immutable and must not be presented as a verified composition disappearance.

Isolated root `/gpfs/pub/data_other/juha/syisr-month-h-basin-20261122` contains
typed signed ACF exports for20May18UTC and31May18UTC (02China local nextday),
and eight starting states: nominal,1000km/150km,1400km/250km, and1000km/150km
with lower-F-anchored cooler topside temperatures. These are initial guesses
only. Existing native Rust fits retain identical data/gain/background, bounds,
altitude/neighbor-time priors and strict0.03gradient threshold. Data-stability
early release is disabled in this diagnostic. All Ne/Ti/Te/vi and H parameters
remain free. No final products are overwritten.

Job6260 requests8workers x8threads=64cores onnode01, afterok6134 (7May).
Next daily job6135 (6May) was rerouted to afterok6260, then released. Remaining
daily dependencies remain unchanged. Verify actual Slurm state/dependencies;
never duplicate the probe or increase simultaneous fitting above64cores.
The staged orchestration source is pushed mainb8081bf; native binary is the
unchanged month/bin/syisr_lp480. Probe output must be checked before deploying
any new campaign fitting logic. No progress automation has been recreated.

Inspected ACF plots at both nighttime epochs show a persistent negative real
long-lag floor above roughly1100km and exaggerated short-lag oscillations in
the nominal prediction. The latter includes the profiled receiver-background
adjustment, not just the plasma signal. Inspect this nuisance component and
the earlier coherent-floor/covariance pilot in addition to alternative H
starts. An apparent recovery after changing background is not automatically a
validated composition measurement. Local ACF HDF5/plots are under
local/syisr-month-202405/h-basin-audit. Display normalization is absolute measured
ReR(10us), not estimated signal-only zero lag; error bars are assigned working
SEMs, not calibrated uncertainties.

## Explicit user stop,22November21:00China time

The user instructed us to stop fitting and resolve receiver-background removal
before any further H fitting. Cancel active daily job6134 (7May); preserve all
accepted fits and atomic checkpoints. Jobs6135–6140 and the alternative-start
probe6260 are USER-HELD. Both month and probe roots contain
`USER_PAUSED_H_FITTING`. Do not release, resubmit or replace these fitting jobs
while investigating the background, and do not recreate a progress automation.
Leave unrelated lower-hybrid and other users' jobs untouched. A cancellation
or resulting FAILED marker is intentional here, not a reason to auto-restart.

Read-only local comparison confirms the plotted nominal prediction above1100km
is strongly anticorrelated with the actual stored quiet-background ACF:
correlation magnitudes0.9708–0.9980 across four selected heights at two nighttime
epochs. This supports dominance of the profiled additive background term in
the short-lag oscillation; it is not a pure plasma prediction or a validated
receiver impulse response. Separate plasma, background adjustment, observed
far-range ACF and coherent mean before changing subtraction. Resolve the
negative long-lag floor and assess receiver/reference stability first. No H
recovery or absence claim is justified by the current maps or restart fixes.

## Receiver-only correction and all-May spectral audit

At the user's request, background work continues while all plasma/H fitting
stays held. Job6264 is receiver-only,32Slurmworkers x2threads=64cores onnode01;
root `/gpfs/pub/data_other/juha/syisr-receiver-background-20261122`, frozen main
916e0b7. It has no native fit command. Do not confuse this job with held6260.
It re-decodes two30minnighttime windows with the exact full-window satellite
screen and unbiased across-pulse covariance correction, then validates against
the independent covariance quiet reference from the completed earlier pilot.
Receiver reports finished; no plasma refits were performed.

New v4pulse-covariance observations subtract the independently measured quiet
covariance shape within each10sblock, using matching gate/lag pulse weights
rather than one window median scale. The static pulse-coherent voltage is
removed from both signal and noise with n/(n-1) correction. Block-difference SEM
and independent reference SEM are retained as working diagonal errors; these
are not calibrated parameter confidence intervals. V4data do not give plasma
optimization a free additive receiver-template amplitude. Legacy v2/v3products
retain their original semantics. New covariance/reference kind guards prevent
accidental use of the old coherent/raw background in corrected compact inputs.

At zenith1126km, mean real ACF over200–470us changes -43.8876->+1.6872ADU² on
20May18UTC and -45.8433->-1.0444ADU² on31May18UTC. Independent current far-noise
longlag residuals/zero lag are -0.0001081 and+0.0001132. Original corrected-data
files remain separate and signed; no clipping or forced H answer. Inspected
before/after PNG/PDFs and compact receiver reports are local under
`local/syisr-month-202405/receiver-check`. Upper-range plasma/reference stability,
moving interference and signal filter-response limitations must still be checked
before declaring the entire month validated or restarting H fits.

The user additionally requested end-of-record spectral windows for every day.
The same job validates/merges all1–31May raw UTC indices including neighboring
acquisition folders, retaining exact duplicate exclusions and quarantines.
It exports existing full-window screened periodograms where already reduced,
and unfiltered sparse raw diagnostics for the whole month: up to32pulses per
10sblock, four final128sample Hann windows, all frequency bins retained.
Raw and unbiased pulse-covariance PSDs remain separate. Sparse samples cannot
exclude brief missed interference; existing screened spectra also cannot prove
absence of signals removed by screening. Frequency uses recorded I/Q FFT sign,
not an independently established LOS sign. Plot window ranges as slant ranges.
Native numerical tests passed23cases. Rustplotlib receiver plots are generated
by lp480_receiver_maps; use full-month/all-beam maps plus available daily zenith
pages. Receiver sampling-window geometry explicitly includes TX, RX and the
noise-lag start region. No automatic monitor has been recreated.

## Authorized limited corrected retry,22November23:45China time

The user now requested a few days, one day at a time, superseding the fitting
pause ONLY for this corrected trial. New root:
`/gpfs/pub/data_other/juha/syisr-lp480-covariance-20240517-19-20261122`.
Frozen main6a4832d contains lp480_corrected_daily_slurm.sh and its guarded HDF5
submission driver. Jobs6273/6274/6275 cover17/18/19May, afterok in that order,
32workers x2threads=64cores per day onnode01,120GB. Verified6273RUNNING with
all528recorded beam-windows re-decoded and initial native fits underway.
The old month jobs6135–6140 and obsolete probe6260 remain held; do not release
them or automatically extend this trial to the rest of the month.

Inputs are freshly computed v4pulse covariance with unchanged full-window
satellite screening and validated/deduplicated UTC daily indices. They use the
independent affine receiver reference at receiver-background-20261122/
affine-calibration/reference/background.h5, not the old scaled raw reference.
Native reference-kind guards apply; no plasma-fitted additive receiver amplitude.
Existing same-time nominal Ne/Ti/Te/vi/composition states are initial guesses
only. Saturated low-H inherited states receive independently optimized interior
H starts, with the same objective selecting the winner. No old damping/history
or old within-day temporal priors are imported.17May has no old16May H boundary
prior.18/19May condition only their UTC boundary on the preceding corrected day;
subsequent complete within-day sweeps use their own corrected measured fits.

Preserve shared skewed H sigmoid, He=0, independent beam Ne/Ti/Te/LOSvi,
native/adaptive gridding to beam-supported1800km, physical curvature priors and
weak H-time sigma0.1. Strict gradient0.03 and data-stable status distinctions
remain unchanged. The controller automatically exports compact snapshots and
all-beam maps after each day. Background/calibration errors in the working
diagonal data SEM are not validated joint parameter uncertainties. Audit signed
ACF residuals and composition starts before claiming H recovery or absence.
No progress automation has been recreated; these Slurm jobs are detached.

## Storm baseline receiver defect confirmed,23November2026

Direct inspection of time-v3-20261120/code-commit.txt (a3832b5), its saved
input/products schema syisr-lp480-pooled-screen-v2, input/background.h5 and
final fit metadata confirms that10–16May used the affected raw-ACF quiet
reference scaling. The stored reference has neither pulse_covariance nor
variable_response. Thus the storm baseline is affected, not exempt because
its maps look plausible. Composition and temperatures, especially weak
topside estimates, are provisional pending corrected raw-data reprocessing;
the magnitude and even direction of each parameter bias need paired fits.
Do not treat the corrected17May result as validation of the old storm fits.
Preserve originals for paired signed-ACF, background and parameter comparisons.
No additional storm rerun has been launched by this audit.

New production compact/prepare paths reject legacy decoded schemas, raw
background references, missing affine calibration and nonfinite response
coefficients. Legacy HDF5 remains readable for diagnostics/warm starts.
Regression tests retain signed injected plasma ACFs under changing receiver
noise and unequal gate weights; add a specific production-input rejection test.
These checks prevent the known path, not every possible future calibration
error: independent quiet-gate residual and calibration-transfer audits remain
necessary. The17May plot-export GLIBC failure is separate from fitting.

## Authorized corrected storm rerun,23November2026

User explicitly authorized corrected10–16May rerun and requested git push.
Use a separate covariance storm root,64cores sequential UTC days; unchanged
validated raw indices, satellite screening, affine reference, original L2 gain,
shared H/He=0 model and convergence criteria. Original combined time-v3 final
states initialize only; new daily boundary priors use corrected days only.
Receiver production guards from58dbd95 apply. Optional plot failure now writes
PLOTS_PENDING instead of failing completed science and blocking later days;
EXPORT_COMPLETE certifies snapshot export, PLOTS_COMPLETE certifies rendering.
Do not misreport missing plots as completed. Legacy month jobs stay held.

User immediately changed this storm rerun allocation to128cores per day;
use64MPIworkers x2Rayonthreads. Preserve sequential day scheduling.

## Required production diagnostics,23November2026

Added native lp480_diagnostics renderer and operations/production_diagnostics.md.
It reads full fits, never compact snapshots, and measures signed (model-data)/
assigned-sigma residuals before robust weighting at actual altitude/lag gates.
All10–470us lags shown, real and imaginary separately. Background maps show
absolute retained raw periodograms and band-median-normalized spectral shape
for four final receive windows, all recorded beams including unsupported looks.
These screened averages are not a replacement for unscreened raw RFI audits.
HDF5 stores lag-wise signed/squared sums/counts, original spectral powers,
display exceedances and actual time coverage. Missing residual times produce
PARTIAL, never COMPLETE. New daily submissions attempt isolated native builds
and diagnostic export; failure leaves DIAGNOSTICS_PENDING. Generation is not
scientific acceptance; explicit diagnostic review is required before production.
Already-running frozen storm jobs unchanged; standalone export is available.
Inspected actual17May18UTC residual and17May time-spectrum example PNGs locally.

## Authorized full corrected May campaign,23November2026

User requested the entire calendar month sequentially, daily parameter and
validation maps, with temporal links only to a close previous-day last measured
fit; missing days require fresh fits. Allocation remains128cores per day,
64workers x2threads acrossnode01/node02; node03 and unrelated LHW jobs untouched.
New campaign root is syisr-month-lp480-covariance-202405-128-20261123.
Fresh per-day initialization when no corrected boundary lies within3600s;
no inherited same-time nominal profile states or old optimizer history.
Only the first measured window may use a fixed previous-day boundary. Actual
30min/60min separation enters shared-H prior scaling and the matching audit.
A missing day/gap>1h never creates a prior or synthetic fit. Existing within-day
30min edges unchanged; the one-hour exception is the previous-day boundary.
Daily input indices cover every May date; 8May26windows/9May20windows are
partial.1May47,20May46,27May46; all other dates48. All required IRI references
are cached. Preserve measured coverage rather than inventing missing times.

Use the previously validated affine covariance receiver reference, fixed supplied
L2 beam gains from the original storm calibration for comparability, He=0 and
shared skewed H sigmoid. Native/adaptive source gating and beam-specific tops
remain unchanged. All days refit under this new causal daily chain, including
storm days whose preceding boundary now comes from corrected9May if close.
Each daily job produces all-beam parameter/background/residual maps and first
10lag time-altitude maps for every supported fitted beam. Native plot binaries
are rebuilt in an isolated cluster target, eliminating old GLIBC dependence.
DAILY_COMPLETE requires parameter maps, measured residual diagnostics and
short-lag maps; missing plots prevent the next afterok day from starting.
COMPLETE still certifies numerical fits only. No automatic scientific approval
or validated parameter-uncertainty claim is implied by generated diagnostics.

Submitted jobs6340–6370 for1–31May, exact pushed code51c08ca, afterok chain.
Verified first job6340 requests128cores (64tasks x2),32tasks per node onnode01/02;
currently waiting for resources, not fitting yet. All old held nominal queues
and blocked17–19May retry jobs remain superseded, never release them.
The detached isolated cluster preflight successfully built lp480_maps,
lp480_diagnostics and lp480_residual_time in release mode in2m39s; all caches
needed for these binaries were available offline. Per-day jobs reuse this
plot-target and copy native executables into the campaign/bin directory.
Native24LP tests and day-boundary regression test passed before submission.

## Compute-node rendering recovery,23November2026

6340 completed all47 measured1May numerical windows, but validation export
failed. Despite both reportingglibc2.17, symaster libc exports a backported
GLIBC_2.18 __cxa_thread_atexit_impl symbol absent on node01/node02. The login
preflight's weak versioned reference prevented plotting binaries from loading
on compute nodes. Code2b71afd now builds plots exclusively in a fresh
compute-plot-target under Slurm, not the login plot-target; tests shell syntax.
It also skips all accepted science on export-only recovery and preserves the
original code-commit.txt separately from export-code-commit.txt.

Recovery job6376 relinked oncompute in1m37s, exported all11parameter PNGs and
is generating validation plots. Actual symbol reference now unversioned on
compute.2May6341 dependency was changed to afterok6376; later daily jobs remain
in the original chain. jobs.h5 records this recovery. No completed data were
refitted or discarded. Failed rendering stops the required afterok chain.
The white1May08:00–08:30UTC stripe has no indexed observations in any beam;
07:30 and08:30both have11recorded looks. Do not fill this stripe with fitted data.

## Full May numerical/export completion verified,24November2026

All31days have DAILY_COMPLETE. Final per-day audits sum to1433strictly
converged measured windows,0data-stable,0unresolved; all temporal checks pass.
Fresh starts occurred1May and9May; close preceding-day contexts otherwise
connected as recorded in day-context.h5. Required daily exports total13238
validation PNGs and279first-ten-lag maps, plus341parameter maps (including
unsupported-look pages). The old held queues remain obsolete; no current
month-fit job is active. This is numerical/export completion, not scientific
approval or validated joint parameter uncertainties.

All-beam combined month parameter maps rendered via Slurm job6459 in
campaign/month-plots, using symlinks in combined-snapshot to the immutable
completed daily snapshots. Exact zenith month PNG and31May parameter,
background and short-lag PNGs retrieved and visually inspected locally.
Local overview: local/syisr-month-202405/full-month/beam-az35799-el9000.png.
31May validation: local/syisr-month-202405/month-progress-20240531.
No June processing launched; the L0archive has June folders, pulse mode unverified.

## Server-independent calendar processing, 25 November 2026

July campaign syisr-month-lp480-covariance-202407-128-20261124 finished
1193 measured windows through26July;27–31July have NO_OBSERVATIONS.
All31daily jobs reached DAILY_COMPLETE. Diagnostics are generated, not yet
scientifically approved; preliminary opacity is not calibrated uncertainty.

August campaign syisr-month-lp480-covariance-202408-128-20261125 was submitted
as Slurm6579,128cores (64ranks x2threads on node01/node02). Its detached job
first generates August IRI tables, validates/indexes the archive, then submits
sequential daily fits and required exports. Missing late-July observations
must yield fresh August initialization, not a prior across the gap.

Official IRI sources/data and compiled Linux driver are installed at
/home/juha/src/iri-lp480-reference/iri-offline. For subsequent campaigns set
SYISR_IRI_DIRECTORY to this directory in the Slurm environment before running
lp480_next_calendar.sh. The launcher generates its own date-matched HDF5
references with base conda Python and the official Fortran model. Reference
HDF5 records the source/data hashes. These external model assets remain outside
Git. Existing fixed supplied-L2 gain, pulse-covariance/affine receiver background,
raw archive, dependency caches and compute-linked Rust plot binaries also live
on the cluster. A submitted campaign needs neither laptop storage nor a live
SSH session nor external internet. Preserve the native reference preflight:
missing assets must fail before daily submission, never silently substitute
another month. Retain the compute-linked renderers from the preceding campaign.
The August launch uses the pushed38da694 science/operations baseline plus its
server-local reference-generation prelude; do not mutate its running checkout.
