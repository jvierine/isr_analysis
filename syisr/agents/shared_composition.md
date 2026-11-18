# Shared composition rerun, 18 November 2026

The shared solver uses the unchanged native/adaptive decoded products and
full range/lag forward model. Each elapsed 30-minute window shares one fixed
zenith IRI scale-height shape, two H transition coordinates and three Gaussian
He coordinates. Sharing only the normalized coordinates while retaining
beam-specific fixed shapes would not produce equal fractions; therefore
the fixed shape is also common. Each beam retains its own four 46-control
plasma profiles and supplied-L2 empirical density gain.

`syisr_shared INPUT WINDOW-or-all OUTPUT PREVIOUS-or-none MAX_ITER` groups
beam jobs by UTC window start. MPI partitions windows; Rayon parallelizes
native source-cell evaluation. `fullprofile::prepare` is used by both the
old single-beam and new shared-beam paths. No raw re-decoding or new decoded
altitude outlier clipping is performed.

The five shared columns are [4,10,17,23,29] of the legacy transformed
control storage. All other H/He slots are unused, not fitted. Beam-local
normal blocks are eliminated with a Schur complement, leaving five shared
coordinates. This avoids a dense all-beam Jacobian. Bounds and projected
steps match the original plasma solver. Uncertainties are disabled.

Huber loss with threshold 3 acts on each standardized signed real/imaginary
lag residual. It downweights residual excursions without clipping measured
products or fitted fractions. Beam-level raw reduced chi-square and the
fraction of residual components beyond 3 sigma remain diagnostic, not
calibrated goodness-of-fit probabilities. Gates with over 20 percent of
valid residual components beyond 3 sigma are flagged, not silently removed.

Time regularization penalizes adjacent-window H fractions at 600:100:1300 km.
Each difference is divided by 0.15 sqrt(abs(dt)/1800 s). This is a weak
regularization setting, not a measurement uncertainty. He and the four
beam-specific plasma fields do not receive this temporal penalty.
Starting from an unregularized pooled fit, simultaneous neighboring-time
sweeps update the full beam nuisances each time. The review compares every
sweep against the same full-day Huber-plus-once-per-edge temporal objective,
and retains the best complete sweep. At most six regularized sweeps are
run; the fixed-point criterion is maximum change in H fraction below 0.005
over 500:10:1300 km plus converged window solves. This is an iterative
conditional fit; a global minimum is not guaranteed. The review reports
nonconvergence rather than relabeling it as success.

Only windows with at least one supported measured beam are fitted. Boundary
windows may have fewer beams. No wholly unmeasured window is fabricated;
review.h5 explicitly records that missing observations were not interpolated.
Pooling beams can recover a zenith gap when another beam has usable data.
Two near-perpendicular/calibration-unsupported directions from the earlier
run remain excluded with their explicit error reasons.

Avaruus run root: /nfs/urdr/scratch/juha/isr_analysis/syisr-ac-shared-v7-20261118.
The detached controller runs 10 MPI ranks with 5 Rayon threads each (50 cores),
using immutable local/NFS inputs. Each beam result and each shared-window
checkpoint is written atomically. `latest` points to the latest complete
sweep; `best` points to the lowest whole-day penalized objective. Existing
independent fits and prior model-comparison plots are unchanged.

## 128-core continuation update

The user explicitly authorized a full-day 128-core run after the v7 pilot
hit 400 iterations. Active root is now
/nfs/urdr/scratch/juha/isr_analysis/syisr-ac-shared-v8-128-20261118.
The v7 pilot gate no longer controls the campaign. Run 32 MPI ranks with
4 Rayon threads each. The supplied initialization covers 51 elapsed windows;
regularization is active from the first sweep, using neighboring independent
fit curves as initialization-only conditional references. Later sweeps replace
these with the previous complete shared-composition estimates.

A zero MAX_ITER argument removes the fixed iteration limit. Termination
still requires the unchanged normalized score criterion, or is reported as
a numerical stall. State is saved atomically every 25 iterations, retaining
all beam controls, damping, history, and shape/prior context. Restart reads
that state and rejects changed prior contexts, allowing only floating-point
roundoff from the JSON/HDF5 round trip. Checkpoints are explicitly labeled
as optimizer states, not accepted fits. The physical two-beam test verifies
checkpoint resumption, unchanged objective, equal shared composition and
independent beam plasma parameters; changed-context restart is rejected.

The archived v7 pilot remains a provisional iteration-limit result. Its
1093 km H crossing, upper-bound transition width and zero He amplitude
must not be presented as validated composition.

## Unsupported first boundary window and MPI recovery

The first partial elapsed window (1759705200) has one recorded beam and no
input gated calibration fit. This produced a rank error, which aborted the
MPI group after that rank finished its other assigned window. Seven measured
window solutions and 37 optimizer checkpoints survived. Missing all gated
inputs is now handled explicitly only at a campaign boundary; an interior
missing-input condition remains fatal. The boundary emits no beam profiles.
An internal temporal-only edge state is labeled `model_inferred_only`, with
its source neighbor time, and excluded from measured/converged-fit counts.
It supplies continuity to the temporal penalty without masquerading as data.
The actual-case regression asserts zero measured windows, one inferred edge
state, and no beam-fit directory. The same 128-core campaign then resumes
its checkpoints and skips completed measured windows. No likelihood,
regularization setting, or convergence threshold is relaxed.
