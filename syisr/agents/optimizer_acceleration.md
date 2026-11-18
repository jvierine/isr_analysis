# Shared-profile optimizer acceleration

## Validated derivative backend

`autodiff.rs` implements six-direction forward-mode dual arithmetic for the
local physical spectrum: log Ne, log Ti, log Te, velocity/1000, H, and
He/(1-H). Custom primitives use the Faddeeva derivative identity and the
same bilinear magnetized-electron cache derivatives as the forward model.
The full profile still uses analytic spline, H-transition, helium-envelope,
and temporal-penalty chain rules. No dense dual vector over all controls
is required. Neither the physical spectrum nor the likelihood is changed.

The 41-test suite passes. Independent difference tests cover all six local
coordinates, H/He/O mixtures, the altitude taper, magnetized/unmagnetized
electrons, electron-fluctuation switches, and the full measured-lag Jacobian.
On avaruus, one 462-cell, 9,176-product beam has residual relative error
6.89e-16, Jacobian Frobenius relative difference 4.89e-7 versus finite
differences, and maximum informative-column difference 6.11e-6.
Six alternating same-process trials with four Rayon threads give median
Jacobian times 0.49635 s (finite differences) and 0.45739 s (AD), about
1.085x. This modest kernel gain is not a whole-fit speedup claim.
Value-only trials now return an empty Jacobian instead of allocating and
zeroing a large unused dense matrix.

## Optimizer probes, same frozen measured-data checkpoints

Preconditioned projected L-BFGS avoids forming J^T J. The first window
(1759707000), already beyond 2,000 block-LM iterations, converged at score
0.0283 in 20.2 s with L-BFGS. The AD/block-LM comparison took 133.5 s for
25 extra iterations and remained at 0.118. On a second window
(1759759200), neither method converged in 25 iterations: block LM took
95.7 s, score 0.487; L-BFGS took 142.3 s, score 1.325. Therefore a blanket
L-BFGS replacement is not justified. Diagonal, adaptive-diagonal and
short-history angle-safeguarded variants remain optional experiments.
Checkpoint tests preserve the L-BFGS preconditioner and curvature history.

`SYISR_DIAGONAL_STEP` adds an optional projected Fisher-diagonal proposal
when Newton progress is tiny, accepting it only if it beats both the current
point and the Newton proposal on the exact same objective. It is not selected
for production merely because it sounds promising.

Benchmark HDF5 and native logs are retained outside Git under
/nfs/urdr/scratch/juha/isr_analysis/syisr-ac-ad-v9-20261118.
Optimizer histories may include the incoming checkpoint history; compare
objective changes against the frozen checkpoint, not blindly against the
first entry of a cumulative history.

## Evidence-stable stopping, kept distinct from gradient convergence

The shared solver originally required a strict normalized-gradient score and
could iterate indefinitely on effectively unchanged forward predictions.
`SYISR_DATA_STABILITY_STOP` checks two consecutive 25-iteration intervals:
- absolute objective change below 0.1;
- total Euclidean change in valid signed predicted lag residuals below 0.1
  in the assigned diagonal-sigma coordinates (not an RMS diluted by sample count);
- maximum change in either physical ion fraction below 0.0025 over
  500:10:1300 km.

The resulting status is `data_stable_gradient_pending`, never `converged`.
The original 0.03 gradient target and measured score are retained. The rule
establishes stability of the modeled data and fractions, not uniqueness of
all plasma controls, calibrated uncertainty, or physical validity of the
receiver/composition model. Quality flags remain visible. Uncertainties are
disabled. The plotting cube distinguishes pending, gradient-converged,
data-stable and unresolved optimizer states; orange exclamation marks identify
data-stable windows. No unmeasured beam profile is invented.

The temporal controller may stop once all windows are gradient-converged or
data-stable and the inter-sweep H fraction fixed-point criterion is met.
Review products preserve separate strict-gradient and data-stable counts.

The hard-window measurement-space check showed a 25-iteration objective
change of 0.0154, predicted-data change norm 0.0123, and maximum ion-fraction
change 4.73e-6 while the gradient score remained around 0.5. The practical
objective tolerance is 0.1 (about 5e-7 relative in that case), with the
stricter predicted-data and fraction tolerances retained. This does not
relax the separately reported strict gradient target.
