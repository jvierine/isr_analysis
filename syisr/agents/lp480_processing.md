# November 2023 LP480 processing

All implementation and deployments use isr_analysis/main. Raw source:
`/mnt/syisr/juha_data/LP/SY/20231115` on compuntu node22000. The recording is
60 GB, 1324 signed Q,I int32 samples per1024-byte header, no extra IQ pair;
11616-byte frames, 480us pulse, 16000us IPP, 100kHz sampling, front767us in the
surveyed files. Header clocks specify15–16Nov2023; filenames contain13Nov and
must not define the observation date. Index every frame and fail closed on
unsupported layouts or truncation.

For lag l, average x[k]conj(x[k+l]) for k=i-l..i+nu-1. The union of scatter-delay
support is [t_i-W,t_i+(nu-1)dt], independent of l. The exact source-cell overlap
is enumerated, including reduced W-l*dt overlap. Products are not divided by
that overlap; the finite pulse is included in the forward model. Sample support
requires i>=l and i+nu+l<=ns. Preserve missing bottomside long-lag products.
GUISDAP source reference https://github.com/ingemarh/guisdap, half_prof.m and
COR_uprog/COR_trilp were inspected. Its current analysis_gating option coalesces
same-group measurements with their coefficients/variances; do not claim this
Rust range-coupled method is an exact GUISDAP implementation. Native/source
cell integration is tested against uniform-scatter overlap and ray geometry.

Range-support.nu increases3to10samples with altitude. Summation support remains
approximately75–85km in range; output sampling/control spacing does not imply
that independent resolution. Altitude gates15km below400km increase aloft.
Forward cells5km over50–1500km use exact delay intervals. Profile splines100–1400km
are clamped on the guard cells. Fixed IRI O/NO/O2/N proportions must be included
at low heights rather than fitting an oxygen-only temperature there. The
collisionless spectral approximation and guard treatment need low-altitude
residual checks; a displayed100km value is not automatically validated.

Whole-pulse rejection reuses the2ms IQR/continuity code before integration:
128-sample Hann Doppler spectra,8-pulse incoherent averages; Q3+6IQR spectral
excess plus narrowband/neighbor support or broadband guard. Every range is
searched; masks include bracketing pulses. The v2 reducer pools the entire
same-beam integration for spectral quartiles before splitting accepted products
into10-second statistical blocks. A10-second-only screen visibly retained
multi-second compact echoes and is superseded, not treated as final. It detects transient contamination,
not satellite identities, and does not remove persistent ground scatter.

A shared complex quiet reference uses the lower20% power zenith far-range10s
blocks across the campaign. Record its SEM and retain raw blocks. Possible
plasma in the reference is not ruled out; compare alternative far gates and
background levels before physical claims. Never infer the signal FIR from the
noise ACF. Signal response defaults to delta; reference adjustment has a1%
working amplitude prior. All spectra/products remain signed after subtraction.

H sigmoid uses integral dz/H(z), H=kB*IRI_Ti/(15amu*g), cumulatively monotone;
50% altitude550–2000km, local width20–300km, same shape for all beams in a window.
H is prescribed0 below500km with a500–530km onset. He amplitude is fixed0 and
three helium derivatives/updates are disabled. Composition is not a free spline.
AD uses six local dual directions and exact analytic composition/spline chains.
Temporal H uses pseudo-Huber delta1.5, sigma0.15/30min at550..1400km. Local
logNe/logTi/logTe/vi coordinates have weak pseudo-Huber neighboring-state priors
with sigma2x[.7,.4,.5,.2] per30min; adjacent windows only. Curvature on profiles
increases with altitude. Use even/odd coordinate passes for parallel updates,
count adjacent edges once in global objective audits and preserve unresolved
optimization status. Do not convert iteration_limit/stalled into convergence.
No uncertainties are computed in this first processing test.

The full-support magnetic-aspect guard excludes358/45and358/60from the present
collisionless fit, as in the earlierAC implementation. Preserve all11beams in
Doppler products and explicitly blank/label the two unsupported parameter maps.
The other9beams share the H profile. No geometry guard is silently loosened.
A fixed IRI-based density gain is estimated per beam across the recording from
high-SNR short-lag F-peak data and exact finite-pulse IRI density weighting;
IRI controls neither individual time-dependent density nor fitted H amplitude.
32MPI ranks x4Rayon threads run detached on avaruus. Checkpoints are atomic;
0iteration limit means convergence/stall, not an arbitrary iteration cap.

Receiver noise amplitude is time/beam dependent: a checked integration had
median far-range N0=7295 versus common quiet N0=6409. Compact likelihoods now
scale the quiet complex ACF by that robust local N0 ratio and propagate its
working scale SEM. A signed injected-ACF invariant passes under doubled noise.
Node finalize.sh recompacts with this version after COMPLETE and writes
CALIBRATION_INPUT_READY; the transport must wait for that second marker.
Old products without background_scale metadata are not final v2 likelihoods.

The video background uses the median of the upper eight same-integration
range-window periodograms, with the identical128-point Hann FFT. Do not
reconstruct its DC pedestal from47ACF lags or use only two possibly contaminated
far gates: both can create artificial vertical negative stripes. Full signed
raw PSD and local display background are retained in range-doppler.h5.
Possible far-range plasma in this display background remains a limitation.
This display correction does not alter the ion-line forward model.

The358/60look has symmetric side features consistent with gyrolines/lower-hybrid
structure. At300km slant (261.5km altitude), IGRF14 gives0.253degrees offset
from perpendicular, B=38858nT and cold dense pure-O+ lower-hybrid scale6369Hz.
This is not a measured peak fit or warm finite-aspect prediction. Preserve the
features; do not identify them as interference. Quantitative interpretation
requires the full magnetized response, finite480us pulse/range weighting and
beam angular averaging. Relevant theory: Hysell, Vierinen and Sulzer2017,
doi10.1002/2017RS006283; avoid using a simple cold gyroline dispersion formula
as a validated finite-aspect prediction. The current ion-line aspect guard stays.
