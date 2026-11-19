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
searched; masks include bracketing pulses. It detects transient contamination,
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
