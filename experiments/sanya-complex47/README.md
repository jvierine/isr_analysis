# Real photoelectron susceptibility: completed model change

This extends the calculations in Memos 43 and 46. Those memos describe the
earlier imaginary-only hot response; their figures are retained as historical
results. This calculation includes the real part and relocates the modes.

The model is

    S(k, omega) = 2 (M_core + M_hot) / |epsilon_core + chi_hot|^2.

The existing SAMI2-PE fluctuation source and imaginary susceptibility are
unchanged. The real susceptibility is reconstructed by the causal relation

    Re chi_hot(omega) = PV integral Im chi_hot(w) / (w - omega) dw / pi.

Both signed frequencies are evaluated. No even/odd frequency symmetry is
assumed for the pitch-anisotropic distribution. A zero-padded, nonperiodic
linear FFT convolution evaluates the discrete principal-value transform. The
kernel is -2/(pi*j) at odd j and zero at even j. The base grid spans +/-96
electron gyrofrequencies with spacing 0.00025 gyrofrequency; direct evaluation
is dense within +/-8.5, with coarser sampling and linear interpolation farther
out. Validation halves the spacing and extends to +/-112. The response is
assumed to vanish at infinite frequency, without a subtraction constant.

Thermal and enhanced roots and derivatives are calculated separately. The
enhanced roots solve Re(epsilon_core + chi_hot)=0. Frequency-bin quadrature
uses both sets of mode locations and linewidths. This avoids retaining the
thermal-only line centers after changing the dielectric. Modes are diagnostics
of the full frequency-dependent PSD, not a replacement Lorentzian model.

## Controlled comparison

`complex-comparison.json` uses the same cache, source, imaginary response and
quadrature with the real part multiplied by zero or one. This isolates the
physical model change from changes in numerical integration. Frequencies are
0.1--6 MHz in 250 Hz bins, at one degree from perpendicular.

For a weak hot correction, the leading frequency shift is approximately

    delta_f = -Re chi_hot(f_core) / [d Re epsilon_core / df](f_core).

The actual calculation solves the combined dielectric rather than using this
linear estimate. This explains why changing the real part can move a line
without greatly changing its integrated power. Fixed-frequency wing ratios
can change substantially even when the integrated change is small.

| Altitude (km) | Largest retained mode shift (Hz) | Integrated power change |
|---:|---:|---:|
| 150 | 112 | -0.000012% |
| 225.5 | 748 | +2.09% |
| 300 | 1563 | -1.06% |
| 500 | 89 | +0.0405% |
| 850 | 10 | -0.0328% |

These are selected heights, not global extrema. Total power means the integral
over the specified positive-frequency band. Individual modes can shift in
opposite directions. The effect is principally a line-frequency correction;
integrated power changes much less. At 300 km the shift is comparable to the
1953 Hz instrumental bin width used in the memo comparisons.

The displayed mode comparison applies a Gaussian instrumental response with
sigma=1595.865 Hz. Each altitude is normalized to the old folded peak, using
the SAME normalization for both curves. It is a fixed-height comparison, not
a 3 km range average. Full-profile figures additionally use the original 3 km
range averaging and common receiver/noise scaling from Memo 46.

## Numerical checks

- Signed, shifted causal oscillator: maximum real-part error less than 1e-8
  of its peak. This also checks the transform sign without an odd-symmetry
  shortcut.
- Analytic collisionless magnetized Maxwellian: error below 7e-14 of the
  peak at the sampled frequencies.
- SAMI signed-grid refinement at 150, 225.5, 300 and 850 km: real-part
  differences below 3.86e-5 of the sampled peak for the original distribution
  and 6.63e-4 for the smoothed distribution (less than 0.067%).
- Off-grid height interpolation: real-part error below 2.15e-6 of the sampled
  peak at four additional heights.
- Old versus new cached source and imaginary susceptibility: relative L1
  changes below 3e-15; maximum absolute changes below 1.7e-9 of peak.
- Reducing root scan spacing from 500 to 100 Hz and using eight-point
  quadrature everywhere changes the enhanced spectra by less than 2.7e-6
  in relative L1 at the five test heights.

These are numerical checks, not independent physical validation. Detailed
machine-readable cases are alongside this file. Model and plotting sources
are archived here and under `../../figures/complex-susceptibility/`.

## Remaining physics

Thermal collisions still use number-conserving BGK with scalar Coulomb and
MSIS neutral collision rates. Hot perturbations remain collisionless; the
SAMI2-PE input distribution itself includes transport and collisions. Adding
the real part does not implement the Milla--Kudeki Coulomb-displacement model.
The forcing, IRI core, magnetic profile, prescribed aspect angle and SAMI
transport assumptions remain those of Memos 43/46. The smoothed distribution
is still a counterfactual sensitivity test, not a better physical model.

This is the experimental suprathermal spectrum calculation. It does not
change the production RUISDAP ion-line fitter or imply that its collision
physics has been replaced.

Both original and smoothed full profiles completed on the 25 m slant-range
grid: 35,354 ranges by 23,600 frequency bins. Each has 106,515 retained
enhanced modes and no retained growing modes. Authoritative HDF5 paths and
SHA-256 hashes are in `provenance.json`; these bulk files remain on avaruus.

Theory: Longley et al. (2021), https://doi.org/10.1029/2020JA028734.
Collision limitation: Milla and Kudeki (2011),
https://doi.org/10.1109/TGRS.2010.2057253.
