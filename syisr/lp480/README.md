# 480 µs uncoded long pulse

The November 2023 Sanya recording is reduced in 30-minute elapsed windows and
fitted jointly across the available monostatic beams, over 100–1400 km altitude.
H⁺ uses one common scale-height-skewed sigmoid per window; helium is fixed zero.
Density, temperatures and LOS velocity have separate adaptive cubic profiles for
each beam. IRI fixes the heavy-ion proportions and supplies an explicitly empirical,
beam-specific density calibration held constant across the recording.

`syisr_lp480` indexes the raw recording, prepares screened complex lag products,
estimates the quiet background, calibrates density and fits shared composition.
It preserves signed products and exact pulse/range support. The fitted grid and
measurement support are separate: 10 µs sampling is not 1.5 km independent
long-pulse range resolution. Unsupported lag pairs are omitted explicitly.

The data and adjacent-time priors use robust losses. Prepared 10-second blocks
retain pulse-mask counts, background products and range–Doppler spectra.
Current likelihood weights are block SEMs with reference SEM contributions;
range/lag covariance is not fully whitened and no calibrated uncertainties are
claimed. See `../agents/lp480_processing.md` for the calculation and checks.
