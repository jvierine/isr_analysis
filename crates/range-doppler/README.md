# Range–Doppler spectra

Reusable Rust building blocks for spectra **within each transmitted pulse**.
This is a fast-time spectrum, not a slow-time FFT between pulses. There are no
radar filenames, beam counts, absolute dates, site coordinates or receiver-unit
calibration constants in the estimator.

A driver decodes its recording into I+iQ, validates its experiment, selects beams
and UTC integration windows, and supplies sample rate, taper, FFT length and gate
spacing. `RangeDopplerPlan::add_pulse` accumulates [range, frequency] power; divide
by the actual retained pulse count for each gate. Reuse a plan and one workspace
per concurrent worker. PSD units are input units²/Hz. Positive complex rotation
maps to positive frequency; physical velocity sign is the driver's responsibility.
Zero padding changes frequency sampling, not resolution.

`slant_range_km` requires explicit receive-gate, transmit-duration and receiver-delay
settings. Gate overlap is not range resolution, and slant range is not altitude.
`integration_windows` uses wall-clock time, including acquisition gaps; the driver
must retain exposure counts. Positive-PSD and signed absolute-area normalization
are separate operations; signed data are never clipped.

The first real driver is [SYISR uncoded long pulse](../../syisr/lower_hybrid).
Its binary reader, supported-layout checks, pulse ambiguity screen, beam geometry
and assumed restart correction remain instrument/experiment-specific. Coded pulse
decoding and physical parameter fitting are outside this library.

Run `cargo test --manifest-path crates/range-doppler/Cargo.toml`.

```rust
use isr_range_doppler::{hann, PeriodogramPlan, RangeDopplerPlan};
let spectrum = PeriodogramPlan::new(200_000.0, hann(64)?, 256)?;
let plan = RangeDopplerPlan::new(1024, 16, spectrum)?;
let mut work = plan.spectrum().workspace();
let mut sum = vec![0.0; plan.gate_starts().len() * plan.spectrum().bins()];
// The instrument driver provides one decoded pulse as `iq`.
plan.add_pulse(&iq, &mut work, &mut sum)?;
```
