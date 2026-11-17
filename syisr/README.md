# SYISR alternating-code campaign analysis

Rust implementation: `syisr_ac index`, `decode`, `fit`, `status`.
Scientific products and exact voltage masks are HDF5 and excluded from Git.

The independent [2 ms uncoded long-pulse package](ulp2ms/) includes raw
processing, full-covariance profile inversion through 5000 km, and an optional
sigmoid-H/Gaussian-He composition trial. It has its own Cargo workspace and
reproduction instructions; its waveform response is distinct from the
alternating-code decoder.

The current campaign is a research pipeline. Full-campaign numerical validation
and output review are pending. Unsupported geometries and failed/weak fits are
explicitly flagged. Below200km the O/H model omits molecular ions.

MPI launcher:24ranks,2Rayon threads each,48cores total on Avaruus;
auxiliary rendering must stay within the user's50core limit. Jobs are restartable
and independent of the originating Mac once the remote driver is running.
`operations/remote_driver.py` is transport/orchestration only and runs on the
work Linux laptop. All scientific processing is Rust. Host paths are deployment
configuration and must be adapted for another installation.

Raw processing retains complete64-code cycles,paired-endpoint satellite masks,
per-dwell quiet-background subtraction and separately detected stationary means.
A full sampled Doppler bank is searched every fourth pulse with bracketing
waveform masks; brief faint targets may escape. Every pulse is checked for
bright power bursts and clipping. No decoded amplitude-outlier rejection occurs.
Bottom-side gates are native20us baud cells through400km. Above400km,
consecutive native cells are combined into progressively wider,non-overlapping
gates, following an assumed scale-height-inspired width reaching150km at2000km.

Error products use4-cycle block resampling without crossing dwells, nuisance
refits and50percent shrinkage of the sample covariance. Bootstrap failures,
bounds and broad intervals are retained as quality flags. The15percent gain
perturbation is an assumed systematic floor, not calibrated coverage. Ne uses
supplied L2 density anchors where available; missing calibration is explicit.
Physical velocity sign, hardware delay and receiver response uncertainty are
not independently calibrated. H/O composition is conditional, not a validated
H+ detection. Robust empirical coverage validation is still required.

Build/test: `cargo test --release --lib`; `cargo build --release --bin syisr_ac`.
Native HDF5 development files are required. UTC is stored; plot labels useUTC+8.

## Native/adaptive resolution correction (v2)

Legacy v1 products used50km gate centers and are incompatible with this run.
v2 uses one20us baud cell(2.99792458km slant range) below400km; altitude
spacing is calculated separately for each beam. Above400km consecutive native
cells are grouped into progressively wider,non-overlapping windows. The
assumed scale-height-inspired target width reaches150km at2000km. Actual
gate edges and widths are stored and used directly by the renderer.
The last gate can be shorter because coverage ends; no interpolation is used.
Backwards/duplicate timestamps are quarantined with exact source offsets in
the campaign manifest; raw recordings are unchanged. Unsupported layouts fail.
