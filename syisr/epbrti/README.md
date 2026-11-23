# EPB RTI editor

Canonical source: `isr_analysis/syisr/epbrti` on `main`.

Persistent SYISR event editor deployed at https://juha.no/epbrti/.

- Rust/Wasm: physical coordinate transforms, finite angular beam forward prediction, brush state.
- WebGL2: nearest-neighbor native RTI textures, zoom, edit overlay, independent cross sections.
- Rust/Axum: shared state with optimistic revisions, immutable HDF5 snapshots, atomic current pointer.
- Server data: `/mnt/shovel/epbrti`; scientific source HDF5 in `science/`, public derived textures in `public/`, edits in `snapshots/`.

Run the following from `syisr/epbrti`. The `web/` interface needs no build apart from its Wasm module:

```
cargo test -p epbrti-core
cargo build -p epbrti-core --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/epbrti_core.wasm
cargo build -p epbrti-server --release
```

Locally use base conda to export products with `tools/export_data.py ROOT DATA_DIR`.
Run the server with `EPBRTI_DATA=DATA_DIR EPBRTI_WEB=web target/release/epbrti-server`.

## Scientific conventions

UTC Unix times. Slant range stays separate from geodetic altitude. IGRF maps beam ray positions onto the plane perpendicular to B at 400 km above the radar. The co-moving field-line label is `(u-vx*(t-tref),v-vy*(t-tref))`. Hover inverts that mapping independently for every beam in the same meridian. Beam 9 is displayed and mapped but excluded from initial fits. It can still be edited manually; edits in this beam should be interpreted in light of suspected coherent contamination.

Current seed is independently fitted for each meridian using finite nine-point angular quadrature, 120 m/s eastward pattern drift, asymmetric Huber deficit loss, and a 2 km parameter grid. A finer grid does not establish independent 2 km physical resolution. Brush edits directly increase or reduce the relative echo deficit with a compact radial kernel; signed cumulative manual weights are stored alongside the resulting model. They do not re-estimate absolute density or prove field alignment. Predictions average the model over the angular beam response and scale by the quiet reference, with unavailable reference/support shown as white.

The refined fit uses signed-linear averages of native display pulse medians at15s ×1.499km, replacing the initial30s ×approximately6km inputs. Total empirical sample weight is normalized for the eightfold sample-count increase. Four spatial penalties are tested with one entire beam withheld per group (7,11,2). Selection minimizes depleted-event heldout RMS while requiring the all-event asymmetric Huber score to remain within2percent of the strongest penalty. Final maps use all eligible group beams, with9 excluded. This is a conditional prediction test for one withheld beam per meridian, not proof of physical resolution or field alignment. Original beam widths and the2km parameter grid remain unchanged.

`tools/export_refined_fit.py` writes `refined_seed.h5`, public `seed.bin`, immutable `previous_seed.bin`, and validation JSON. It never replaces the original `seed.h5` or shared snapshots. On first server startup with that fit, a new revision updates only cells still equal to the original seed with zero manual weight; every changed or painted cell remains exactly unchanged. The version is persisted, so subsequent restarts do not apply the refinement twice. Older open clients cannot save against the new fit version. The previous starting fit can be compared read-only in both model and predicted RTIs without altering working state. The RTI display-resolution selector remains independent of inversion input sampling.

Signed measured signals remain unchanged in source HDF5. Public display textures map 78–105 dB to 255 levels (0.106 dB steps); finite nonpositive estimates use the lowest color. Full native time/range samples are retained. WebGL uses nearest-neighbor texture sampling. Zoom allows native pixels to be inspected. The optional cyan overlay displays the inferred deficit and manual edit weights; it is separate from measured data.

Shared saves are public collaboration state on the requested URL. Every accepted save has an immutable revision. Before importing an older working state, the user can download their current state. A conflict keeps local unsaved work and requires an explicit reload or export; it never overwrites a newer revision automatically.

## Linked magnetic zoom

Wheel zoom and Alt-drag pan act in the co-moving IGRF `(u,v)` plane. The same selected field-line label is centred in the model and all beams of its meridian. Each RTI receives its own time and slant-range window by inverting its magnetic mapping. Curved beam tracks are sampled over the requested perpendicular span; rectangular RTI windows bound that footprint. The initial full-event atlas is restored by Reset view. Missing field-line coverage is shown explicitly, with no geometric extrapolation. A stationary wheel gesture retains its field-line anchor across events.

## Measurement views and display averaging

Default: background-matched apparent electron density, 60 s × approximately 6 km, Storm with log10 scale10–14 (display cap1e14 m^-3). The source is Memo81's calibrated arithmetic-mean power product and fixed perbeam background gain. Beam9 remains gain-uncertain. This is an apparent density under the thermal-ISR interpretation; enhancements are not automatically physical density. The GUI's morphology maps still represent relative echo deficit.

The 120 s ×12 km and240 s ×24 km levels average signed linear estimates before logarithmic encoding. The power observable additionally offers native5 s ×0.375 km and15 s ×1.5 km levels. Coarse power starts from the native display pulse-median cells, so averaging them is not a replacement for a raw-pulse arithmetic mean. Native power stays lazy-loaded until selected. Each texture retains its own exact time/range bounds; the magnetic mappings and edit coordinates are independent of display resolution. Overlay texture coordinates use native bounds separately.

Density predictions multiply the finite-beam relative-echo model by the calibrated quiet median density profile at each range, and then average in linear units to the selected resolution. This is a conditional comparison, not a new density or spectral inversion. All exported signed estimates and level coordinates are archived in `science/measurement_levels.h5`; the source is `science/background_equalized.h5`. Export with base conda: `tools/export_measurement_levels.py EVENT_ROOT DATA_DIR`.
