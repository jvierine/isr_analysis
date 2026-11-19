# Extended joint profiles and uncertainty display

Set `SYISR_MAX_ALTITUDE_KM=1300` consistently for raw decoding and joint fitting. Use a fresh decoded-product directory: existing 1000 km products are deliberately not overwritten. Native bottom-side cells and altitude-dependent grouping are retained. The H spline retains twice the spacing of the other four parameter splines. Empirical density calibration still comes from the original supplied L2 anchors.

Run nominal fits first (`syisr_joint INPUT all 0 NOMINAL`), then full-profile block-bootstrap fits (`syisr_joint INPUT all 48 UNCERTAINTY`) with `SYISR_JOINT_SEED_ROOT=NOMINAL/joint-fits`. At least 32 converged draws are needed for stored intervals. These are conditional intervals, not a demonstrated coverage calibration; they do not include receiver-response model uncertainty.

The plotter infers 1000 or 1300 km from the stored gate edges. H below 500 km is displayed as the prescribed zero, not a measured fraction. Raw estimates and intervals stay in HDF5 even when hidden in the image.

For an uncertainty-filtered plot, set `SYISR_MAX_95_HALF_WIDTH` to five comma-separated positive thresholds: relative half-width for Ne, Ti, Te, then absolute half-width for velocity (m/s) and H fraction. For example, `0.5,0.5,0.5,50,0.2` hides values with 95% interval half-width greater than 50% for density/temperatures, 50 m/s for velocity, or 0.2 for H. These are exploratory display thresholds, not validated acceptance criteria. Missing intervals are hidden when filtering is enabled. Prescribed H=0 below500 is exempt. Unset the variable for the unfiltered diagnostic view. Separate per-parameter uncertainty-mask datasets are exported alongside unmodified values and intervals.
