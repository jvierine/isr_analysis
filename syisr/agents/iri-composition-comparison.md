# Direct topside-composition comparison

The `iri_compare FIT_ROOT IRI_BATCH_DRIVER IRI_DATA_DIRECTORY OUTPUT` plot executable evaluates official IRI2020 at the actual nominal spherical beam location of every 500--1300 km gate and the UTC midpoint of its 30-minute integration. Compile `operations/iri_compare_driver.f90` with the external IRI2020 sources and historical indices; the model sources/data stay outside Git.

The reference is the TBT-2015 topside composition option (`JF(6)=false`), with percentage outputs (`JF(22)=true`) divided by100. Fractions retain IRI's total-ion normalization, including N+; no H/He/O-only renormalization is applied. This direct IRI composition is distinct from the fixed Ti/g skew used inside the custom sigmoid fit.

The comparison plots use profiles whose optimizer status begins `converged` and whose normalized gradient is below0.05. This is numerical convergence, not physical validation. Nonconverged values remain in `raw_fit_*` datasets, while the accepted-profile mask controls the plotted fit and difference. The IRI reference remains visible for every available window. Stored query coordinates, raw/selected estimates, reference O/N fractions and signed differences allow reproducing and filtering the comparison. No uncertainty is computed here. IRI is an empirical reference, not ground truth.
