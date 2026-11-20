# May 2024 storm on the SYISR cluster

User-authorized host: `ssh -p11033 SYISR_SSH_HOST` (`symaster`).
Source checkout: `/home/juha/src/isr_analysis`, main branch only.
Initial campaign: `/gpfs/pub/data/L0/2024/20240510`, `20240511`, `20240512`.
These contain approximately179GB. Outputs belong under
`/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-12-20261120`.
Never copy raw recordings into Git or run production reductions on the login host.

The first surveyed header confirms480us pulse,16000us IPP,100kHz sampling,
1324signed-int32 Q/I samples,1024-byte header,767us front and430.2MHz frequency.
Indexing validates every frame and rejects unsupported layouts or truncation.
The raw clocks are China local time, not UTC: across all62file starts in the
20240511directory, subtracting28800seconds puts every sampled beam in its
matching L2 dwell (all within30s; median13.01535s from L2 timestamp).
Neither zero nor positive8h offset aligns any sample within30s. Use
`SYISR_HEADER_UTC_OFFSET_S=28800` before forming UTC windows and blocks.
Store the offset in the manifest. Earlier November2023 products that assumed
UTC have not been retrospectively revalidated by this storm-specific check.

Matching Sanya L2 files are under
`/gpfs/pub/data/L2/SYISR_results/h5/L2/2024` with names
`OSAYA_TISR01_IFIT_L2_DAY_20240511000000_V01.00_A_LP_30sec.h5` (and10/12May).
Beamdir, Time, GEO, Fittedparameter, Error and Fitstate use the verified Sanya
schema. Only positive Ne with Fitstate0 and relative density error<=0.5 enters
the empirical calibration; match beam and30minUTC window. Median Ne profiles
are used for one fixed gain per beam over the campaign, with complete finite
pulse support. No time-dependent density profile is forced to equal L2.
IRI2020 supplies initial temperatures, heavy-ion ratios and the scale-height
coordinate, not storm composition truth. Gain is empirical, not absolute.

Use Slurm from `/opt/slurm/slurm-22.05.11/bin`, partition `node`.
At inspection node01was idle with128physical cores and257000MB RAM; node02had
another user's16-core job. Use node01for this campaign; the node03pin in the
home instructions applies to the separate YAMI3simulation.
`lp480_slurm.sh` requests64tasks x2cores,180GB, stages indexing/screening,
quiet-reference subtraction, compact products, L2 calibration and temporal
shared-composition fitting. Slurm supplies process rank; no OpenMPI runtime
is required for these independent file-parallel workers. The same solver and
objective are retained. Full10sblock products stay available for uncertainty
work. Fits use atomic checkpoints and per-run copied binary/commit metadata.

Use the established100–1400kmadaptive profile, shared skewed Hsigmoid,
He=0, beam-specificNe/Ti/Te/LOSvi, weak robust temporal priors and soft topside
temperature-slope priors with multistart branch checks. The ordinary spectral
model retains its3degperpendicular-aspect guard; two near-perpendicular looks
must not be silently claimed as fitted. Lower-hybrid full-profile support and
joint uncertainty validation remain separate unfinished work. Provisional
weak-temperature opacity is not an uncertainty interval.

Do not label an sbatch receipt as active processing: inspect squeue, stage logs,
the built binary, indexed coverage, product counts and running worker processes.

## Extension through16May

The user requested several further days to examine the negative phase/recovery.
The identified paper is Chen,Aa,Yue etal.(2025), doi10.1093/nsr/nwaf307,
https://pmc.ncbi.nlm.nih.gov/articles/PMC12587155/ . Its Fig3contains SYISR
10–13May and uses14May as the Ne reference because8–9May lack full-day coverage.
An extension through16May includes the paper's comparison day and later recovery.
Treat14May as that study's reference, not proof of completely quiet conditions.

Use a separate combined-campaign root
`/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-16-20261120` with
`BASE_RUN` pointing to the initial10–12May run. `syisr_lp480 extend` merges
new raw days, deduplicating identical pulse segments and rejecting partial
overlaps. Reuse immutable screened products/accepted initial fits only when
the full indexed observations are unchanged; recompute windows spanning the
new boundary. Keep the initial receiver-background reference and fixed gains
to avoid introducing a processing-induced density step. New times receive
date-matched IRI reference shapes. Run temporal refinement over the combined
record. Preserve the original job/results and leave node03available for YAMI3.
At extension inspection node02had112unallocated cores and~191GB free memory;
48tasks x2cores (96cores),160GB fits alongside the existing16-core other-user job.
Use sbatch CLI overrides for node02/task count/memory rather than modifying the
initial-run defaults. Full10sblocks remain accessible through product links.

Compare H fraction together with Ne and derived H density=fH*Ne; a fraction
change alone does not establish a change in the number of H ions. Retain the
existing composition and low-SNR/uncertainty qualifications.
