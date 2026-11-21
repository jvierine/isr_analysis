"""Copy only display datasets; retain complete scientific products on cluster."""
from pathlib import Path
import sys,h5py,numpy as np,json
root=Path(sys.argv[1]);source=root/(sys.argv[2] if len(sys.argv)>2 else '15min-step5min.h5')
with h5py.File(source) as src,h5py.File(root/'movie-input.h5','w') as out:
 for k,v in src.attrs.items():out.attrs[k]=v
 for k in ['window_start_utc_s','window_end_utc_s','slant_range_km','frequency_hz','beam_geometry','lag_s','processing_json_utf8']:src.copy(k,out)
 for b in range(5):
  g=out.create_group(f'beam_{b}')
  for k in ['background_subtracted_psd_adu2_per_hz','psd_adu2_per_hz','normalization_received_power_adu2','perpendicular_cut_psd_adu2_per_hz','reference_cut_psd_adu2_per_hz','cut_slant_range_km','cut_linear_limits','cut_log_limits','accepted_cycles']:src[f'beam_{b}'].copy(k,g)
 commit=(root/'source-commit.txt').read_text().strip();raw_commit=(root/'blocks').resolve().parent.joinpath('source-commit.txt').read_text().strip();out.attrs['processor_git_commit']=raw_commit;out.attrs['processor_version']='syisr-lhw-ac0.1.0';out.attrs['acf_reducer_git_commit']=commit;out.attrs['acf_reducer_version']='syisr-lhw-ac-reducer0.2.0';out.attrs['source_product']=str(source)
rows=[]
for f in sorted((root/'blocks').glob('[0-9][0-9][0-9][0-9].h5')):
 with h5py.File(f) as h:
  for b in range(5):
   g=h[f'beam_{b}'];m=g['rejected_cycle_gate'][:];rows.append([int(f.stem),b,m.sum(),m.size,*g['screen_summary_counts'][:]])
with h5py.File(root/'screening-summary.h5','w') as h:
 h['block_beam_rejected_total_candidates_masked_samples']=rows;h.attrs['columns']='block,beam,rejected_cycle_gates,total_cycle_gates,code_match_candidates,masked_sample_flags_before_persistent_control';h.attrs['meaning']='exposure/support screening counts, not unique satellite objects'
print('Display subset complete',root/'movie-input.h5',flush=True)
