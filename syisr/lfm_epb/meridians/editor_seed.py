from pathlib import Path
import sys,subprocess,h5py
from mpi4py import MPI
root=Path(sys.argv[1]);out=root/'editor_seed';out.mkdir(exist_ok=True)
name=['west','central','east'][MPI.COMM_WORLD.rank]
subprocess.run(['/home/juha/src/isr_analysis/syisr/target/release/lfm_epb_editor_seed','fit',str(root/f'meridian_cutouts/{name}_observations.h5'),str(out/f'{name}.h5'),'.12','0','100','-1'],check=True)
with h5py.File(out/f'{name}.h5') as h:print(name,h['diagnostics'][()],flush=True)
