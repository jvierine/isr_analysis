"""MPI orchestration only; scientific reduction is the native syisr_lfm_epb binary."""
import argparse
import datetime as dt
import pathlib
import re
import subprocess
from mpi4py import MPI


def main():
    p = argparse.ArgumentParser()
    p.add_argument('binary'); p.add_argument('source'); p.add_argument('output')
    p.add_argument('--start', required=True, help='UTC ISO time, inclusive')
    p.add_argument('--stop', required=True, help='UTC ISO time, exclusive')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    comm = MPI.COMM_WORLD
    if comm.size > 64:
        raise ValueError("LFM processing is limited to 64 MPI ranks")
    start = dt.datetime.fromisoformat(a.start.replace('Z', '+00:00')).timestamp()
    stop = dt.datetime.fromisoformat(a.stop.replace('Z', '+00:00')).timestamp()
    files = None
    if comm.rank == 0:
        files = []
        for f in sorted(pathlib.Path(a.source).glob('*.mat')):
            m = re.search(r'raw(\d{12})_', f.name)
            if not m:
                raise ValueError('Unsupported filename: ' + str(f))
            t = dt.datetime.strptime(m[1], '%y%m%d%H%M%S').replace(
                tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
            # Selection is only a prefilter; the binary selects individual timestamps.
            if start - 120 <= t < stop:
                files.append(str(f))
        if a.limit:
            files = files[:a.limit]
        if not files:
            raise ValueError('No input files overlap interval')
        pathlib.Path(a.output, 'parts').mkdir(parents=True, exist_ok=True)
        print('Selected', len(files), 'files on', comm.size, 'MPI ranks', flush=True)
    files = comm.bcast(files, root=0)
    failures = []
    for i in range(comm.rank, len(files), comm.size):
        f = files[i]
        out = pathlib.Path(a.output, 'parts', pathlib.Path(f).stem + '.h5')
        # Always recompute selected parts; atomic output prevents incomplete products.
        result = subprocess.run([a.binary, f, str(out), str(start), str(stop)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            failures.append((f, result.stdout))
        print(f'rank={comm.rank} file={i+1}/{len(files)} rc={result.returncode} {result.stdout.strip()}', flush=True)
    failures = comm.gather(failures, root=0)
    failed = False
    if comm.rank == 0:
        failed = any(failures)
        if failed:
            print(failures, flush=True)
    failed = comm.bcast(failed, root=0)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
