"""Process a recording into separate receiver range–Doppler HDF5 products."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes as ct
import ctypes.util
from datetime import datetime, timezone
import glob
import threading
import json
import math
from pathlib import Path
import subprocess
import sys



def gpu_devices():
    """Use CUDA runtime ordinals, honoring CUDA_VISIBLE_DEVICES."""
    candidates = [ctypes.util.find_library('cudart'), 'libcudart.so']
    candidates += glob.glob('/usr/local/cuda*/lib64/libcudart.so*')
    candidates += glob.glob('/usr/lib/x86_64-linux-gnu/libcudart.so*')
    runtime = None
    for name in filter(None, candidates):
        try:
            runtime = ct.CDLL(name)
            break
        except OSError:
            continue
    if runtime is None:
        raise RuntimeError('CUDA runtime library not found. Add CUDA lib64 to LD_LIBRARY_PATH.')
    count = ct.c_int()
    error = runtime.cudaGetDeviceCount(ct.byref(count))
    if error or count.value == 0:
        raise RuntimeError(f'CUDA device discovery failed (error {error}, count {count.value})')
    devices = []
    for device in range(count.value):
        if runtime.cudaSetDevice(device) == 0 and runtime.cudaFree(ct.c_void_p()) == 0:
            devices.append(device)
        else:
            print(f'Skipping unavailable GPU {device}', flush=True)
    if not devices:
        raise RuntimeError('No usable CUDA GPUs')
    return devices


def process_minute(raw, output, epoch, device):
    here = Path(__file__).resolve().parent
    stamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    for center in (436000000, 444000000):
        folder = output/'dual-channel-hour'/str(int(epoch))/str(center)
        folder.mkdir(parents=True, exist_ok=True)
        diagnostic = folder/'diagnostic.h5'
        matched = folder/'matched-step2us.h5'
        if matched.exists():
            print(f'REUSE {stamp} {center//1000000} MHz', flush=True)
            continue
        print(f'START GPU {device} {stamp} {center//1000000} MHz', flush=True)
        with (folder/'processing.log').open('a') as log:
            if not diagnostic.exists():
                subprocess.run([sys.executable, '-u', str(here/'process.py'), str(raw),
                                str(diagnostic), '--start', str(epoch), '--catalogue',
                                str(output/'catalogue.h5'), '--device', str(device),
                                '--receiver-center-hz', str(center)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, '-u', str(here/'matched.py'), str(raw),
                            str(diagnostic), str(matched), '--device', str(device),
                            '--range-step-us', '2'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'DONE GPU {device} {stamp} {center//1000000} MHz', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON configuration file')
    parser.add_argument('--data', type=Path, help='Raw recording directory')
    parser.add_argument('--output', type=Path, help='Product directory')
    parser.add_argument('--start', type=float, help='First integration Unix time in UTC')
    parser.add_argument('--minutes', type=int, help='Optional limit on 60-second integrations (default: entire directory)')
    parser.add_argument('--device', type=int, help='Use only this CUDA device (default: all usable GPUs)')
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    for key in ('data', 'output', 'start', 'minutes', 'device'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if not config.get('data') or not config.get('output'):
        parser.error('--data and --output are required, or provide them in --config')
    raw = Path(config['data']).expanduser().resolve()
    output = Path(config['output']).expanduser().resolve()
    if not raw.is_dir():
        parser.error('Data directory does not exist')
    output.mkdir(parents=True, exist_ok=True)
    from process import catalogue, layout
    indexed = catalogue(raw, output/'catalogue.h5')
    if not indexed:
        parser.error('No recording files found')
    start = float(config.get('start', indexed[0][0]))
    stop = max(t + (path.stat().st_size // info['stride']) * info['ipp'] * 1e-6
               for t, path in indexed for info in [layout(path)])
    if start >= stop:
        parser.error('Start is beyond the recording')
    minutes = math.ceil((stop-start)/60)
    if 'minutes' in config:
        limit = int(config['minutes'])
        if limit < 1:
            parser.error('minutes must be positive')
        minutes = min(minutes, limit)
    devices = gpu_devices()
    if 'device' in config:
        selected = int(config['device'])
        if selected not in devices:
            parser.error(f'GPU {selected} is unavailable')
        devices = [selected]
    print(f'Processing {minutes} x 60-second integrations on GPUs {devices}', flush=True)
    jobs = iter(range(minutes))
    lock = threading.Lock()
    stopped = threading.Event()
    completed = 0

    def worker(device):
        nonlocal completed
        while not stopped.is_set():
            with lock:
                minute = next(jobs, None)
            if minute is None:
                return
            try:
                process_minute(raw, output, start + minute*60, device)
            except Exception:
                stopped.set()
                raise
            with lock:
                completed += 1
                print(f'PROGRESS {completed}/{minutes} minutes complete', flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for future in as_completed([pool.submit(worker, device) for device in devices]):
            future.result()
    print(f'FINISHED {completed}/{minutes} minutes', flush=True)


if __name__ == '__main__':
    main()
