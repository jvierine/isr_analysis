"""Process a recording into separate receiver range–Doppler HDF5 products."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON configuration file')
    parser.add_argument('--data', type=Path, help='Raw recording directory')
    parser.add_argument('--output', type=Path, help='Product directory')
    parser.add_argument('--start', type=float, help='First integration Unix time in UTC')
    parser.add_argument('--minutes', type=int, help='Number of 60-second integrations')
    parser.add_argument('--device', type=int, help='CUDA device index')
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
    minutes = int(config.get('minutes', 1))
    if minutes < 1:
        parser.error('minutes must be positive')
    output.mkdir(parents=True, exist_ok=True)
    from process import catalogue
    indexed = catalogue(raw, output/'catalogue.h5')
    if not indexed:
        parser.error('No recording files found')
    start = float(config.get('start', indexed[0][0]))
    device = str(config.get('device', 0))
    here = Path(__file__).resolve().parent
    for minute in range(minutes):
        epoch = start + minute*60
        for center in (436000000, 444000000):
            folder = output/'dual-channel-hour'/str(int(epoch))/str(center)
            folder.mkdir(parents=True, exist_ok=True)
            diagnostic = folder/'diagnostic.h5'
            matched = folder/'matched-step2us.h5'
            if not diagnostic.exists():
                subprocess.run([sys.executable, str(here/'process.py'), str(raw),
                                str(diagnostic), '--start', str(epoch), '--catalogue',
                                str(output/'catalogue.h5'), '--device', device,
                                '--receiver-center-hz', str(center)], check=True)
            if not matched.exists():
                subprocess.run([sys.executable, str(here/'matched.py'), str(raw),
                                str(diagnostic), str(matched), '--device', device,
                                '--range-step-us', '2'], check=True)
            print(f'Complete {epoch:g} UTC Unix seconds, receiver {center} Hz', flush=True)


if __name__ == '__main__':
    main()
