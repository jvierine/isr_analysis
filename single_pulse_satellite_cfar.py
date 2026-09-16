#!/usr/bin/env python3
"""MPI single-pulse range--Doppler CFAR satellite detector.

The final product is a Digital Metadata channel keyed by the absolute sample
of the transmitted pulse.  At 1 MHz these keys are also Unix microseconds.
Each record contains arrays, so one pulse can contain several non-overlapping
echoes without manufacturing extra timestamps.  Receiver and transmit-tap
voltages are coherently integrated and decimated before the matched-filter
bank, reducing the number of searched range gates.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Set this before h5py/digital_rf load HDF5.  MPI ranks only read the raw
# archive and each writes a different chunk file.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pyfftw
from digital_rf import DigitalMetadataReader, DigitalMetadataWriter, DigitalRFReader
from mpi4py import MPI
from scipy.fft import next_fast_len
from scipy.ndimage import maximum_filter1d, uniform_filter


FS = 1_000_000
C = 299_792_458.0
CHANNEL_NAMES = ("zenith-l", "misa-l")


@dataclass(frozen=True)
class Mode:
    tx0: int
    tx1: int
    clutter_end: int
    noise0: int
    noise1: int

    @property
    def pulse_length(self) -> int:
        return self.tx1 - self.tx0


AC_MODE = Mode(tx0=76, tx1=624, clutter_end=1000, noise0=8400, noise1=8850)
MODES = {
    **{sweep: AC_MODE for sweep in range(1, 33)},
    300: Mode(tx0=76, tx1=645, clutter_end=1000, noise0=7800, noise1=8371),
    800: Mode(tx0=69, tx1=2171, clutter_end=3721, noise0=30176, noise1=32033),
}

CHUNK_FIELDS = {
    "pulse_sample": np.uint64,
    "echo_index": np.uint16,
    "channel_code": np.uint8,
    "sweep_id": np.uint16,
    "raw_delay_sample": np.uint32,
    "corrected_delay_samples": np.float32,
    "range_km": np.float32,
    "doppler_hz": np.float32,
    "matched_filter_power": np.float32,
    "snr_db": np.float32,
    "cfar_ratio": np.float32,
    "cfar_threshold": np.float32,
    "noise_power": np.float32,
    "pulse_length": np.uint16,
    "integrated_pulse_length": np.uint16,
    "decimation_factor": np.uint8,
    "effective_sample_rate_hz": np.uint32,
    "nfft": np.uint16,
}


def parse_modes(text: str) -> set[int]:
    modes: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            modes.update(range(lo, hi + 1))
        else:
            modes.add(int(part))
    unknown = modes.difference(MODES)
    if unknown:
        raise ValueError(f"unsupported sweep IDs: {sorted(unknown)}")
    return modes


def scalar(value):
    """Return a scalar from Digital Metadata's scalar/array variants."""
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"expected one metadata field, got {value.keys()}")
        value = next(iter(value.values()))
    return np.asarray(value).reshape(-1)[0]


def read_antenna_state(data_dir: Path):
    path = data_dir / "metadata" / "antenna_control_metadata"
    reader = DigitalMetadataReader(str(path))
    lo, hi = reader.get_bounds()

    def read_field(name):
        records = reader.read(lo, hi, name)
        keys = np.asarray(sorted(records), dtype=np.uint64)
        vals = []
        for key in keys:
            value = scalar(records[int(key)])
            if isinstance(value, np.bytes_):
                value = bytes(value)
            if isinstance(value, bytes):
                value = value.decode("ascii", "replace")
            vals.append(str(value).upper())
        return keys, np.asarray(vals, dtype="S16")

    tx_keys, tx_values = read_field("tx_antenna")
    rx_keys, rx_values = read_field("rx_antenna")
    return tx_keys, tx_values, rx_keys, rx_values


def state_at(keys: np.ndarray, values: np.ndarray, sample: int) -> str | None:
    idx = int(np.searchsorted(keys, sample, side="right") - 1)
    if idx < 0:
        return None
    return bytes(values[idx]).decode("ascii", "replace").upper()


def selected_channel(antenna_state, sample: int) -> str | None:
    tx_keys, tx_values, rx_keys, rx_values = antenna_state
    tx = state_at(tx_keys, tx_values, sample)
    rx = state_at(rx_keys, rx_values, sample)
    if tx != rx:
        return None
    if tx and "MISA" in tx:
        return "misa-l"
    if tx and ("ZENITH" in tx or "ZEN" in tx):
        return "zenith-l"
    return None


class FFTBank:
    """Reusable single-threaded FFTW plans, one per (nfft, batch size)."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.plans = {}

    def plan(self, nfft: int):
        key = (nfft, self.batch_size)
        if key not in self.plans:
            shape = (self.batch_size, nfft)
            fft_in = pyfftw.empty_aligned(shape, dtype="complex64")
            fft_out = pyfftw.empty_aligned(shape, dtype="complex64")
            plan = pyfftw.FFTW(
                fft_in,
                fft_out,
                axes=(1,),
                direction="FFTW_FORWARD",
                flags=("FFTW_ESTIMATE",),
                threads=1,
            )
            self.plans[key] = fft_in, fft_out, plan
        return self.plans[key]


def fft_frequencies(nfft: int, sample_rate_hz: float = FS) -> np.ndarray:
    bins = np.arange(nfft, dtype=np.int32)
    signed = bins.copy()
    signed[bins > nfft // 2] -= nfft
    return signed.astype(np.float64) * (sample_rate_hz / nfft)


def integrate_and_decimate(values: np.ndarray, factor: int, offset: int = 0) -> np.ndarray:
    """Coherently sum non-overlapping blocks while retaining complex64."""
    if factor < 1:
        raise ValueError("decimation factor must be positive")
    if not 0 <= offset < factor:
        raise ValueError("decimation offset must be in [0, factor)")
    values = np.asarray(values, dtype=np.complex64)
    usable = ((len(values) - offset) // factor) * factor
    if usable <= 0:
        return np.empty(0, dtype=np.complex64)
    blocks = values[offset : offset + usable].reshape(-1, factor)
    return np.add.reduce(blocks, axis=1, dtype=np.complex64)


def ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def decimation_geometry(mode: Mode, factor: int) -> tuple[int, int, int, int]:
    """Return alignment, template length, and search bounds after decimation."""
    offset = mode.tx0 % factor
    template_length = mode.pulse_length // factor
    if template_length < 1:
        raise ValueError("decimation factor exceeds pulse length")
    search_start = ceil_div(mode.clutter_end - offset, factor)
    # A window beginning at the last included gate ends at or before noise0.
    search_stop = (mode.noise0 - offset) // factor - template_length + 1
    if search_stop <= search_start:
        raise ValueError("empty decimated range search")
    return offset, template_length, search_start, search_stop


def ambiguity_power(
    echo: np.ndarray,
    template: np.ndarray,
    start: int,
    stop: int,
    nfft: int,
    max_doppler_hz: float,
    noise_power: float,
    fft_bank: FFTBank,
    sample_rate_hz: float = FS,
):
    """Return normalized matched-filter power and retained Doppler bins."""
    length = len(template)
    count = stop - start
    if count <= 0:
        raise ValueError("empty range search")

    frequencies = fft_frequencies(nfft, sample_rate_hz)
    freq_idx = np.flatnonzero(np.abs(frequencies) <= max_doppler_hz)
    # FFTW returns natural FFT order (DC, positive, then negative).  CFAR and
    # sub-bin interpolation require physically adjacent, monotonic bins.
    freq_idx = freq_idx[np.argsort(frequencies[freq_idx])]
    kept_frequencies = frequencies[freq_idx].astype(np.float32)
    power = np.empty((count, len(freq_idx)), dtype=np.float32)

    template_conj = np.conjugate(template).astype(np.complex64, copy=False)
    energy = float(np.vdot(template, template).real)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("invalid transmit-template energy")
    scale = np.float32(1.0 / (energy * noise_power))

    windows = np.lib.stride_tricks.sliding_window_view(echo, length)
    fft_in, fft_out, plan = fft_bank.plan(nfft)
    batch_size = fft_bank.batch_size
    for out0 in range(0, count, batch_size):
        rows = min(batch_size, count - out0)
        fft_in.fill(0)
        np.multiply(
            windows[start + out0 : start + out0 + rows],
            template_conj,
            out=fft_in[:rows, :length],
        )
        plan()
        selected = fft_out[:rows, freq_idx]
        np.multiply(selected.real, selected.real, out=power[out0 : out0 + rows])
        power[out0 : out0 + rows] += selected.imag * selected.imag
        power[out0 : out0 + rows] *= scale
    return power, kept_frequencies


def cfar_peaks(
    power: np.ndarray,
    frequencies: np.ndarray,
    pulse_length: int,
    pfa: float,
    max_echoes: int,
    training_padding: int = 128,
):
    """Return separated 2-D CA-CFAR peaks from one ambiguity surface."""
    length = pulse_length
    # The range guard spans the full triangular ambiguity response.  With 4x
    # Doppler padding, four bins span the main-lobe null-to-peak distance.
    guard_r = length
    train_r = length + max(training_padding, length // 4)
    guard_f = 5
    train_f = 12
    if power.shape[0] <= 2 * train_r or power.shape[1] <= 2 * train_f:
        return [], np.float32(np.nan)

    outer_shape = (2 * train_r + 1, 2 * train_f + 1)
    inner_shape = (2 * guard_r + 1, 2 * guard_f + 1)
    outer_n = outer_shape[0] * outer_shape[1]
    inner_n = inner_shape[0] * inner_shape[1]
    ntrain = outer_n - inner_n
    alpha = np.float32(ntrain * math.expm1(-math.log(pfa) / ntrain))

    outer = uniform_filter(power, size=outer_shape, mode="nearest")
    inner = uniform_filter(power, size=inner_shape, mode="nearest")
    noise = outer
    noise *= np.float32(outer_n)
    noise -= inner * np.float32(inner_n)
    noise /= np.float32(ntrain)
    del inner

    ratio = power / np.maximum(noise, np.finfo(np.float32).tiny)
    ratio[:train_r, :] = -np.inf
    ratio[-train_r:, :] = -np.inf
    ratio[:, :train_f] = -np.inf
    ratio[:, -train_f:] = -np.inf

    best_fi = np.argmax(ratio, axis=1)
    rows = np.arange(ratio.shape[0])
    best_ratio = ratio[rows, best_fi]
    local_max = maximum_filter1d(best_ratio, size=2 * length + 1, mode="constant", cval=-np.inf)
    candidates = np.flatnonzero((best_ratio >= alpha) & (best_ratio == local_max))
    candidates = candidates[np.argsort(best_ratio[candidates])[::-1]]

    selected_rows: list[int] = []
    peaks = []
    df = float(frequencies[1] - frequencies[0])
    for ri in candidates:
        if any(abs(int(ri) - old) < length for old in selected_rows):
            continue
        fi = int(best_fi[ri])
        frac = 0.0
        if 0 < fi < power.shape[1] - 1:
            ym, y0, yp = np.log(np.maximum(power[ri, fi - 1 : fi + 2], 1e-30))
            denom = float(ym - 2 * y0 + yp)
            if denom != 0 and np.isfinite(denom):
                frac = float(np.clip(0.5 * (ym - yp) / denom, -0.5, 0.5))
        peaks.append(
            {
                "range_index": int(ri),
                "frequency_index": fi,
                "doppler_hz": float(frequencies[fi] + frac * df),
                "power": float(power[ri, fi]),
                "local_noise": float(noise[ri, fi]),
                "ratio": float(best_ratio[ri]),
                "alpha": float(alpha),
            }
        )
        selected_rows.append(int(ri))
        if len(peaks) >= max_echoes:
            break
    return peaks, alpha


def detect_pulse(
    rf: DigitalRFReader,
    sample: int,
    sweep_id: int,
    channel: str,
    args,
    fft_bank: FFTBank,
):
    mode = MODES[sweep_id]
    factor = args.decimation_factor
    effective_sample_rate = FS / factor
    offset, template_length, search_start, search_stop = decimation_geometry(mode, factor)
    read_length = mode.noise1
    echo = rf.read_vector(sample, read_length, channel).astype(np.complex64, copy=False)
    tx = rf.read_vector(sample, read_length, "tx-h").astype(np.complex64, copy=False)

    quiet_raw = echo[mode.noise0 - 500 : mode.noise0]
    echo_dc = np.complex64(np.median(quiet_raw.real) + 1j * np.median(quiet_raw.imag))
    echo = np.asarray(echo - echo_dc, dtype=np.complex64)
    echo = integrate_and_decimate(echo, factor, offset)
    quiet_start = ceil_div(mode.noise0 - 500 - offset, factor)
    quiet_stop = (mode.noise0 - offset) // factor
    quiet = echo[quiet_start:quiet_stop]
    quiet_power = float(np.median(np.abs(quiet) ** 2) / math.log(2.0))
    if not np.isfinite(quiet_power) or quiet_power <= 0:
        return []

    baseline = np.complex64(np.mean(tx[: mode.tx0]))
    template_stop = mode.tx0 + template_length * factor
    template = integrate_and_decimate(tx[mode.tx0:template_stop] - baseline, factor)
    nfft = next_fast_len(int(math.ceil(args.fft_padding * len(template))))
    power, frequencies = ambiguity_power(
        echo,
        template,
        search_start,
        search_stop,
        nfft,
        args.max_doppler_hz,
        quiet_power,
        fft_bank,
        effective_sample_rate,
    )
    peaks, _ = cfar_peaks(
        power,
        frequencies,
        len(template),
        args.pfa,
        args.max_echoes_per_pulse,
        training_padding=ceil_div(128, factor),
    )

    detections = []
    for echo_index, peak in enumerate(sorted(peaks, key=lambda p: p["range_index"])):
        decimated_delay = search_start + peak["range_index"]
        raw_delay = offset + factor * decimated_delay
        corrected = raw_delay - mode.tx0 - args.receiver_delay_samples
        detections.append(
            {
                "pulse_sample": sample,
                "echo_index": echo_index,
                "channel_code": CHANNEL_NAMES.index(channel),
                "sweep_id": sweep_id,
                "raw_delay_sample": raw_delay,
                "corrected_delay_samples": corrected,
                "range_km": C * corrected / FS / 2000.0,
                "doppler_hz": peak["doppler_hz"],
                "matched_filter_power": peak["power"],
                "snr_db": 10.0 * math.log10(max(peak["power"] - 1.0, 1e-12)),
                "cfar_ratio": peak["ratio"],
                "cfar_threshold": peak["alpha"],
                "noise_power": quiet_power,
                "pulse_length": mode.pulse_length,
                "integrated_pulse_length": len(template),
                "decimation_factor": factor,
                "effective_sample_rate_hz": int(effective_sample_rate),
                "nfft": nfft,
            }
        )
    return detections


def chunk_path(work_dir: Path, start: int, stop: int) -> Path:
    return work_dir / "chunks" / f"detections_{start}_{stop}.h5"


def chunk_complete(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            return bool(h5.attrs.get("complete", False))
    except (OSError, KeyError):
        return False


def write_chunk(path: Path, detections: list[dict], attrs: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".rank{MPI.COMM_WORLD.rank}.tmp")
    with h5py.File(temp, "w") as h5:
        for field, dtype in CHUNK_FIELDS.items():
            values = np.asarray([row[field] for row in detections], dtype=dtype)
            h5.create_dataset(field, data=values, compression="gzip", shuffle=True)
        for key, value in attrs.items():
            h5.attrs[key] = value
        h5.attrs["complete"] = True
        h5.flush()
    os.replace(temp, path)


def process_chunk(data_dir: Path, start: int, stop: int, modes, antenna_state, args, fft_bank):
    metadata = DigitalMetadataReader(str(data_dir / "metadata" / "id_metadata"))
    rf = DigitalRFReader(str(data_dir / "rf_data"))
    records = metadata.read(start, stop, "sweepid")
    detections = []
    pulses_seen = 0
    pulses_processed = 0
    failures = 0
    for sample in sorted(records):
        sweep_id = int(scalar(records[sample]))
        if sweep_id not in modes:
            continue
        pulses_seen += 1
        channel = selected_channel(antenna_state, int(sample))
        if channel is None:
            continue
        try:
            detections.extend(detect_pulse(rf, int(sample), sweep_id, channel, args, fft_bank))
            pulses_processed += 1
        except Exception as exc:
            failures += 1
            print(
                f"rank {MPI.COMM_WORLD.rank}: pulse {sample} sweep {sweep_id} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
    return detections, {
        "start_sample": np.uint64(start),
        "stop_sample": np.uint64(stop),
        "pulses_seen": np.uint64(pulses_seen),
        "pulses_processed": np.uint64(pulses_processed),
        "failures": np.uint64(failures),
        "detections": np.uint64(len(detections)),
        "decimation_factor": np.uint8(args.decimation_factor),
        "effective_sample_rate_hz": np.uint32(FS // args.decimation_factor),
    }


def detection_record(arrays, indices):
    first = int(indices[0])
    code = int(arrays["channel_code"][first])
    fields = {
        "echo_count": np.uint16(len(indices)),
        "channel": np.asarray(CHANNEL_NAMES[code].encode("ascii")),
    }
    for field, values in arrays.items():
        if field in ("pulse_sample", "channel_code"):
            continue
        fields[field] = values[indices]
    return fields


def finalize_metadata(output_dir: Path, work_dir: Path, chunks: list[tuple[int, int]], args):
    missing = [chunk_path(work_dir, a, b) for a, b in chunks if not chunk_complete(chunk_path(work_dir, a, b))]
    if missing:
        raise RuntimeError(f"cannot finalize: {len(missing)} chunks are missing or incomplete")
    if output_dir.exists():
        print(f"final Digital Metadata channel already exists: {output_dir}")
        return

    building = output_dir.with_name(output_dir.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    metadata_file_cadence_seconds = 3600
    writer = DigitalMetadataWriter(
        str(building), 3600, metadata_file_cadence_seconds, FS, 1, "satellite"
    )

    total_records = 0
    total_echoes = 0
    total_pulses = 0
    total_failures = 0
    duplicate_boundary_keys = 0
    last_written_sample: int | None = None
    batch_samples: list[int] = []
    batch_records: list[dict] = []
    batch_size = 100_000

    def flush_batch():
        if batch_samples:
            writer.write(np.asarray(batch_samples, dtype=np.uint64), batch_records)
            batch_samples.clear()
            batch_records.clear()

    for start, stop in chunks:
        path = chunk_path(work_dir, start, stop)
        with h5py.File(path, "r") as h5:
            total_pulses += int(h5.attrs["pulses_processed"])
            total_failures += int(h5.attrs["failures"])
            arrays = {field: h5[field][()] for field in CHUNK_FIELDS}
        samples = arrays["pulse_sample"]
        if not len(samples):
            continue
        unique, first = np.unique(samples, return_index=True)
        order = np.argsort(first)
        unique = unique[order]
        if last_written_sample is not None:
            if np.any(unique < last_written_sample):
                raise RuntimeError("chunk detections are not ordered by pulse sample")
            duplicate_boundary_keys += int(np.count_nonzero(unique == last_written_sample))
            unique = unique[unique > last_written_sample]
        if not len(unique):
            continue
        written_echoes = 0
        for sample in unique:
            indices = np.flatnonzero(samples == sample)
            batch_samples.append(int(sample))
            batch_records.append(detection_record(arrays, indices))
            written_echoes += len(indices)
        last_written_sample = int(unique[-1])
        total_records += len(unique)
        total_echoes += written_echoes
        if len(batch_samples) >= batch_size:
            flush_batch()
    flush_batch()
    del writer
    os.replace(building, output_dir)

    summary_path = output_dir.with_name(output_dir.name + "_summary.h5")
    with h5py.File(summary_path, "w") as h5:
        h5.attrs["complete"] = True
        h5.attrs["data_dir"] = str(args.data)
        h5.attrs["output_dir"] = str(output_dir)
        h5.attrs["sample_rate_hz"] = FS
        h5.attrs["decimation_factor"] = args.decimation_factor
        h5.attrs["effective_sample_rate_hz"] = FS / args.decimation_factor
        h5.attrs["range_gate_spacing_samples"] = args.decimation_factor
        h5.attrs["metadata_file_cadence_seconds"] = metadata_file_cadence_seconds
        h5.attrs["receiver_delay_samples"] = args.receiver_delay_samples
        h5.attrs["fft_padding"] = args.fft_padding
        h5.attrs["max_doppler_hz"] = args.max_doppler_hz
        h5.attrs["pfa_per_cell"] = args.pfa
        h5.attrs["modes"] = args.modes
        h5.attrs["pulses_processed"] = total_pulses
        h5.attrs["failed_pulses"] = total_failures
        h5.attrs["metadata_records"] = total_records
        h5.attrs["detected_echoes"] = total_echoes
        h5.attrs["duplicate_boundary_keys_skipped"] = duplicate_boundary_keys
    print(
        f"finalized {total_echoes} echoes in {total_records} pulse records "
        f"from {total_pulses} pulses; failures={total_failures}; "
        f"duplicate boundary keys skipped={duplicate_boundary_keys}",
        flush=True,
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="final Digital Metadata channel")
    p.add_argument("--work-dir", type=Path, help="restartable HDF5 chunks (default: OUTPUT.work)")
    p.add_argument("--modes", default="1-32,300,800")
    p.add_argument("--chunk-seconds", type=int, default=300)
    p.add_argument("--start-sample", type=int)
    p.add_argument("--stop-sample", type=int)
    p.add_argument("--receiver-delay-samples", type=float, default=11.0)
    p.add_argument("--decimation-factor", type=int, default=8)
    p.add_argument("--fft-padding", type=float, default=4.0)
    p.add_argument("--fft-batch", type=int, default=128)
    p.add_argument("--max-doppler-hz", type=float, default=60_000.0)
    p.add_argument("--pfa", type=float, default=1e-10)
    p.add_argument("--max-echoes-per-pulse", type=int, default=32)
    p.add_argument("--no-finalize", action="store_true")
    return p


def main():
    args = parser().parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    modes = parse_modes(args.modes)
    if args.fft_padding < 4:
        raise ValueError("--fft-padding must be at least 4")
    if args.decimation_factor < 1 or FS % args.decimation_factor:
        raise ValueError("--decimation-factor must be a positive divisor of the sample rate")
    if args.max_doppler_hz > FS / (2 * args.decimation_factor):
        raise ValueError("--max-doppler-hz exceeds the decimated Nyquist frequency")
    if not 0 < args.pfa < 1:
        raise ValueError("--pfa must be between zero and one")
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    work_dir = (args.work_dir or args.output.with_name(args.output.name + ".work")).resolve()

    if rank == 0:
        id_reader = DigitalMetadataReader(str(args.data / "metadata" / "id_metadata"))
        bound0, bound1 = (int(v) for v in id_reader.get_bounds())
        antenna_state = read_antenna_state(args.data)
    else:
        bound0 = bound1 = None
        antenna_state = None
    bound0, bound1, antenna_state = comm.bcast((bound0, bound1, antenna_state), root=0)
    start = max(bound0, args.start_sample if args.start_sample is not None else bound0)
    stop = min(bound1 + 1, args.stop_sample if args.stop_sample is not None else bound1 + 1)
    chunk_size = args.chunk_seconds * FS
    chunks = [(a, min(a + chunk_size, stop)) for a in range(start, stop, chunk_size)]

    fft_bank = FFTBank(args.fft_batch)
    began = time.monotonic()
    completed_here = 0
    for chunk_index in range(rank, len(chunks), comm.size):
        chunk_start, chunk_stop = chunks[chunk_index]
        path = chunk_path(work_dir, chunk_start, chunk_stop)
        if chunk_complete(path):
            continue
        detections, attrs = process_chunk(
            args.data, chunk_start, chunk_stop, modes, antenna_state, args, fft_bank
        )
        write_chunk(path, detections, attrs)
        completed_here += 1
        print(
            f"rank {rank}: chunk {chunk_index + 1}/{len(chunks)} "
            f"pulses={attrs['pulses_processed']} echoes={len(detections)} "
            f"elapsed={time.monotonic() - began:.1f}s",
            flush=True,
        )
    completed_total = comm.reduce(completed_here, op=MPI.SUM, root=0)
    comm.barrier()
    if rank == 0:
        print(f"MPI processing complete; wrote {completed_total} new chunks", flush=True)
        if not args.no_finalize:
            finalize_metadata(args.output, work_dir, chunks, args)


if __name__ == "__main__":
    main()
