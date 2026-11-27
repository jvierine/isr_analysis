"""Independent LP480 timing check using catalogued crossings and raw first-lag echoes."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar
from sgp4.api import Satrec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lhw_ac"))
import satellite_crossings as orbit

C = 299792.458
STEP_US = 10.
PULSE_US = 480.
FRAME = 11616
DELAYS = np.arange(-1000., 1000.1, 2.)
CORE = np.abs(DELAYS) <= 450
OFF = np.abs(DELAYS) >= 700
PARTS = {}
COARSE = None
CLOCKS = None


def init_geometry(beams, start):
    lat, lon, altitude = 18.3492, 109.6222, 0.
    p, l = np.deg2rad([lat, lon])
    n = 6378.137 / np.sqrt(1 - 6.6943799901413165e-3 * np.sin(p)**2)
    orbit.SITE = np.array([(n+altitude)*np.cos(p)*np.cos(l), (n+altitude)*np.cos(p)*np.sin(l),
                           (n*(1-6.6943799901413165e-3)+altitude)*np.sin(p)])
    orbit.ENU = np.array([[-np.sin(l), np.cos(l), 0],
                          [-np.sin(p)*np.cos(l), -np.sin(p)*np.sin(l), np.cos(p)],
                          [np.cos(p)*np.cos(l), np.cos(p)*np.sin(l), np.sin(p)]])
    a, e = np.deg2rad(np.array(beams).T)
    orbit.BEAMS = np.array([np.sin(a)*np.cos(e), np.cos(a)*np.cos(e), np.sin(e)]).T
    global COARSE, CLOCKS
    COARSE = np.arange(start - 10, start + 86411, 10.)
    CLOCKS = orbit.time_arrays(COARSE)
    return lat, lon, altitude


def crossings(record):
    sat = Satrec.twoline2rv(record["TLE_LINE1"], record["TLE_LINE2"])
    sep, r, _, _, _ = orbit.observe(sat, COARSE, CLOCKS)
    rows = []
    for b in PARTS:
        a = sep[:, b]
        ix = np.flatnonzero((a[1:-1] < a[:-2]) & (a[1:-1] <= a[2:]) &
                            (a[1:-1] < 12) & (r[1:-1] < 2400)) + 1
        for i in ix:
            t0 = COARSE[i]
            if not any(block in PARTS[b] for block in range(int(t0)//10*10-20, int(t0)//10*10+21, 10)):
                continue
            best = minimize_scalar(lambda dt: orbit.observe(sat, [t0+dt])[0][0,b],
                                   bounds=(-10, 10), method="bounded")
            t = t0 + best.x
            if best.fun < 2.5 and COARSE[0]+10 <= t < COARSE[-1]-10:
                rows.append((float(best.fun), t, b, record))
    return rows


def read_part(part):
    with open(part["file"], "rb") as f:
        f.seek(part["offset"])
        raw = f.read(part["pulses"]*FRAME)
    if len(raw) != part["pulses"]*FRAME:
        raise ValueError("Truncated indexed raw segment")
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, FRAME)[::3]
    if np.any(a[:, :4] != [85,85,170,170]) or np.any(a[:,1020:1024] != [170,85,85,170]):
        raise ValueError("Indexed header sentinel mismatch")
    h = a[:, :1024]
    fields = h[:, 12:18]
    stamps = np.array([datetime(2000+int(y), int(m), int(d), int(hh), int(mm), int(ss),
                                 tzinfo=timezone.utc).timestamp()
                       for m,y,hh,d,ss,mm in fields])
    stamp = stamps + (h[:,18].astype(int)+256*h[:,19].astype(int))*25e-6 - 28800
    front = h[:,766].astype(int)+256*h[:,767].astype(int)
    iq = np.ascontiguousarray(a[:,1024:]).view("<i4").reshape(len(a), -1, 2)
    x = iq[:,:,1].astype(float) + 1j*iq[:,:,0]
    return stamp, front, x


def peak(profile):
    # A rectangular pulse echo has 470 us of nonzero first-lag overlap.
    smoothed = uniform_filter1d(profile.real, size=235, mode="nearest") + 1j*uniform_filter1d(profile.imag, size=235, mode="nearest")
    score = np.abs(smoothed)
    i = np.where(CORE)[0][np.argmax(score[CORE])]
    off = profile[OFF]
    level = np.median(np.abs(off - np.median(off))) * 1.4826
    second = score[CORE & (np.abs(DELAYS-DELAYS[i]) > 180)].max()
    return DELAYS[i], score[i]/max(level, 1e-12), smoothed[i], float(score[i]/max(second,1e-12))


def measure(candidate):
    _, t0, b, rec = candidate
    sat = Satrec.twoline2rv(rec["TLE_LINE1"], rec["TLE_LINE2"])
    parts = [p for k in range(int(t0-15)//10*10, int(t0+15)//10*10+1, 10) for p in PARTS[b].get(k, [])]
    aligned, results = [], []
    for part in parts:
        t, front, x = read_part(part)
        sep,r,rr,_,_ = orbit.observe(sat,t)
        select = np.flatnonzero((abs(t-t0)<15) & (sep[:,b]<2.0) &
                                (r>300) & (r<1850) & (np.abs(rr)>.7))
        for i in select:
            lag = x[i,:-1]*x[i,1:].conj()
            delay = front[i] + (np.arange(lag.size)+.5)*STEP_US - PULSE_US/2 - r[i]*2e6/C
            if delay[0]>DELAYS[0] or delay[-1]<DELAYS[-1]:
                continue
            z = np.interp(DELAYS,delay,lag.real) + 1j*np.interp(DELAYS,delay,lag.imag)
            z -= np.median(z[OFF].real)+1j*np.median(z[OFF].imag)
            aligned.append(z)
            results.append((t[i],r[i],rr[i],sep[i,b]))
    if len(results)<30:
        return None
    results, aligned = np.asarray(results), np.asarray(aligned)
    order = results[:,0].argsort()
    results, aligned = results[order], aligned[order]
    if np.ptp(results[:,0])<1.0 or np.ptp(results[:,1])<1.0:
        return None
    # Fit independent time halves, so a single strong pulse cannot pass.
    halves = [peak(g.mean(axis=0)) for g in np.array_split(aligned, 2)]
    delay,snr,amp,contrast = peak(aligned.mean(axis=0))
    vr = np.median(results[:,2])
    phase = np.angle(amp)
    predicted_phase = 2*np.pi*2*430.2e6*vr/C*10e-6
    phase_distance = [abs(np.angle(np.exp(1j*(phase-s*predicted_phase)))) for s in (-1,1)]
    accepted = (min(x[1] for x in halves)>=5 and abs(halves[0][0]-halves[1][0])<=28
                and abs(delay)<400 and contrast>=1.25 and min(phase_distance)<.45)
    row = dict(norad=int(rec["NORAD_CAT_ID"]), name=rec["OBJECT_NAME"], epoch=rec["EPOCH"],
               beam=b, predicted_utc=t0, median_utc=float(np.median(results[:,0])),
               range_km=float(np.median(results[:,1])), velocity_km_s=float(vr),
               measured_minus_predicted_us=float(delay), snr=float(snr), contrast=contrast,
               phase_rad=float(phase), phase_error_rad=phase_distance, pulses=len(results),
               split_delay_us=[float(k[0]) for k in halves], split_snr=[float(k[1]) for k in halves],
               minimum_recorded_separation_deg=float(np.min(results[:,3])), accepted=bool(accepted))
    return row, aligned.mean(axis=0), results


def main():
    a = argparse.ArgumentParser(__doc__)
    a.add_argument("index", type=Path)
    a.add_argument("tles", type=Path)
    a.add_argument("output", type=Path)
    a.add_argument("--workers", type=int, default=12)
    q = a.parse_args()
    with h5py.File(q.index) as f:
        index = json.loads(bytes(f["json_utf8"][:]))
    assert index["header_utc_offset_s"] == 28800
    start = datetime(2024,1,1,tzinfo=timezone.utc).timestamp()
    beams = sorted({(j["az"],j["el"]) for j in index["jobs"]})
    site = init_geometry(beams,start)
    for b in range(len(beams)):
        PARTS[b] = defaultdict(list)
    for j in index["jobs"]:
        b = beams.index((j["az"],j["el"]))
        for part in j["parts"]:
            PARTS[b][part["block"]].append(part)
    raw = q.tles.read_bytes()
    records = json.loads(gzip.decompress(raw))
    with ProcessPoolExecutor(max_workers=q.workers) as pool:
        candidates = [x for rows in pool.map(crossings,records,chunksize=32) for x in rows]
        candidates.sort(key=lambda row: row[0])
        print("PREDICTED",len(candidates),flush=True)
        measurements = [x for x in pool.map(measure,candidates,chunksize=3) if x is not None]
    accepted = [x[0] for x in measurements if x[0]["accepted"]]
    with h5py.File(q.output,"w") as f:
        f.attrs.update(snapshot_sha256=hashlib.sha256(raw).hexdigest(), site_lat_lon_alt_km=site,
                       convention="Observed minus predicted two-way time; subtract a positive offset from the current range origin",
                       method="Raw x[k] conj(x[k+1]); pair midpoint less half pulse; track-centered rectangular-envelope correlation")
        f["beam_az_el_deg"] = beams
        f["delay_axis_us"] = DELAYS
        for i,(row,profile,tracks) in enumerate(measurements):
            g=f.create_group(f"crossing/{i:05}")
            g.attrs["metrics_json"]=json.dumps(row)
            g["profile"]=profile
            g["recorded_utc_range_rate_separation"]=tracks
    print("ACCEPTED",len(accepted),"unique objects",len({x["norad"] for x in accepted}),flush=True)
    for row in accepted:
        print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
