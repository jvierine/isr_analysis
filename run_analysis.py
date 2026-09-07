#!/usr/bin/env python3
"""
Driver for Millstone Hill ISR analysis pipeline.

Usage:
    python run_analysis.py <config.json>
    mpirun -np 8 python run_analysis.py <config.json>

Steps in the pipeline:
    lpi         - Lag-profile inversion (coded long pulse, ACF estimation)
    fit_lpi     - Fit ACFs to plasma parameters
    long_pulse  - Range-Doppler spectra (uncoded long pulse)
    fit_lp      - Fit Doppler spectra to plasma parameters
"""

import sys
import os
import json
import numpy as n

# Ensure local modules (stuffr, il_interp, millstone_radar_state, …) are found
# when MPI workers are spawned in a different working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fall back to Agg only when no display is available (headless / MPI on server).
# If DISPLAY or WAYLAND_DISPLAY is set, use whatever backend matplotlib picks normally.
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"


def load_config(path):
    with open(path) as f:
        return json.load(f)


def run_lpi(datadir, output_base, cfg, max_time_s=None):
    import outlier_lpi as olpi
    olpi.lpi_files(
        dirname=datadir,
        channel=cfg["channel"],
        rg=cfg["range_gate_us"],
        avg_dur=cfg.get("avg_dur", 10),
        min_tx_frac=cfg.get("min_tx_frac", 0.5),
        pass_band=cfg["pass_band_hz"],
        filter_len=cfg.get("filter_len", 20),
        maximum_range_delay=cfg.get("max_range_delay_us", 7000),
        save_acf_images=cfg.get("save_acf_images", True),
        lag_avg=cfg.get("lag_avg", 1),
        reanalyze=cfg.get("reanalyze", False),
        output_base=output_base,
        max_time_s=max_time_s,
    )


def run_fit_lpi(datadir, output_base, cfg, lpi_cfg, radar_freq_hz=440.2e6, table_dir=None):
    import fit_lpi as flpi
    postfix = "_%d" % lpi_cfg["range_gate_us"]
    flpi.fit_lpifiles(
        dirn=datadir,
        channel=cfg["channel"],
        postfix=postfix,
        max_dt=cfg.get("max_dt", 300),
        plot=False,
        first_lag=cfg.get("first_lag", 0),
        reanalyze=cfg.get("reanalyze", False),
        range_avg=n.array(cfg.get("range_avg", [1, 3, 5])),
        output_base=output_base,
        radar_freq_hz=radar_freq_hz,
        table_dir=table_dir,
    )


def run_long_pulse(datadir, output_base, cfg, max_time_s=None):
    import avg_range_doppler_spec as ards
    ards.avg_range_doppler_spectra(
        dirname=datadir,
        channel=cfg["channel"],
        mode=cfg.get("mode", 300),
        avg_dur=cfg.get("avg_dur", 10),
        step=cfg.get("step", 10),
        avg_type=cfg.get("avg_type", "outlier_removal"),
        postfix=cfg.get("postfix", "_outlier"),
        min_tx_pulses=cfg.get("min_tx_pulses", 100),
        reanalyze=cfg.get("reanalyze", False),
        output_base=output_base,
        max_time_s=max_time_s,
    )


def run_fit_lp(datadir, output_base, cfg, lp_cfg):
    import fit_lp as flp
    postfix = "_%d%s" % (lp_cfg.get("mode", 300), lp_cfg.get("postfix", "_outlier"))
    flp.fit_spectra(
        dirname=datadir,
        channel=cfg["channel"],
        postfix=postfix,
        avg_dur=cfg.get("avg_dur", 600),
        ridx=cfg.get("ridx", [35, 230]),
        remove_space_objects=cfg.get("remove_space_objects", False),
        reanalyze=cfg.get("reanalyze", False),
        output_base=output_base,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <config.json>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    datadir = config["data_dir"]
    output_base = config.get("output_dir", None)
    max_time_s = config.get("max_time_s", None)
    radar_freq_hz = config.get("radar_freq_hz", 440.2e6)
    table_dir = config.get("table_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    steps = config.get("steps", {})

    lpi_cfg = steps.get("lpi", {})
    lp_cfg = steps.get("long_pulse", {})

    if lpi_cfg.get("enabled", False):
        print("=== Step: LPI (lag-profile inversion) ===")
        run_lpi(datadir, output_base, lpi_cfg, max_time_s=max_time_s)

    if steps.get("fit_lpi", {}).get("enabled", False):
        if not lpi_cfg:
            print("ERROR: fit_lpi requires lpi config for postfix")
            sys.exit(1)
        print("=== Step: fit_lpi (ACF fitting) ===")
        run_fit_lpi(datadir, output_base, steps["fit_lpi"], lpi_cfg,
                    radar_freq_hz=radar_freq_hz, table_dir=table_dir)

    if lp_cfg.get("enabled", False):
        print("=== Step: long_pulse (range-Doppler spectra) ===")
        run_long_pulse(datadir, output_base, lp_cfg, max_time_s=max_time_s)

    if steps.get("fit_lp", {}).get("enabled", False):
        if not lp_cfg:
            print("ERROR: fit_lp requires long_pulse config for postfix")
            sys.exit(1)
        print("=== Step: fit_lp (Doppler spectrum fitting) ===")
        run_fit_lp(datadir, output_base, steps["fit_lp"], lp_cfg)


if __name__ == "__main__":
    main()
