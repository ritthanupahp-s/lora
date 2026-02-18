# rssi_autocorr.py
# Autocorrelation + coherence-time estimation for RSSI series

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def acf_fft_unbiased(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalized autocorrelation using FFT, with an unbiased divisor.
    Returns ACF[0..max_lag], where ACF[0] = 1.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 samples for autocorrelation.")

    max_lag = min(max_lag, n - 1)

    # FFT-based autocorrelation (via power spectrum)
    f = np.fft.rfft(x, n=2 * n)          # zero-pad to reduce circular effects
    p = f * np.conjugate(f)
    ac = np.fft.irfft(p)[:n]             # raw autocorrelation

    # Unbiased normalization by (n-k)
    ac = ac / np.arange(n, 0, -1)

    # Normalize so acf[0] = 1
    ac = ac / ac[0]

    return ac[: max_lag + 1]


def first_crossing(acf: np.ndarray, threshold: float) -> int | None:
    """Smallest lag k>0 where ACF[k] <= threshold. Returns None if never crosses."""
    for k in range(1, len(acf)):
        if acf[k] <= threshold:
            return k
    return None


def estimate_dt_ms_from_t(t_ms: np.ndarray) -> float:
    """Robust sampling interval estimate (median of diffs)."""
    diffs = np.diff(t_ms.astype(float))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Could not estimate dt from t_ms (no positive diffs).")
    return float(np.median(diffs))


def load_series_from_csv(path: str, rssi_col: str, time_col: str | None):
    df = pd.read_csv(path)
    if rssi_col not in df.columns:
        raise ValueError(f"RSSI column '{rssi_col}' not found. Columns: {list(df.columns)}")

    x = df[rssi_col].to_numpy(dtype=float)

    dt_ms = None
    if time_col:
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found. Columns: {list(df.columns)}")
        dt_ms = estimate_dt_ms_from_t(df[time_col].to_numpy())

    return x, dt_ms, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV containing RSSI samples.")
    ap.add_argument("--rssi-col", type=str, default="rssi_dbm", help="RSSI column name.")
    ap.add_argument("--time-col", type=str, default="t_ms", help="Time column name (set '' to disable).")
    ap.add_argument("--max-lag-sec", type=float, default=15.0, help="Max lag to plot/compute in seconds.")
    ap.add_argument("--thr", type=float, nargs="*", default=[0.5, 1 / math.e, 0.0],
                    help="Thresholds for coherence time (ACF <= thr).")
    ap.add_argument("--show-ci", action="store_true",
                    help="Overlay approx 95-percent white-noise CI: ±1.96/sqrt(N).")
    args = ap.parse_args()

    time_col = args.time_col if args.time_col.strip() else None
    x, dt_ms, df = load_series_from_csv(args.csv, args.rssi_col, time_col)

    # If dt unknown, assume 1 sample per unit lag (you can still interpret in samples)
    if dt_ms is None:
        dt_ms = 1.0

    max_lag = int(min(len(x) - 1, round((args.max_lag_sec * 1000.0) / dt_ms)))
    ac = acf_fft_unbiased(x, max_lag=max_lag)

    lags = np.arange(len(ac))
    lags_ms = lags * dt_ms

    # ---- Report coherence lags ----
    print(f"\nFile: {args.csv}")
    print(f"N={len(x)} samples | dt≈{dt_ms:.3f} ms | max_lag={max_lag} samples (~{lags_ms[-1]/1000:.2f} s)")
    for thr in args.thr:
        k = first_crossing(ac, thr)
        if k is None:
            print(f"  ACF never drops to <= {thr:g} within max lag.")
        else:
            print(f"  coherence_lag(thr={thr:g}) = {k} samples  (~{k*dt_ms:.1f} ms)")

    # ---- Plot ----
    plt.figure()
    plt.plot(lags_ms, ac)

    for thr in args.thr:
        plt.axhline(thr, linestyle="--")

    if args.show_ci:
        ci = 1.96 / math.sqrt(len(x))
        plt.axhline(+ci, linestyle=":")
        plt.axhline(-ci, linestyle=":")

    plt.xlabel("Lag (ms)")
    plt.ylabel("Autocorrelation (normalized)")
    plt.title(f"RSSI Autocorrelation (ACF) — {args.csv}")
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
