"""
benchmark.py - run dbscan with several MPI process counts and report
speedup, efficiency and load imbalance. Optionally compares the two ways
of cutting the map (balanced vs equal-width strips).

Settings come from config.ini: [benchmark], [mpi], [dbscan], [paths].
It uses the trajectories already in data_dir (run run_sumo.py first).

Usage:
    python benchmark.py

Writes (in <data_dir>/bench):  benchmark_results.csv  and  benchmark.png
"""

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def find_launcher(name):
    if name != "auto":
        return name

    for cand in ("mpirun", "mpiexec"):
        if shutil.which(cand):
            return cand

    sys.exit("No mpirun / mpiexec found. Install MPI first (see README).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="settings file (default: config.ini next to the scripts)")
    cli = ap.parse_args()

    cfg = config.load(cli.config)

    exe_name = "dbscan.exe" if platform.system() == "Windows" else "dbscan"
    exe = os.path.join(config.HERE, exe_name)

    if not os.path.isfile(exe):
        sys.exit("%s not found. Build it first (see README), or run "
                 "python run_dbscan.py once, which builds it." % exe_name)

    launcher = find_launcher(cfg.str("mpi", "launcher"))
    extra = shlex.split(cfg.str("mpi", "extra_args"))
    procs = cfg.ints("benchmark", "processes")
    reps = cfg.int("benchmark", "reps")

    out_root = os.path.join(cfg.data_dir, "bench")
    os.makedirs(out_root, exist_ok=True)

    splits = [("balanced", 1), ("equal", 0)] \
        if cfg.bool("benchmark", "compare_split") else [("balanced", 1)]

    rows = []

    for split_name, _ in splits:
        for p in procs:
            out = os.path.join(out_root, "p%d_%s" % (p, split_name))
            os.makedirs(out, exist_ok=True)

            cmd = ([launcher, "-n", str(p)] + extra
                   + [exe, cfg.path,
                      "split=" + split_name,
                      "reps=%d" % reps,
                      "out_dir=" + out])

            print("running:", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)

            if res.returncode != 0:
                print(res.stdout)
                print(res.stderr)
                sys.exit("dbscan failed")

            t = pd.read_csv(os.path.join(out, "timings.csv"))

            ratio = (t["par_compute_max_ms"]
                     / t["par_compute_mean_ms"].replace(0, float("nan")))

            rows.append({
                "procs": p,
                "split": split_name,
                "vehicles_per_step": t["n"].mean(),
                "seq_total_ms": t["seq_ms"].sum(),
                "par_total_ms": t["par_ms"].sum(),
                "load_imbalance": ratio.mean(),
            })

    df = pd.DataFrame(rows)

    # The sequential work is the same in every run; use the fastest
    # measurement as the reference so noise does not distort the speedup.
    ref = df["seq_total_ms"].min()
    df["seq_ref_ms"] = ref
    df["speedup"] = ref / df["par_total_ms"]
    df["efficiency_pct"] = 100.0 * df["speedup"] / df["procs"]

    df.to_csv(os.path.join(out_root, "benchmark_results.csv"), index=False)

    pd.set_option("display.width", 140)
    print()
    print(df.round(3).to_string(index=False))

    # ---------------- chart ----------------
    colors = {"balanced": "#1f77b4", "equal": "#d95f02"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    for name, g in df.groupby("split"):
        c = colors[name]
        ax[0].plot(g["procs"], g["par_total_ms"], "o-", color=c, label=name)
        ax[1].plot(g["procs"], g["speedup"], "o-", color=c, label=name)
        ax[2].plot(g["procs"], g["load_imbalance"], "o-", color=c, label=name)

    ax[0].axhline(ref, color="#555555", ls="--", label="sequential")
    ax[0].set_ylabel("total time (ms)")
    ax[0].set_title("Total clustering time")

    pmax = max(procs)
    ax[1].plot([1, pmax], [1, pmax], color="#999999", ls=":", label="ideal")
    ax[1].axhline(1.0, color="#555555", ls="--", lw=0.8)
    ax[1].set_ylabel("speedup vs sequential")
    ax[1].set_title("Speedup")

    ax[2].axhline(1.0, color="#999999", ls=":")
    ax[2].set_ylabel("slowest rank / average rank")
    ax[2].set_title("Load imbalance (1.0 = perfect)")

    for a in ax:
        a.set_xlabel("MPI processes")
        a.set_xticks(procs)
        a.grid(alpha=0.25)
        a.legend(frameon=False)

    fig.suptitle("Parallel DBSCAN - %.0f vehicles per step"
                 % df["vehicles_per_step"].iloc[0])
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "benchmark.png"), dpi=150)

    print("\nWrote", os.path.join(out_root, "benchmark_results.csv"),
          "and benchmark.png")


if __name__ == "__main__":
    main()