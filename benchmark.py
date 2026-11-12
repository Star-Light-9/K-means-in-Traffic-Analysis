"""
benchmark.py - run dbscan with 1, 2, 4, 8 ... MPI processes and report
speedup, efficiency and load imbalance. Optionally compares the two ways
of cutting the map (count-balanced vs equal-width strips).

Usage:
    python benchmark.py --input data/trajectories.csv
    python benchmark.py --input data_big/trajectories.csv --procs 1 2 4 8 \
                        --reps 3 --compare-split

Windows (MS-MPI):  add  --mpirun mpiexec --exe dbscan.exe
OpenMPI with more processes than cores: --mpi-extra --oversubscribe

Writes benchmark_results.csv and benchmark.png next to --out-dir.
"""

import argparse
import os
import shlex
import subprocess
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/trajectories.csv")
    ap.add_argument("--exe", default="./dbscan")
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--mpi-extra", default="",
                    help='extra mpirun flags, e.g. "--oversubscribe"')
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out-dir", default="bench")
    ap.add_argument("--eps-pos", default="25")
    ap.add_argument("--eps-vel", default="4")
    ap.add_argument("--minpts", default="4")
    ap.add_argument("--reps", default="3",
                    help="repeat each step this many times and average")
    ap.add_argument("--max-frames", default="0")
    ap.add_argument("--compare-split", action="store_true",
                    help="also run equal-width strips (BALANCE=0)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = [1, 0] if args.compare_split else [1]
    rows = []

    for bal in splits:
        for p in args.procs:
            out = os.path.join(args.out_dir, "p%d_bal%d" % (p, bal))
            os.makedirs(out, exist_ok=True)

            cmd = ([args.mpirun, "-n", str(p)]
                   + shlex.split(args.mpi_extra)
                   + [args.exe, args.input, out, args.eps_pos, args.eps_vel,
                      args.minpts, str(bal), args.reps, args.max_frames])

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
                "split": "count-balanced" if bal else "equal-width",
                "vehicles_per_step": t["n"].mean(),
                "seq_total_ms": t["seq_ms"].sum(),
                "par_total_ms": t["par_ms"].sum(),
                "load_imbalance": ratio.mean(),
            })

    df = pd.DataFrame(rows)

    # The sequential time is the same work in every run; use the fastest
    # measurement as the reference so noise does not distort the speedup.
    ref = df["seq_total_ms"].min()
    df["seq_ref_ms"] = ref
    df["speedup"] = ref / df["par_total_ms"]
    df["efficiency_pct"] = 100.0 * df["speedup"] / df["procs"]

    df.to_csv(os.path.join(args.out_dir, "benchmark_results.csv"), index=False)

    pd.set_option("display.width", 140)
    print()
    print(df.round(3).to_string(index=False))

    # ---------------- chart ----------------
    colors = {"count-balanced": "#1f77b4", "equal-width": "#d95f02"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    for name, g in df.groupby("split"):
        c = colors[name]
        ax[0].plot(g["procs"], g["par_total_ms"], "o-", color=c, label=name)
        ax[1].plot(g["procs"], g["speedup"], "o-", color=c, label=name)
        ax[2].plot(g["procs"], g["load_imbalance"], "o-", color=c, label=name)

    ax[0].axhline(ref, color="#555555", ls="--", label="sequential")
    ax[0].set_ylabel("total time (ms)")
    ax[0].set_title("Total clustering time")

    pmax = max(args.procs)
    ax[1].plot([1, pmax], [1, pmax], color="#999999", ls=":", label="ideal")
    ax[1].axhline(1.0, color="#555555", ls="--", lw=0.8)
    ax[1].set_ylabel("speedup vs sequential")
    ax[1].set_title("Speedup")

    ax[2].axhline(1.0, color="#999999", ls=":")
    ax[2].set_ylabel("slowest rank / average rank")
    ax[2].set_title("Load imbalance (1.0 = perfect)")

    for a in ax:
        a.set_xlabel("MPI processes")
        a.set_xticks(args.procs)
        a.grid(alpha=0.25)
        a.legend(frameon=False)

    fig.suptitle("Parallel DBSCAN - %.0f vehicles per step"
                 % df["vehicles_per_step"].iloc[0])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "benchmark.png"), dpi=150)
    print("\nWrote", os.path.join(args.out_dir, "benchmark_results.csv"),
          "and benchmark.png")


if __name__ == "__main__":
    main()
