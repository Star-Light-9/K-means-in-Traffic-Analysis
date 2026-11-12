"""
track.py - follow clusters from step to step and decide which are stuck.

For each mode (seq, par) it reads clusters_<mode>.csv written by dbscan.c
and the original trajectories.csv, then:

  * matches every cluster to a cluster of the previous step by shared
    vehicles (Jaccard overlap), so a cluster keeps the same id ("track")
    and the same colour as it moves;
  * computes the cluster's speed = mean speed of its member vehicles;
  * marks a cluster as STUCK when that mean speed stays below
    --stuck-speed for at least --stuck-steps consecutive steps.

Outputs (in --data-dir):
    tracked_<mode>.csv        step,id,cluster,track,stuck      (per vehicle)
    cluster_stats_<mode>.csv  step,track,size,mean_speed,stuck (per cluster)

Usage:
    python track.py --data-dir data
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd


def track_clusters(df, stuck_speed, stuck_steps, min_jaccard=0.2):
    n = len(df)
    ids = df["id"].to_numpy()
    cl = df["cluster"].to_numpy()
    spd = df["speed"].to_numpy()

    out_track = np.full(n, -1, dtype=np.int64)
    out_stuck = np.zeros(n, dtype=np.uint8)

    prev_owner = {}     # vehicle id -> track id in previous step
    prev_size = {}      # track id -> size in previous step
    low_count = {}      # track id -> consecutive slow steps
    next_track = 0
    stats = []

    groups = df.groupby("step", sort=True).indices

    for step in sorted(groups):
        idx = groups[step]
        valid = cl[idx] >= 0

        new_owner = {}
        new_size = {}

        if valid.any():
            rows = idx[valid]
            uniq, inv = np.unique(cl[rows], return_inverse=True)

            order = np.argsort(inv, kind="stable")
            sorted_rows = rows[order]
            bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))

            members = []
            cand = []

            for c in range(len(uniq)):
                r = sorted_rows[bounds[c]:bounds[c + 1]]
                members.append(r)

                counts = Counter(prev_owner.get(int(i)) for i in ids[r])
                counts.pop(None, None)

                for t, k in counts.items():
                    jac = k / (len(r) + prev_size[t] - k)

                    if jac >= min_jaccard:
                        cand.append((jac, c, t))

            cand.sort(reverse=True)
            assigned = {}
            used = set()

            for jac, c, t in cand:
                if c not in assigned and t not in used:
                    assigned[c] = t
                    used.add(t)

            for c, r in enumerate(members):
                t = assigned.get(c)

                if t is None:
                    t = next_track
                    next_track += 1

                mean_speed = float(spd[r].mean())

                if mean_speed < stuck_speed:
                    low_count[t] = low_count.get(t, 0) + 1
                else:
                    low_count[t] = 0

                stuck = low_count[t] >= stuck_steps

                out_track[r] = t
                out_stuck[r] = 1 if stuck else 0
                new_size[t] = len(r)

                for i in ids[r]:
                    new_owner[int(i)] = t

                stats.append((step, t, len(r), mean_speed, int(stuck)))

        prev_owner = new_owner
        prev_size = new_size

    tracked = pd.DataFrame({
        "step": df["step"].to_numpy(),
        "id": ids,
        "cluster": cl,
        "track": out_track,
        "stuck": out_stuck,
    })

    stats_df = pd.DataFrame(
        stats, columns=["step", "track", "size", "mean_speed", "stuck"])

    return tracked, stats_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--stuck-speed", type=float, default=2.0,
                    help="mean cluster speed (m/s) below which it is slow")
    ap.add_argument("--stuck-steps", type=int, default=3,
                    help="consecutive slow steps before a cluster is stuck")
    args = ap.parse_args()

    traj = pd.read_csv(os.path.join(args.data_dir, "trajectories.csv"),
                       usecols=["step", "id", "speed"])

    for mode in ("seq", "par"):
        path = os.path.join(args.data_dir, "clusters_%s.csv" % mode)

        if not os.path.isfile(path):
            print("Missing", path, "- run dbscan first")
            continue

        cl = pd.read_csv(path)
        df = cl.merge(traj, on=["step", "id"], how="left")
        df["speed"] = df["speed"].fillna(0.0)

        tracked, stats = track_clusters(
            df, args.stuck_speed, args.stuck_steps)

        tracked.to_csv(
            os.path.join(args.data_dir, "tracked_%s.csv" % mode), index=False)
        stats.to_csv(
            os.path.join(args.data_dir, "cluster_stats_%s.csv" % mode),
            index=False)

        n_tracks = stats["track"].nunique()
        n_stuck = stats.loc[stats["stuck"] == 1, "track"].nunique()

        print("%s: %d tracks, %d of them were stuck at some point"
              % (mode, n_tracks, n_stuck))


if __name__ == "__main__":
    main()
