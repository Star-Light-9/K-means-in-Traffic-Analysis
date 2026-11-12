"""
make_synthetic.py - generate vehicle trajectories WITHOUT SUMO.

A simple car-following model on a grid of two-way roads, with a few
temporary bottlenecks that create real queues (jams) which then dissolve.
Use it to try the pipeline before SUMO is set up, or to make very large
scenarios (--vehicles 50000) for scaling tests.

Output (in --out-dir):
    trajectories.csv   step,id,x,y,vx,vy,speed
    network.json       road polylines for the background of the replay

Usage:
    python make_synthetic.py --out-dir data
    python make_synthetic.py --out-dir data_big --vehicles 40000 --steps 120
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--vehicles", type=int, default=1800)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--roads", type=int, default=5,
                    help="roads per axis (grid is roads x roads)")
    ap.add_argument("--length", type=float, default=2000.0,
                    help="road length in metres")
    ap.add_argument("--spacing", type=float, default=250.0,
                    help="distance between parallel roads in metres")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    L = args.length
    N = args.vehicles
    SUB = 2                  # sub-steps per recorded step
    dt_s = 1.0 / SUB

    # ---- road groups: (axis, road index, direction) ----
    groups = []
    for axis in (0, 1):                      # 0 = horizontal, 1 = vertical
        for k in range(args.roads):
            for d in (+1, -1):
                groups.append((axis, k, d))
    G = len(groups)

    g_axis = np.array([g[0] for g in groups])
    g_lane = np.array([g[1] * args.spacing + (2.0 if g[2] > 0 else -2.0)
                       for g in groups])
    g_dir = np.array([g[2] for g in groups])

    # ---- vehicles, started in platoons so clusters are visible ----
    veh_group = rng.integers(0, G, N)
    s = np.zeros(N)

    for g in range(G):
        idx = np.where(veh_group == g)[0]

        if len(idx) == 0:
            continue

        # platoons of ~20 cars, 9-14 m apart, spread along the road
        n_pl = max(1, len(idx) // 20)
        centers = rng.uniform(0, L, n_pl)
        pid = rng.permutation(len(idx)) % n_pl

        for p in range(n_pl):
            members = idx[pid == p]

            if len(members) == 0:
                continue

            back = np.cumsum(rng.uniform(9.0, 14.0, len(members)))
            s[members] = (centers[p] - back) % L

    vmax = rng.uniform(12.0, 14.0, N)            # individual desired speeds
    v = vmax * rng.uniform(0.8, 1.0, N)
    jitter = rng.normal(0, 0.7, N)               # lateral offset
    veh_len = 5.0
    acc = 2.0

    # ---- bottlenecks: (group, s0, s1, t_start, t_end, speed cap) ----
    T = args.steps
    pick_groups = rng.choice(G, size=min(3, G), replace=False)
    bottlenecks = []

    for n_b, g in enumerate(pick_groups):
        t0 = int(T * (0.12 + 0.22 * n_b))
        t1 = t0 + int(T * 0.35)
        b0 = L * rng.uniform(0.35, 0.65)
        bottlenecks.append((int(g), b0, b0 + 60.0, t0, t1, 0.4))

    rows = []

    for step in range(T):
        cap = vmax.copy()

        for (g, b0, b1, t0, t1, vc) in bottlenecks:
            if t0 <= step < t1:
                m = (veh_group == g) & (s >= b0) & (s <= b1)
                cap[m] = vc

        # random slowdowns (Nagel-Schreckenberg style noise)
        slow = rng.random(N) < 0.02
        cap[slow] = np.maximum(cap[slow] - 2.0, 0.0)

        # each recorded step is split in sub-steps so that followers can
        # keep a short (0.5 s) headway without driving through the leader
        for _ in range(SUB):
            v_new = np.minimum(v + acc * dt_s, cap)

            for g in range(G):
                idx = np.where(veh_group == g)[0]

                if len(idx) < 2:
                    continue

                order = idx[np.argsort(-s[idx])]     # leader first
                gap = s[order[:-1]] - s[order[1:]] - veh_len
                safe = np.maximum(gap, 0.0) / dt_s

                followers = order[1:]
                v_new[followers] = np.minimum(v_new[followers], safe)

                # roads are rings: the front car follows the last car
                # (through the seam), so nothing piles up at the road end
                gap0 = s[order[-1]] + L - s[order[0]] - veh_len
                lead = order[0]
                v_new[lead] = min(v_new[lead], max(gap0, 0.0) / dt_s)

            v = np.maximum(v_new, 0.0)
            s = s + v * dt_s

            wrap = s >= L
            s[wrap] -= L

        # ---- coordinates and velocity vector ----
        coord = np.where(g_dir[veh_group] > 0, s, L - s)
        lane = g_lane[veh_group] + jitter
        horiz = g_axis[veh_group] == 0

        x = np.where(horiz, coord, lane)
        y = np.where(horiz, lane, coord)

        sign = g_dir[veh_group]
        vx = np.where(horiz, sign * v, 0.0)
        vy = np.where(horiz, 0.0, sign * v)

        rows.append(pd.DataFrame({
            "step": step,
            "id": np.arange(N),
            "x": np.round(x, 2),
            "y": np.round(y, 2),
            "vx": np.round(vx, 3),
            "vy": np.round(vy, 3),
            "speed": np.round(v, 3),
        }))

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(args.out_dir, "trajectories.csv"), index=False)

    lines = []

    for k in range(args.roads):
        c = k * args.spacing
        lines.append([[0, c], [L, c]])
        lines.append([[c, 0], [c, L]])

    with open(os.path.join(args.out_dir, "network.json"), "w") as f:
        json.dump(lines, f)

    print("Vehicles:", N, " Steps:", T, " Rows:", len(df))
    print("Bottlenecks (road group, start step, end step):",
          [(b[0], b[3], b[4]) for b in bottlenecks])
    print("Wrote", os.path.join(args.out_dir, "trajectories.csv"))


if __name__ == "__main__":
    main()
