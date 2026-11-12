"""
run_sumo.py - run a SUMO traffic simulation and record every vehicle's
position and velocity at every step.

Output (in --out-dir):
    trajectories.csv   step,id,x,y,vx,vy,speed
    network.json       road polylines for the background of the replay

Install SUMO first:   pip install eclipse-sumo traci sumolib
(or the installer from https://sumo.dlr.de, then set SUMO_HOME)

Usage:
    python run_sumo.py --out-dir data                      # built-in grid city
    python run_sumo.py --out-dir data --net pune.net.xml   # your own network

Congestion is created on purpose: dense demand plus a few temporary
"blockages" (one vehicle is held at speed 0, queues form behind it, then
it is released).
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys

if os.environ.get("SUMO_HOME"):
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

try:
    import traci
    import traci.constants as tc
    import sumolib
except ImportError:
    sys.exit("SUMO python packages not found.\n"
             "Install with:  pip install eclipse-sumo traci sumolib")


def make_grid_net(path, n, length):
    """Create an n x n grid road network with netgenerate."""
    netgenerate = sumolib.checkBinary("netgenerate")

    base = [netgenerate, "--grid",
            "--grid.number", str(n),
            "--grid.length", str(length),
            "--default.lanenumber", "2",
            "--default.speed", "13.9",
            "--no-warnings", "true",
            "-o", path]

    # Try with traffic lights first; fall back if this SUMO version
    # does not accept the option.
    for extra in (["-j", "traffic_light"], ["--tls.guess", "true"], []):
        res = subprocess.run(base + extra, capture_output=True, text=True)

        if res.returncode == 0 and os.path.exists(path):
            print("Network created", "(options: %s)" % " ".join(extra)
                  if extra else "(no traffic lights)")
            return

    sys.exit("netgenerate failed:\n" + res.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--net", default=None,
                    help="existing .net.xml (default: generate a grid city)")
    ap.add_argument("--grid", type=int, default=8,
                    help="grid size when generating a network")
    ap.add_argument("--grid-length", type=float, default=200.0)
    ap.add_argument("--steps", type=int, default=400,
                    help="recorded steps (after warm-up)")
    ap.add_argument("--warmup", type=int, default=60,
                    help="steps simulated but not recorded")
    ap.add_argument("--initial", type=int, default=800,
                    help="vehicles released at the start")
    ap.add_argument("--flow", type=float, default=3.0,
                    help="new vehicles per step (average)")
    ap.add_argument("--jams", type=int, default=3,
                    help="number of temporary blockages")
    ap.add_argument("--jam-start", type=int, default=80,
                    help="step (from recording start) of the first blockage")
    ap.add_argument("--jam-gap", type=int, default=60)
    ap.add_argument("--jam-duration", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    net_file = args.net

    if net_file is None:
        net_file = os.path.join(args.out_dir, "grid.net.xml")
        make_grid_net(net_file, args.grid, args.grid_length)

    net = sumolib.net.readNet(net_file)

    edges = [e for e in net.getEdges()
             if e.allows("passenger") and e.getLength() > 30]

    if len(edges) < 2:
        sys.exit("Network has no usable edges.")

    # ---- background polylines for the replay ----
    lines = [[list(p) for p in e.getShape()] for e in net.getEdges()]

    with open(os.path.join(args.out_dir, "network.json"), "w") as f:
        json.dump(lines, f)

    # ---- start SUMO ----
    sumo = sumolib.checkBinary("sumo")

    traci.start([sumo, "-n", net_file,
                 "--step-length", "1",
                 "--no-warnings", "true",
                 "--no-step-log", "true",
                 "--time-to-teleport", "120",
                 "--seed", str(args.seed)])

    counter = 0
    numeric_id = {}

    def spawn():
        nonlocal counter

        for _ in range(10):
            a, b = random.sample(edges, 2)

            try:
                route = traci.simulation.findRoute(a.getID(), b.getID())
            except traci.TraCIException:
                continue

            if len(route.edges) < 3:
                continue

            rid = "r%d" % counter
            vid = "v%d" % counter

            try:
                traci.route.add(rid, list(route.edges))
                traci.vehicle.add(vid, rid,
                                  departLane="best", departSpeed="max")
            except traci.TraCIException:
                continue

            numeric_id[vid] = counter
            counter += 1
            return

    for _ in range(args.initial):
        spawn()

    total_steps = args.warmup + args.steps
    jam_events = []     # (start_step, end_step, vehicle id or None)

    for k in range(args.jams):
        start = args.warmup + args.jam_start + k * args.jam_gap
        jam_events.append([start, start + args.jam_duration, None])

    out_path = os.path.join(args.out_dir, "trajectories.csv")

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "id", "x", "y", "vx", "vy", "speed"])

        subscribed = set()

        for sim_step in range(total_steps):
            # ---- demand ----
            n_new = int(args.flow) + (1 if random.random() < args.flow % 1 else 0)

            for _ in range(n_new):
                spawn()

            traci.simulationStep()

            # ---- blockages ----
            for ev in jam_events:
                if sim_step == ev[0]:
                    moving = [v for v in traci.vehicle.getIDList()
                              if traci.vehicle.getSpeed(v) > 3.0]

                    if moving:
                        ev[2] = random.choice(moving)
                        traci.vehicle.setSpeed(ev[2], 0.0)

                if sim_step == ev[1] and ev[2] is not None:
                    try:
                        traci.vehicle.setSpeed(ev[2], -1)
                    except traci.TraCIException:
                        pass

            # ---- record ----
            for vid in traci.simulation.getDepartedIDList():
                if vid not in subscribed:
                    traci.vehicle.subscribe(
                        vid, [tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ANGLE])
                    subscribed.add(vid)

            if sim_step < args.warmup:
                continue

            rec_step = sim_step - args.warmup
            results = traci.vehicle.getAllSubscriptionResults()

            for vid, r in results.items():
                if tc.VAR_POSITION not in r:
                    continue

                x, y = r[tc.VAR_POSITION]
                speed = r[tc.VAR_SPEED]
                ang = math.radians(r[tc.VAR_ANGLE])   # 0 = north, clockwise

                if vid not in numeric_id:
                    numeric_id[vid] = len(numeric_id) + 10 ** 6

                writer.writerow([
                    rec_step, numeric_id[vid],
                    "%.2f" % x, "%.2f" % y,
                    "%.3f" % (speed * math.sin(ang)),
                    "%.3f" % (speed * math.cos(ang)),
                    "%.3f" % speed,
                ])

            if rec_step % 50 == 0:
                print("recorded step", rec_step, "/", args.steps,
                      " vehicles:", len(results))

    traci.close()
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
