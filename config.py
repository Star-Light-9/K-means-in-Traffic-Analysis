"""
config.py - shared reader for config.ini (used by every Python script).

The C program (dbscan.c) reads the same file with its own small parser.
"""

import configparser
import os
import sys

DEFAULTS = {
    "paths": {"data_dir": "data"},
    "sumo": {
        "network": "grid", "grid_size": "8", "grid_length": "200",
        "steps": "400", "warmup": "60", "initial_vehicles": "800",
        "flow": "3.0", "jams": "3", "jam_start": "80", "jam_gap": "60",
        "jam_duration": "120", "seed": "1",
    },
    "dbscan": {
        "eps_pos": "30", "eps_vel": "4", "min_pts": "3",
        "split": "balanced", "reps": "1", "max_steps": "0",
    },
    "mpi": {"processes": "4", "launcher": "auto", "extra_args": ""},
    "tracking": {"stuck_speed": "2.0", "stuck_steps": "3"},
    "replay": {
        "seq_duration": "20", "fps": "10", "show_window": "true",
        "make_gif": "true", "gif_name": "replay.gif",
    },
    "benchmark": {
        "processes": "1 2 4 8", "reps": "3", "compare_split": "true",
    },
}

HERE = os.path.dirname(os.path.abspath(__file__))


class Config:
    def __init__(self, path=None):
        self.path = os.path.abspath(path or os.path.join(HERE, "config.ini"))

        if not os.path.isfile(self.path):
            sys.exit("Config file not found: %s" % self.path)

        self.base = os.path.dirname(self.path)

        cp = configparser.ConfigParser(
            interpolation=None, inline_comment_prefixes=None)
        cp.read_dict(DEFAULTS)
        cp.read(self.path, encoding="utf-8")
        self._cp = cp

    def str(self, section, key):
        return self._cp.get(section, key).strip()

    def int(self, section, key):
        return int(float(self.str(section, key)))

    def float(self, section, key):
        return float(self.str(section, key))

    def bool(self, section, key):
        return self._cp.getboolean(section, key)

    def ints(self, section, key):
        return [int(x) for x in self.str(section, key).replace(",", " ").split()]

    def resolve(self, path):
        """Relative paths are relative to the folder holding config.ini."""
        return path if os.path.isabs(path) else os.path.join(self.base, path)

    @property
    def data_dir(self):
        return self.resolve(self.str("paths", "data_dir"))

    def split_flag(self):
        """1 = balanced strips, 0 = equal-width strips."""
        v = self.str("dbscan", "split").lower()
        return 0 if v in ("equal", "equal-width", "0") else 1


def load(path=None):
    return Config(path)