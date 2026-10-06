"""Measure the memory a saved AutoGluon predictor occupies once loaded and used for prediction.

Run by config_stats.py in a fresh process for every config:

    python memory_probe.py <predictor_path> <data.csv>

Prints the increase of the resident memory (in bytes) from loading the predictor and predicting on
the data. The predictor is first loaded and used once as a warm-up, then freed, and only the second
load and prediction is measured. The warm-up loads all lazily imported library code, which the
members of an ensemble share in one process, so it is not attributed to every single model.

The resident memory is read from /proc/self/smaps_rollup, which is exact. The per-process counters
in /proc/self/status (VmRSS, VmHWM) are approximations whose error grows with the number of CPUs,
which made peak measurements of small models unreliable on large cluster nodes.

Linux only: requires /proc/self/smaps_rollup and glibc's malloc_trim.
"""

import ctypes
import gc
import json
import sys


def _resident_memory() -> int:
    """Exact resident memory of this process in bytes."""
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            if line.startswith("Rss:"):
                return int(line.split()[1]) * 1024  # reported in kB
    raise RuntimeError("Rss not found in /proc/self/smaps_rollup")


def _load_and_predict(predictor_path, data):
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(predictor_path, verbosity=0)
    predictor.predict(data)
    return predictor


def measure(predictor_path: str, data_path: str) -> int:
    import pandas as pd

    data = pd.read_csv(data_path)

    # Warm-up, then free everything the predictor allocated
    predictor = _load_and_predict(predictor_path, data)
    del predictor
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)

    baseline = _resident_memory()
    predictor = _load_and_predict(predictor_path, data)  # noqa: F841 (kept alive for the measurement)
    return _resident_memory() - baseline


if __name__ == "__main__":
    print(json.dumps({"inference_memory": measure(sys.argv[1], sys.argv[2])}))
