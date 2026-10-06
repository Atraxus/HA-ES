"""Measure the memory needed to load a saved AutoGluon predictor and predict with it.

Run by config_stats.py in a fresh process for every config:

    python memory_probe.py <predictor_path> <data.csv>

Prints the peak resident memory (in bytes) while loading the predictor and predicting on the data.
The predictor is first loaded and used once as a warm-up, then freed, and only the second load and
prediction is measured. The warm-up loads all lazily imported library code, which the members of an
ensemble share in one process, so it is not attributed to every single model.

Linux only: the peak resident memory is reset via /proc/self/clear_refs, and freed memory is
returned to the operating system with glibc's malloc_trim.
"""

import ctypes
import gc
import json
import sys


def _status_bytes(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024  # reported in kB
    raise RuntimeError(f"{field} not found in /proc/self/status")


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

    # Reset the peak resident memory (VmHWM) to the current resident memory and use it as baseline.
    # The peak only grows from here, so the result cannot become negative.
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    baseline = _status_bytes("VmHWM")

    predictor = _load_and_predict(predictor_path, data)
    return _status_bytes("VmHWM") - baseline


if __name__ == "__main__":
    print(json.dumps({"inference_memory": measure(sys.argv[1], sys.argv[2])}))
