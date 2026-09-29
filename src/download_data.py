"""Download the processed TabArena artifacts of the given methods into $TABARENA_CACHE.

Run this once before submitting jobs, so that parallel jobs do not download the same
artifacts concurrently. Each method is several GB.
"""

import argparse
import time

from tabarena_data import DEFAULT_METHODS, get_context


def parse_args():
    parser = argparse.ArgumentParser(description="Download TabArena artifacts.")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Download attempts per method (a download that breaks off restarts from scratch).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="TabArena methods to download.",
    )
    return parser.parse_args()


def download_processed(method_metadata, retries: int):
    for attempt in range(1, retries + 1):
        try:
            method_metadata.method_downloader().download_processed()
            return
        except Exception as e:
            if attempt == retries:
                raise
            print(f"\tAttempt {attempt}/{retries} failed ({e}), retrying ...")
            time.sleep(30)


def main(methods: list[str], retries: int = 3):
    context = get_context()
    for method in methods:
        method_metadata = context.method_metadata(method=method)
        # The processed directory can exist without the artifacts (e.g. holding only the
        # configs downloaded by config_stats.py), so check by loading them
        try:
            method_metadata.load_processed(download=False)
            print(f"{method}: already at {method_metadata.path_processed}")
        except Exception:
            print(f"{method}: downloading to {method_metadata.path_processed} ...")
            download_processed(method_metadata, retries)
        # Used by src/config_stats.py
        method_metadata.load_configs_hyperparameters(download="auto")


if __name__ == "__main__":
    args = parse_args()
    main(args.methods, args.retries)
