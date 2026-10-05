"""Download the processed TabArena artifacts of the given methods into $TABARENA_CACHE.

Run this once before submitting jobs, so that parallel jobs do not download the same
artifacts concurrently. Each method is several GB.
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from tabarena.models._artifacts.downloader_s3 import MethodDownloaderS3

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


def _download_processed_s3(downloader: MethodDownloaderS3, method_metadata):
    """Like `MethodDownloaderS3.download_processed`, but through a temporary file.

    TabArena reads the whole zip from S3 into memory before extracting it, which needs several
    GB of RAM per method. The processed artifacts are public, so an anonymous client is used.
    """
    key = (downloader.key_prefix / "processed.zip").as_posix()
    tmp_dir = tempfile.mkdtemp(prefix="tabarena_download_")
    try:
        zip_path = Path(tmp_dir) / "processed.zip"
        downloader.unsigned_client.download_file(downloader.bucket, key, str(zip_path))
        downloader._extract_zip(
            zip_path, dest_dir=Path(method_metadata.path_processed), clear_dir=downloader.clear_dirs
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_processed(method_metadata, retries: int):
    for attempt in range(1, retries + 1):
        try:
            downloader = method_metadata.method_downloader()
            if isinstance(downloader, MethodDownloaderS3):
                _download_processed_s3(downloader, method_metadata)
            else:
                downloader.download_processed()
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
