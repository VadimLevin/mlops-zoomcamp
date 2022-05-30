__all__ = ["fetch", "url_to_hash", "url_to_filename"]

import tempfile
from pathlib import Path
from typing import Callable
import requests
import os


def url_to_hash(url: str) -> str:
    import hashlib

    return hashlib.md5(url.encode("utf-8")).hexdigest()


def url_to_filename(url: str) -> str:
    return url.split("/")[-1]


def fetch(url: str, force: bool = False,
          output_dir: str = tempfile.gettempdir(),
          naming_policy: Callable[[str], str] = url_to_filename) -> bytes:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filepath = Path(output_dir, naming_policy(url))

    has_cached = output_filepath.is_file() and \
        os.stat(output_filepath).st_size > 0

    is_download_required = force or not has_cached
    if is_download_required:
        print("Fetching", url, "to", output_filepath)
        response = requests.get(url)
        response.raise_for_status()
        data = response.content

        temp = output_filepath.with_suffix(".tmp")
        with open(temp, "wb") as fh:
            fh.write(data)
        temp.replace(output_filepath)
    else:
        print(url, "is already downloaded as", output_filepath)
        with open(output_filepath, "rb") as fh:
            data = fh.read()
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optionally download files from the specified URL"
    )
    parser.add_argument("url", type=str,
                        help="URL to download file from")
    parser.add_argument("--out", "-o", type=Path, default=Path.cwd(),
                        help="Directory to output downloaded files")
    parser.add_argument("--use_hash_name", action="store_true",
                        help="Use hash instead of filename")
    parser.add_argument("--force", action="store_true",
                        help="Download file regardless of its existence")

    args = parser.parse_args()
    fetch(args.url, args.force, args.out,
          url_to_hash if args.use_hash_name else url_to_filename)
