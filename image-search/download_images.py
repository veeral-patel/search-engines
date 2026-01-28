import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

from clip_utils import ensure_dir


def download_image(url: str, dest: Path, timeout: int = 20) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a batch of images.")
    parser.add_argument("--count", type=int, default=100, help="Number of images.")
    parser.add_argument(
        "--out-dir", type=str, default="data/images", help="Output directory."
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Square image size in pixels."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for URLs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    count = args.count
    size = args.size
    seed = args.seed

    successes = 0
    with tqdm(total=count, desc="Downloading") as pbar:
        i = 0
        while successes < count:
            image_id = seed + i
            url = f"https://picsum.photos/seed/{image_id}/{size}/{size}"
            dest = out_dir / f"image_{successes:04d}.jpg"
            ok = download_image(url, dest)
            if ok:
                successes += 1
                pbar.update(1)
            else:
                time.sleep(0.2)
            i += 1


if __name__ == "__main__":
    main()
