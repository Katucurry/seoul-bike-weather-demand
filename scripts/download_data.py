#!/usr/bin/env python3
"""Download the original UCI dataset; no API credentials required."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
ROOT = Path(__file__).resolve().parents[1]

def main():
    target = ROOT / "data" / "raw" / "SeoulBikeData.csv"
    if target.exists():
        print(f"Already exists: {target}")
        return
    with urlopen(URL, timeout=60) as response:
        payload = response.read(5_000_001)
    if len(payload) > 5_000_000:
        raise ValueError("Unexpected dataset download size")
    with ZipFile(BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if Path(name).name == "SeoulBikeData.csv"]
        if len(names) != 1:
            raise ValueError("Expected exactly one SeoulBikeData.csv")
        info = archive.getinfo(names[0])
        if info.file_size > 5_000_000:
            raise ValueError("Unexpected CSV size")
        content = archive.read(names[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(f"Saved: {target}")

if __name__ == "__main__":
    main()
