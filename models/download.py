import hashlib
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_DIR = Path("models/pretrained")

SOURCES = {
    "aasist.pth": {
        "url": "https://raw.githubusercontent.com/clovaai/aasist/main/pretrained/aasist.pth",
        "sha256": None,
    },
    "rawnet2.pth": {
        "url": "https://raw.githubusercontent.com/Jungjee/RawNet/master/pretrained/rawnet2.pth",
        "sha256": None,
    },
}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(name: str, url: str, sha256: str | None) -> None:
    target = MODEL_DIR / name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    request = Request(url, headers={"User-Agent": "VoxGuard/1.0"})
    with urlopen(request) as response, target.open("wb") as handle:
        handle.write(response.read())
    if sha256:
        digest = sha256_file(target)
        if digest != sha256:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {name}: {digest} != {sha256}")


def ensure_weights() -> None:
    if os.environ.get("VOXGUARD_SKIP_DOWNLOAD") == "1":
        return
    for name, info in SOURCES.items():
        download_file(name, info["url"], info["sha256"])


def main() -> None:
    try:
        ensure_weights()
        print("Done.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
