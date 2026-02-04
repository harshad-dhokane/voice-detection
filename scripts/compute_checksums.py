import hashlib
from pathlib import Path

MODEL_DIR = Path("models/pretrained")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    for file_path in sorted(MODEL_DIR.glob("*.pth")):
        print(f"{file_path.name}: {sha256_file(file_path)}")


if __name__ == "__main__":
    main()
