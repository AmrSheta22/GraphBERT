from __future__ import annotations

import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print the highest numbered model checkpoint.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--fallback", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    weights = ("model.safetensors", "pytorch_model.bin")
    checkpoints = []
    if root.exists():
        for path in root.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match and path.is_dir() and any((path / name).exists() for name in weights):
                checkpoints.append((int(match.group(1)), path))

    if checkpoints:
        print(max(checkpoints)[1])
    elif any((root / name).exists() for name in weights):
        print(root)
    else:
        print(args.fallback)


if __name__ == "__main__":
    main()
