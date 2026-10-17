"""Demo data for the UI, so a fresh checkout (or `docker compose up`) has something to show.

    python scripts/seed_demo.py            # unpack demo/demo_traces.tar.gz into traces/, if empty
    python scripts/seed_demo.py --force    # unpack even if traces/ already has traces
    python scripts/seed_demo.py --export   # pack the current traces/ into the snapshot

The snapshot holds real runs from this project: every trace (with prompts, raw
replies and reasoning), the diagnoses, the flags, and the evaluation database
behind reports/diagnosis_accuracy.md. No API key is needed to browse it.
Exporting makes local file paths relative, so no home directories end up in it.
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

from forensics.config import PROJECT_ROOT, load_settings

SNAPSHOT = PROJECT_ROOT / "demo" / "demo_traces.tar.gz"
DATABASES = ("index.sqlite", "evaluation.sqlite")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", action="store_true", help="pack traces/ into the snapshot")
    parser.add_argument("--force", action="store_true", help="unpack even if traces/ has traces")
    args = parser.parse_args()

    settings = load_settings()
    traces_dir = settings.resolve(settings.paths.traces_dir)
    return export(traces_dir) if args.export else unpack(traces_dir, args.force)


def export(traces_dir: Path) -> int:
    files = sorted(traces_dir.glob("*.json")) + sorted((traces_dir / "diagnoses").glob("*.json"))
    if not files:
        print(f"No traces in {traces_dir} to export.", file=sys.stderr)
        return 1
    local = f"{PROJECT_ROOT}/"
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SNAPSHOT, "w:gz") as tar, tempfile.TemporaryDirectory() as tmp:
        for path in files:
            text = path.read_text(encoding="utf-8").replace(local, "")
            _add(tar, str(path.relative_to(traces_dir)), text.encode("utf-8"))
        for name in DATABASES:
            if (traces_dir / name).is_file():
                copy = Path(tmp) / name  # sqlite's backup API: a consistent copy even if in use
                with sqlite3.connect(traces_dir / name) as src, sqlite3.connect(copy) as dst:
                    src.backup(dst)
                _add(tar, name, copy.read_bytes())
    print(f"Wrote {_show(SNAPSHOT)}: {len(files)} JSON files, "
          f"{SNAPSHOT.stat().st_size / 1e6:.1f} MB")
    return 0


def unpack(traces_dir: Path, force: bool) -> int:
    if not SNAPSHOT.is_file():
        print(f"No snapshot at {_show(SNAPSHOT)}.", file=sys.stderr)
        return 1
    if any(traces_dir.glob("*.json")) and not force:
        print(f"{traces_dir} already has traces; leaving them alone (use --force to unpack anyway).")
        return 0
    traces_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SNAPSHOT, "r:gz") as tar:
        tar.extractall(traces_dir, filter="data")  # "data": refuses paths outside traces_dir
        count = sum(1 for m in tar.getmembers() if m.name.endswith(".json"))
    print(f"Unpacked {count} trace/diagnosis files into {traces_dir}.")
    return 0


def _show(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0  # reproducible archive
    tar.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    sys.exit(main())
