"""Free disk space in finished runs: delete what can be rebuilt, keep the results.

Only runs whose run_info.json says ``"status": "done"`` are touched. Per run it deletes:

    decoder_corpus/*.txt          the token corpus (only needed to train again: notebook 02)
    embeddings/*.npy              per-transfer embeddings, labels, head scores
    embeddings/*_features.parquet the same embeddings + raw features for notebook 05
    embeddings/finetuned_model.pt fine-tuned weights (ft_* runs)
    model/checkpoints/<other>     every checkpoint except the one latest_checkpoint.json names

It keeps everything notebooks 06 and 07 read: tokenizer/, the checkpoint in use,
outputs/ (metrics, predictions with every score), notebooks/, logs/ and the json files.
Rerunning notebook 05 alone afterwards needs notebook 04 first (`--stages 04,05 --resume`):
04 sees the missing files and recomputes them.

    python scripts/clean_runs.py                  # dry run: what would go, and how much
    python scripts/clean_runs.py --delete         # do it
    python scripts/clean_runs.py base seed_1      # only these runs
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Patterns deleted inside a finished run, relative to the run folder.
PATTERNS = (
    "decoder_corpus/*.txt",
    "embeddings/*.npy",
    "embeddings/*_features.parquet",
    "embeddings/finetuned_model.pt",
)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def removable(run_dir: Path) -> List[Path]:
    """Files and checkpoint folders of a finished run that can go (nothing for other runs)."""
    info_path = run_dir / "run_info.json"
    if not info_path.exists():
        return []
    try:
        if json.loads(info_path.read_text()).get("status") != "done":
            return []
    except (OSError, ValueError):
        return []
    out = [p for pattern in PATTERNS for p in sorted(run_dir.glob(pattern))]

    # Checkpoints: keep the one in use. If it is not under this run's checkpoints
    # folder (a moved or renamed runs folder), leave every checkpoint alone.
    ckpt_dir = run_dir / "model" / "checkpoints"
    latest = run_dir / "model" / "latest_checkpoint.json"
    if ckpt_dir.is_dir() and latest.exists():
        try:
            keep = Path(json.loads(latest.read_text())["checkpoint"]).resolve()
        except (OSError, ValueError, KeyError):
            keep = None
        folders = [p for p in sorted(ckpt_dir.iterdir()) if p.is_dir()]
        if keep is not None and any(p.resolve() == keep for p in folders):
            out += [p for p in folders if p.resolve() != keep]
    return out


def clean_run(run_dir: Path, delete: bool = False) -> Tuple[int, List[Path]]:
    """Bytes freed (or that would be) and the paths, for one run."""
    paths = removable(run_dir)
    total = sum(_size(p) for p in paths)
    if delete:
        for p in paths:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    return total, paths


def _gb(n: int) -> str:
    return f"{n / 1e9:8.1f} GB"


def main(argv=None) -> int:
    from src.experiment import runs_root

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="*", help="run names (default: every run)")
    parser.add_argument("--delete", action="store_true", help="delete (default: dry run)")
    parser.add_argument("--root", type=Path, default=None, help="runs folder (default: runs_dir)")
    args = parser.parse_args(argv)

    root = args.root or runs_root()
    runs = [root / name for name in args.runs] if args.runs else sorted(
        p for p in root.iterdir() if p.is_dir())
    grand = 0
    for run_dir in runs:
        total, paths = clean_run(run_dir, delete=args.delete)
        if paths:
            print(f"{_gb(total)}  {run_dir.name}  ({len(paths)} files/folders)")
        grand += total
    verb = "freed" if args.delete else "would free (dry run; add --delete)"
    print(f"{_gb(grand)}  total {verb}  in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
