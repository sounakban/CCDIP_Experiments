"""
Merge the per-chunk CSVs a job array wrote into one file, and refuse a short merge.

WHY A SEPARATE STEP
    `run_human_inference.py` and `run_synthetic_recovery.py` each fit their share of the
    work in a SLURM array task and write into that task's own directory, so no two tasks
    ever append to one file. The read-outs downstream want one file. This puts the pieces
    back together and, because a merge that quietly dropped a chunk would produce plots
    that look complete, it checks the row count against what the run was supposed to
    produce and exits non-zero when they disagree -- the sbatch gates every later stage on
    that exit code.

Typical usage::

    uv run python -m human_modeling.inference.scripts.merge_chunk_csvs \\
        --inputs temp/fit/chunk_*/fit_variant_A_full.csv \\
        --out temp/fit/fit_variant_A_full.csv --key subject_id game_index --expect 230
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import sys
from pathlib import Path


def merge_rows(paths: list[Path], key: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """
    Concatenate the rows of several same-schema CSVs, first occurrence winning on a key clash.

    :param paths: The CSVs, read in the order given.
    :type paths: list[Path]
    :param key: Column names that identify a row; a repeat is dropped, not duplicated.
    :type key: list[str]
    :return: ``(header, rows)`` -- the first file's header, and the merged rows in file order.
    :rtype: tuple[list[str], list[dict[str, str]]]
    :raises ValueError: If a later file's header differs from the first's.
    """
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        with io.open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue                                  # an empty chunk file
            if header is None:
                header = list(reader.fieldnames)
            elif list(reader.fieldnames) != header:
                raise ValueError(f"{path} has a different header from {paths[0]}")
            for row in reader:
                k = tuple(row[c] for c in key)
                if k in seen:
                    continue
                seen.add(k)
                rows.append(row)
    return header or [], rows


def main() -> None:
    """CLI: glob the inputs, merge, check the count, write."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="CSV paths or globs (quoted, so this script expands them)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", nargs="+", required=True, help="columns identifying a row")
    ap.add_argument("--expect", type=int, required=True,
                    help="required merged row count; a mismatch exits 1 and writes nothing")
    args = ap.parse_args()

    paths = sorted({Path(p) for pattern in args.inputs for p in glob.glob(pattern)})
    if not paths:
        sys.exit(f"no files match {args.inputs}")
    header, rows = merge_rows(paths, args.key)
    print(f"merged {len(rows)} rows from {len(paths)} files")
    if len(rows) != args.expect:
        sys.exit(f"INCOMPLETE: {len(rows)} rows merged, {args.expect} expected -- a chunk is "
                 f"missing or short; re-run the array before merging")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
