#!/usr/bin/env python3
"""
Utility to shorten long file and directory names under a root (default: toc_openai_filtered).

Why: Some TOC entries contain very long book titles which exceed filesystem limits
for a single path segment (commonly 255 bytes), causing errors like:
  fatal: cannot create directory ...: File name too long

What it does:
- Traverses the tree, computes safe, shortened names for segments that are too long
  or contain risky characters.
- Renames paths bottom-up (deepest first) to avoid breaking traversal.
- Ensures uniqueness by appending a short hash suffix if a collision would occur.
- Preserves file extensions.

Usage:
  Dry run (default):
    python tools/shorten_filenames.py

  Apply changes:
    python tools/shorten_filenames.py --apply

  Specify a different root:
    python tools/shorten_filenames.py --root path/to/root --apply

Outputs:
- When applying, writes a JSONL mapping file at the project root named
  rename_mapping_YYYYmmdd_HHMMSS.jsonl with one record per rename:
  {"old": "/abs/old", "new": "/abs/new"}

Notes:
- We keep Unicode; we mainly shorten and normalize. If you need transliteration
  to ASCII, we can add it later (would require an extra dependency).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Configuration
MAX_SEG_LEN = 80  # Max characters per path segment (safe under common filesystems)
FORBIDDEN_CHARS = set("\0")  # NUL is not allowed; slash is a separator and not present in segment
WHITESPACE_REPLACE_WITH = " "  # collapse whitespace later


def _safe_segment(seg: str) -> str:
    # Strip leading/trailing whitespace
    seg = seg.strip()
    # Replace sequences of whitespace with single spaces
    seg = " ".join(seg.split())
    # Remove forbidden characters (keep Unicode letters, digits, punctuation except path sep)
    seg = "".join(ch for ch in seg if ch not in FORBIDDEN_CHARS and ch != "/")
    if not seg:
        seg = "untitled"
    return seg


def _truncate_with_hash(name_no_ext: str, max_len: int, orig_for_hash: str) -> str:
    if len(name_no_ext) <= max_len:
        return name_no_ext
    # Short hash from original full name to help uniqueness and traceability
    h = hashlib.sha1(orig_for_hash.encode("utf-8", errors="ignore")).hexdigest()[:6]
    # Leave room for suffix "-" + 6 chars
    keep = max(1, max_len - 7)
    return f"{name_no_ext[:keep]}-{h}"


def _shorten_filename(filename: str) -> str:
    """Shorten a filename preserving extension and ensuring max segment length."""
    if "." in filename and not filename.startswith('.'):
        stem = filename.rsplit('.', 1)[0]
        ext = '.' + filename.rsplit('.', 1)[1]
    else:
        stem, ext = filename, ''
    safe_stem = _safe_segment(stem)
    new_stem = _truncate_with_hash(safe_stem, MAX_SEG_LEN - len(ext), filename)
    return new_stem + ext


def _shorten_dirname(dirname: str) -> str:
    safe = _safe_segment(dirname)
    return _truncate_with_hash(safe, MAX_SEG_LEN, dirname)


def _iter_paths_bottom_up(root: Path) -> Iterable[Path]:
    """Yield all files and directories under root, deepest first.
    We will rename files first, then directories from deepest to root.
    """
    # Collect all paths
    all_paths: List[Path] = []
    for p in root.rglob("*"):
        all_paths.append(p)
    # Sort by depth descending so children come before parents
    all_paths.sort(key=lambda p: len(p.parts), reverse=True)
    return all_paths


def plan_renames(root: Path) -> List[Tuple[Path, Path]]:
    """Compute a list of (old_path, new_path) to rename. Does not touch disk."""
    renames: List[Tuple[Path, Path]] = []

    # Track occupied sibling names per parent to avoid collisions in planning phase
    sibling_taken: Dict[Path, set] = {}

    for path in _iter_paths_bottom_up(root):
        parent = path.parent
        if parent not in sibling_taken:
            try:
                sibling_taken[parent] = {p.name for p in parent.iterdir()}
            except FileNotFoundError:
                sibling_taken[parent] = set()

        old_name = path.name
        if path.is_dir():
            new_name = _shorten_dirname(old_name)
        else:
            new_name = _shorten_filename(old_name)

        if new_name == old_name:
            continue  # Nothing to do

        candidate = parent / new_name
        # Ensure uniqueness within the parent directory
        if new_name in sibling_taken[parent]:
            # Append extra hash from full path to avoid collision
            h = hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:6]
            if path.is_dir():
                base = _truncate_with_hash(_safe_segment(old_name), MAX_SEG_LEN - 7, old_name)
                new_name = f"{base}-{h}"
            else:
                if "." in new_name and not new_name.startswith('.'):
                    stem, ext = new_name.rsplit('.', 1)
                    ext = '.' + ext
                else:
                    stem, ext = new_name, ''
                stem = _truncate_with_hash(stem, MAX_SEG_LEN - len(ext) - 7, old_name)
                new_name = f"{stem}-{h}{ext}"
            candidate = parent / new_name

        renames.append((path, candidate))

        # Update sibling set to reflect planned new name
        sibling_taken[parent].discard(old_name)
        sibling_taken[parent].add(new_name)

    return renames


def apply_renames(renames: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    """Apply renames. Returns list of successfully applied (old, new)."""
    applied: List[Tuple[Path, Path]] = []
    for old, new in renames:
        try:
            if not old.exists():
                continue
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            applied.append((old, new))
        except Exception as e:
            print(f"WARN: Failed to rename {old} -> {new}: {e}")
    return applied


def write_mapping(applied: List[Tuple[Path, Path]], out_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"rename_mapping_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for old, new in applied:
            f.write(json.dumps({"old": str(old), "new": str(new)}, ensure_ascii=False) + "\n")
    return out_path


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Shorten long file/dir names under a root directory.")
    parser.add_argument("--root", type=str, default="toc_openai_filtered", help="Root directory to process")
    parser.add_argument("--apply", action="store_true", help="Actually perform renames (default: dry run)")
    parser.add_argument("--max-seg-len", type=int, default=MAX_SEG_LEN, help="Maximum characters per path segment")
    args = parser.parse_args(argv)

    global MAX_SEG_LEN
    MAX_SEG_LEN = int(args.max_seg_len)

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"ERROR: Root not found or not a directory: {root}")
        return 2

    renames = plan_renames(root)
    if not renames:
        print("No renames needed. All names within limits.")
        return 0

    print(f"Planned renames: {len(renames)}")
    for old, new in renames[:50]:
        print(f"  {old} -> {new}")
    if len(renames) > 50:
        print(f"  ... and {len(renames) - 50} more")

    if not args.apply:
        print("Dry run complete. Re-run with --apply to perform changes.")
        return 0

    applied = apply_renames(renames)
    print(f"Applied renames: {len(applied)}/{len(renames)}")
    mapping_path = write_mapping(applied, Path.cwd())
    print(f"Mapping written to: {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
