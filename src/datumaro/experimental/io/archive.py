"""
Archive utilities for Datumaro Experimental.

This module contains helpers to package a saved dataset directory into a single
zip file and to extract such archives during loading. It is intentionally
minimal and independent from the Dataset class to keep IO responsibilities
separated and easy to test.
"""
from __future__ import annotations

from pathlib import Path


def create_zip_archive(
    *, out_parquet_path: Path, media_dir: str, archive_path: Path | None
) -> None:
    """
    Create a zip archive for a saved dataset directory.

    The archive contains:
    - the Parquet file referenced by `out_parquet_path`,
    - the sidecar JSON with the same stem, and
    - the media directory (if it exists) located next to the Parquet file.

    Args:
        out_parquet_path: Path to the already written Parquet file that should be archived.
        media_dir: Directory name (relative to the Parquet parent) where media was saved.
        archive_path: Optional explicit output path to the zip file. If None, a sibling
            file with the same stem as the Parquet and `.zip` extension will be created.

    Notes:
        - Uses ZIP_DEFLATED compression.
        - Paths inside the archive are stored relative to the Parquet parent directory
          to keep the bundle portable after extraction.
    """
    import zipfile

    out_dir = out_parquet_path.parent
    zip_path = archive_path if archive_path is not None else out_parquet_path.with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    to_add: list[Path] = []
    if out_parquet_path.exists():
        to_add.append(out_parquet_path)
    sidecar_path = out_parquet_path.with_suffix(".json")
    if sidecar_path.exists():
        to_add.append(sidecar_path)
    media_root = out_dir / media_dir
    if media_root.exists() and media_root.is_dir():
        for p in media_root.rglob("*"):
            to_add.append(p)

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_add:
            try:
                arcname = p.relative_to(out_dir)
            except Exception:
                arcname = p.name
            zf.write(p, arcname.as_posix())


def extract_zip_to_dir(path: str, extract_dir: str | None) -> tuple[Path, Path]:
    """
    Extract a dataset zip archive to a directory and locate the Parquet file.

    The function unpacks the given `path` (a `.zip`) into `extract_dir` if
    provided, otherwise into a temporary directory. It then searches for the
    contained Parquet file using the following strategy:
    1) try `<archive_stem>.parquet` at the extraction root,
    2) if not found, pick the single Parquet present,
    3) if multiple candidates exist, prefer a top-level Parquet; otherwise pick
       the first found.

    Args:
        path: Path to the zip archive produced by `create_zip_archive`.
        extract_dir: Optional directory to extract into. When None, a temp
            directory is created and returned.

    Returns:
        A tuple `(parquet_path, extract_root)` where `parquet_path` points to the
        discovered Parquet file and `extract_root` is the directory where the
        archive was extracted. Callers may use `extract_root` as the rebase root
        for media paths.
    """
    import zipfile
    from pathlib import Path as _Path

    # Decide extraction root
    if extract_dir is None:
        import tempfile

        extract_root = _Path(tempfile.mkdtemp(prefix="datumaro_ds_"))
    else:
        extract_root = _Path(extract_dir)
        extract_root.mkdir(parents=True, exist_ok=True)
    # Extract all
    with zipfile.ZipFile(path, mode="r") as zf:
        zf.extractall(extract_root)
    # Locate parquet inside (prefer matching stem, then top-level, then any)
    expected_name = _Path(path).with_suffix(".parquet").name
    candidate = extract_root / expected_name
    if not candidate.exists():
        parquets = list(extract_root.glob("*.parquet")) or list(extract_root.rglob("*.parquet"))
        if len(parquets) == 1:
            candidate = parquets[0]
        elif len(parquets) > 1:
            top = [p for p in parquets if p.parent == extract_root]
            candidate = top[0] if top else parquets[0]
        else:
            raise FileNotFoundError("No .parquet file found inside the archive")
    return candidate, extract_root
