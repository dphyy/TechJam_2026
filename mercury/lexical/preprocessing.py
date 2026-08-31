from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path

from .constraint_index import (
    CATALOG_INDEX_SCHEMA_VERSION,
    ConstraintIndex,
    default_catalog_index_path,
)
from .retrieval import CatalogSearch
from .vector_index import catalog_sha256


def build_catalog_index(
    catalog_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Atomically build the persistent FTS and exact-constraint artifact."""
    catalog = Path(catalog_path)
    output = (
        Path(output_path)
        if output_path is not None
        else default_catalog_index_path(catalog)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    search = CatalogSearch(catalog, use_prebuilt_index=False)
    try:
        if not isinstance(search.constraint_index, ConstraintIndex):
            raise RuntimeError("artifact builder requires the in-memory constraint index")
        target = sqlite3.connect(temporary)
        try:
            search.connection.backup(target)
            target.executescript(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) "
                "WITHOUT ROWID;"
                "CREATE TABLE product_rows("
                "parent_asin TEXT PRIMARY KEY, row_id INTEGER NOT NULL UNIQUE"
                ") WITHOUT ROWID;"
                "CREATE TABLE constraint_entries("
                "index_name TEXT NOT NULL, value TEXT NOT NULL, parent_asin TEXT NOT NULL,"
                "PRIMARY KEY(index_name, value, parent_asin)"
                ") WITHOUT ROWID;"
            )
            target.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                (
                    ("schema_version", CATALOG_INDEX_SCHEMA_VERSION),
                    ("catalog_sha256", catalog_sha256(catalog)),
                    ("catalog_rows", str(len(search._row_id_by_asin))),
                ),
            )
            target.executemany(
                "INSERT INTO product_rows VALUES (?, ?)",
                search._row_id_by_asin.items(),
            )
            target.executemany(
                "INSERT INTO constraint_entries VALUES (?, ?, ?)",
                search.constraint_index.iter_entries(),
            )
            target.commit()
            target.execute("VACUUM")
        finally:
            target.close()
    finally:
        search.close()
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild the catalog FTS and exact-constraint SQLite index"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()
    started = time.perf_counter()
    output = build_catalog_index(args.catalog, args.output)
    print(
        f"Wrote {output} ({output.stat().st_size / (1024 * 1024):.1f} MiB) "
        f"in {time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
