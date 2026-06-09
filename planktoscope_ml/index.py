"""LanceDB vector store over the unlabeled ROI pool + query-by-example mining.

The ~190k unlabeled wharf ROIs are embedded once and stored alongside their
metadata (sample date, path). To grow a rare class, embed its few seed images
and retrieve their nearest neighbors from the pool for manual verification --
a far more targeted way to find rare morphotypes than waiting for a clustering
run to surface them.

At this scale (~190k x 384) LanceDB's default flat search is fast and exact, so
no ANN index is required.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


def _vector_table(embeddings: np.ndarray, metadata: pd.DataFrame) -> pa.Table:
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, dim = embeddings.shape
    if len(metadata) != n:
        raise ValueError(f"metadata rows ({len(metadata)}) != embeddings ({n})")
    vec = pa.FixedSizeListArray.from_arrays(
        pa.array(embeddings.reshape(-1)), dim)
    cols = {"vector": vec}
    for c in metadata.columns:
        cols[c] = pa.array(metadata[c].tolist())
    return pa.table(cols)


def build_index(db_path, embeddings: np.ndarray, metadata: pd.DataFrame,
                table_name: str = "pool", mode: str = "overwrite"):
    """Create/overwrite a LanceDB table of embeddings + metadata.

    ``metadata`` must be row-aligned with ``embeddings`` and contain at least a
    ``path`` column. Returns the opened table.
    """
    import lancedb
    db = lancedb.connect(str(db_path))
    return db.create_table(table_name, data=_vector_table(embeddings, metadata),
                           mode=mode)


def open_table(db_path, table_name: str = "pool"):
    import lancedb
    return lancedb.connect(str(db_path)).open_table(table_name)


def add_to_index(db_path, embeddings: np.ndarray, metadata: pd.DataFrame,
                 table_name: str = "pool"):
    """Append rows (e.g. a newly collected sample) to an existing table."""
    import lancedb
    tbl = lancedb.connect(str(db_path)).open_table(table_name)
    tbl.add(_vector_table(embeddings, metadata))
    return tbl


def query_by_example(table, seed_vectors: np.ndarray, k: int = 50,
                     metric: str = "cosine", aggregate: str = "each",
                     exclude_paths=None) -> pd.DataFrame:
    """Retrieve the pool ROIs most similar to a set of seed embeddings.

    Parameters
    ----------
    seed_vectors : np.ndarray, shape (n_seeds, dim)
    aggregate : {"each", "centroid"}
        "each": search every seed and merge, keeping each candidate's best
        (smallest) distance -- good recall across morphological variants.
        "centroid": mean-pool the seeds and run one search -- tighter, returns
        the prototypical center of the class.
    exclude_paths : set[str] | None
        Paths to drop from results (e.g. ROIs already in the training set).

    Returns a DataFrame sorted by distance (ascending), one row per candidate.
    """
    seeds = np.atleast_2d(np.asarray(seed_vectors, dtype=np.float32))
    if aggregate == "centroid":
        seeds = seeds.mean(axis=0, keepdims=True)

    frames = []
    for v in seeds:
        res = (table.search(v).metric(metric)
               .limit(k if aggregate == "centroid" else k * 2)
               .to_pandas())
        frames.append(res)
    out = pd.concat(frames, ignore_index=True)

    if exclude_paths:
        out = out[~out["path"].isin(set(exclude_paths))]
    # Each candidate keeps its single best distance across seeds.
    out = (out.sort_values("_distance")
           .drop_duplicates(subset="path", keep="first")
           .head(k).reset_index(drop=True))
    return out
