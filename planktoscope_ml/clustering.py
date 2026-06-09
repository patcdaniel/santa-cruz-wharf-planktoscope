from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import umap
from PIL import Image


class ClusteringSession:
    """
    Stateful UMAP + HDBSCAN hierarchical clustering session.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame with an image path column (``'img_path'`` or ``'path'``).
    features : np.ndarray
        CNN feature matrix aligned with df rows.
    output_dir : str | Path
        Root directory for saved training-set images.
    """

    _PALETTE = plt.cm.tab20.colors

    def __init__(self, df: pd.DataFrame, features: np.ndarray,
                 output_dir: str | Path = 'data/training'):
        for col in ('img_path', 'path'):
            if col in df.columns:
                self._img_path_col = col
                break
        else:
            raise ValueError("DataFrame must contain an 'img_path' or 'path' column.")

        self._root_df = df.copy()
        self._root_features = features.copy()
        self.output_dir = Path(output_dir)

        # Current working state
        self._df = df.copy()
        self._features = features.copy()
        self._labels: np.ndarray | None = None
        self._embedding: np.ndarray | None = None
        self._path = 'root'

        # Navigation stack: each entry is the full state before drilling in
        self._stack: list[tuple] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def cluster(self,
                n_neighbors: int = 30,
                min_dist: float = 0.1,
                min_cluster_size: int = 30,
                min_samples: int = 5,
                umap_metric: str = 'cosine',
                cluster_selection_method: str = "eom") -> None:
        """Run UMAP → HDBSCAN on the current working set and plot results."""
        print(f'[{self._path}]  UMAP on {len(self._df):,} particles …')
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=umap_metric,
            random_state=42,
            verbose=False,
        )
        self._embedding = reducer.fit_transform(self._features)

        print(f'[{self._path}]  HDBSCAN …')
        cl = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric='euclidean',
        )
        self._labels = cl.fit_predict(self._embedding)
        self._df = self._df.copy()
        self._df['_cluster'] = self._labels

        n_cl = int((np.unique(self._labels) >= 0).sum())
        n_noise = int((self._labels == -1).sum())
        pct = 100 * n_noise / len(self._labels)
        print(f'[{self._path}]  {n_cl} clusters  |  {n_noise:,} noise ({pct:.1f}%)\n')
        self.plot()

    def recluster(self, cluster_id: int,
                  n_neighbors: int = 20,
                  min_dist: float = 0.05,
                  min_cluster_size: int = 10,
                  min_samples: int = 5,
                  cluster_selection_method: str = "eom",
                  umap_metric: str = 'cosine') -> None:
        """Drill into *cluster_id* and re-run UMAP + HDBSCAN on its particles."""
        if self._labels is None:
            print('Run .cluster() first.')
            return
        mask = self._labels == cluster_id
        if not mask.any():
            print(f'No particles found with cluster label {cluster_id}.')
            return

        # Save current state so we can go back
        self._stack.append((
            self._df.copy(),
            self._features.copy(),
            self._labels.copy(),
            self._embedding.copy(),
            self._path,
        ))

        self._df = self._df[mask].reset_index(drop=True)
        self._features = self._features[mask]
        self._path = f'{self._path} > C{cluster_id}'

        print(f'Entering cluster {cluster_id}  ({mask.sum():,} particles)')
        self.cluster(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            umap_metric=umap_metric,
            cluster_selection_method=cluster_selection_method,
        )

    def back(self) -> None:
        """Return to the parent clustering level."""
        if not self._stack:
            print('Already at root.')
            return
        self._df, self._features, self._labels, self._embedding, self._path = \
            self._stack.pop()
        print(f'Back to: {self._path}')
        self.plot()

    def reset(self) -> None:
        """Return all the way to the root and clear history."""
        self._df = self._root_df.copy()
        self._features = self._root_features.copy()
        self._labels = None
        self._embedding = None
        self._path = 'root'
        self._stack.clear()
        print('Session reset to root.')

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self) -> None:
        """UMAP scatter coloured by cluster label."""
        if self._embedding is None:
            print('Run .cluster() first.')
            return
        fig, ax = plt.subplots(figsize=(7, 6))
        for lbl in sorted(np.unique(self._labels)):
            m = self._labels == lbl
            if lbl == -1:
                ax.scatter(self._embedding[m, 0], self._embedding[m, 1],
                           c='lightgrey', s=3, alpha=0.3, linewidths=0,
                           rasterized=True, label='noise')
            else:
                col = [self._PALETTE[lbl % len(self._PALETTE)]]
                ax.scatter(self._embedding[m, 0], self._embedding[m, 1],
                           c=col, s=5, alpha=0.6, linewidths=0,
                           rasterized=True, label=f'C{lbl} (n={m.sum()})')
        ax.legend(markerscale=3, fontsize=7, framealpha=0.8,
                  loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Clusters — {self._path}')
        plt.tight_layout()
        plt.show()

    def show(self, n_per_cluster: int = 8, seed: int = 42) -> None:
        """Show a grid of sample images for every cluster at the current level."""
        if self._labels is None:
            print('Run .cluster() first.')
            return
        clusters = sorted(c for c in np.unique(self._labels) if c >= 0)
        if not clusters:
            print('No clusters to show (all noise?).')
            return
        rng = np.random.default_rng(seed)
        fig, axes = plt.subplots(
            len(clusters), n_per_cluster,
            figsize=(n_per_cluster * 1.8, len(clusters) * 1.8),
            squeeze=False,
        )
        for row_i, lbl in enumerate(clusters):
            grp = self._df[self._df['_cluster'] == lbl]
            idxs = rng.choice(len(grp), size=min(n_per_cluster, len(grp)), replace=False)
            for col_i in range(n_per_cluster):
                ax = axes[row_i, col_i]
                ax.axis('off')
                if col_i < len(idxs):
                    try:
                        img = Image.open(grp.iloc[idxs[col_i]][self._img_path_col]).convert('RGB')
                        ax.imshow(img)
                    except Exception:
                        pass
                if col_i == 0:
                    n = len(grp)
                    ax.text(0, 1, f'C{lbl}\nn={n}',
                            transform=ax.transAxes, fontsize=7,
                            va='top', ha='left',
                            bbox=dict(fc='white', ec='none', alpha=0.7))
        plt.suptitle(f'Cluster samples — {self._path}', fontsize=11)
        plt.tight_layout()
        plt.show()

    def status(self) -> None:
        """Print a summary table of clusters and current tree position."""
        print(f'\nPath : {self._path}')
        print(f'Depth: {len(self._stack)}  (0 = root)')
        if self._labels is None:
            print('No clustering yet. Run .cluster().')
            return
        print(f'N    : {len(self._df):,}\n')
        rows = []
        for lbl in sorted(np.unique(self._labels)):
            mask = self._labels == lbl
            rows.append({
                'Cluster': 'noise' if lbl == -1 else f'C{lbl}',
                'N': int(mask.sum()),
                'pct': f'{100 * mask.sum() / len(self._labels):.1f}%',
            })
        print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_cluster(self, cluster_id: int, label: str,
                     review: bool = False) -> Path | None:
        """
        Copy all images from *cluster_id* to ``output_dir/label/``.

        Parameters
        ----------
        cluster_id : int
            Cluster label to export (must be >= 0).
        label : str
            Class name used as the subdirectory (e.g. ``'Chaetoceros'``).
        review : bool
            If True, save under ``output_dir/label/review/`` for manual
            inspection before committing to the training set.
        """
        if self._labels is None:
            print('Run .cluster() first.')
            return None
        mask = self._labels == cluster_id
        if not mask.any():
            print(f'Cluster {cluster_id} not found.')
            return None

        out_dir = self.output_dir / label / ('review' if review else '')
        out_dir.mkdir(parents=True, exist_ok=True)

        grp = self._df[mask]
        for src in grp[self._img_path_col]:
            copy2(src, out_dir / Path(src).name)

        print(f'Saved {mask.sum():,} images → {out_dir}')
        return out_dir
