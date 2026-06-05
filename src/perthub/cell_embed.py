from __future__ import annotations

import os
import torch
import logging
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm.auto import tqdm
from typing import Any, Union
from scipy.sparse import issparse, csr_matrix
from wppkg import hf_download, guess_is_lognorm, get_sorted_indices_in_array_2d_by_row

logger = logging.getLogger(__name__)


class ArcStackEmbeddingExtractor:
    """Extract gene expression embeddings using the Arc Institute STACK model.

    Downloads the model checkpoint from HuggingFace on first use and caches it locally.

    .. attention::

       **Input data expectations (enforced by STACK internally):**

       - If ``adata.raw`` exists, STACK reads from ``adata.raw.X``; otherwise falls back to ``adata.X``.
       - STACK only applies ``log1p`` to the data — **no normalization** is performed.
       - STACK maps genes to its fixed vocabulary of **15,012 genes**:
         matched genes retain expression values, unmatched vocabulary genes are zero-filled,
         and extra genes in the input are silently dropped.

    Parameters
    ----------
    cache_dir : str or None
        Directory to cache the downloaded model. Defaults to ``~/.cache/nextcell/stack-large``.
    device : str or torch.device or None
        Device to run inference on. Defaults to ``"cuda:0"`` if available, else ``"cpu"``.
    use_mirror : bool
        Whether to download via ``https://hf-mirror.com``. Defaults to ``True``.
    """

    REPO_ID: str = "arcinstitute/Stack-Large"
    CKPT_FILENAME: str = "bc_large.ckpt"
    GENELIST_FILENAME: str = "basecount_1000per_15000max.pkl"

    def __init__(
        self,
        cache_dir: str | None = None,
        device: str | torch.device | None = None,
        use_mirror: bool = False,
    ) -> None:
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/nextcell/stack-large")
        self.cache_dir = cache_dir

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.use_mirror = use_mirror
        self._model = None

    def _ensure_downloaded(self) -> tuple[str, str]:
        """Download model files if not already cached."""
        ckpt_path = os.path.join(self.cache_dir, self.CKPT_FILENAME)
        genelist_path = os.path.join(self.cache_dir, self.GENELIST_FILENAME)

        if os.path.exists(ckpt_path) and os.path.exists(genelist_path):
            return ckpt_path, genelist_path

        os.makedirs(self.cache_dir, exist_ok=True)
        hf_download(
            repo_id=self.REPO_ID,
            repo_type="model",
            local_dir=self.cache_dir,
            endpoint="https://hf-mirror.com" if self.use_mirror else None,
        )
        return ckpt_path, genelist_path

    @property
    def model(self) -> Any:
        """Lazy-loaded STACK model."""
        if self._model is None:
            ckpt_path, _ = self._ensure_downloaded()
            from stack.model_loading import load_model_from_checkpoint

            self._model = load_model_from_checkpoint(
                checkpoint_path=ckpt_path,
                device=self.device,
            )
        return self._model

    def extract(
        self,
        adata_path: str,
        gene_name_col: str | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> np.ndarray:
        """Extract STACK embeddings from an AnnData file.

        Parameters
        ----------
        adata_path : str
            Path to the ``.h5ad`` file.
        gene_name_col : str or None
            Column in ``.var`` to use as gene names. If None, uses ``.var.index``.
        batch_size : int
            Batch size for inference.
        num_workers : int
            Number of dataloader workers.

        Returns
        -------
        embeddings : np.ndarray
            Cell-by-embedding matrix with shape ``(n_cells, n_features)``.

        Notes
        -----
        For more advanced use, consider splitting the data by sample-level columns
        (e.g. donor, condition), feeding each split to the model separately, then
        concatenating the results.
        """
        logger.info(
            "STACK will: "
            "(1) read from adata.raw.X if present, else adata.X; "
            "(2) apply log1p only (no normalization); "
            "(3) map genes to its 15,012-gene vocabulary."
        )
        _, genelist_path = self._ensure_downloaded()

        embeddings, _ = self.model.get_latent_representation(
            adata_path=adata_path,
            genelist_path=genelist_path,
            gene_name_col=gene_name_col,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return embeddings

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Alias for :meth:`extract`."""
        return self.extract(*args, **kwargs)


def reverse_adata_to_raw_counts(
    adata: Union[str, ad.AnnData],
    int_tol: float = 1e-3,
    return_scaling_factors: bool = False
) -> Union[ad.AnnData, tuple[ad.AnnData, list[float]]]:
    if isinstance(adata, str):
        logger.info(f"Reading adata from {adata} ...")
        adata = sc.read_h5ad(adata)
    else:
        adata = adata.copy()

    if not guess_is_lognorm(adata):
        raise ValueError("Input adata is likely in raw (non-log1p) scale.")
    
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X = np.expm1(X)  # log1p reverse
    sorted_indices = get_sorted_indices_in_array_2d_by_row(
        arr=X,
        ignore_zero_values=True,
        descending=False,
        stable=None,
        n_jobs=1,
        enable_tqdm=False
    )
    logger.info("Start reversing back to raw counts.")
    res, scaling_factors = [], []
    for cell_idx in tqdm(range(len(adata))):
        if len(sorted_indices[cell_idx]) == 0:  # all zero values in a cell
            res.append(np.zeros(X.shape[1], dtype=np.float32))
            scaling_factors.append(0.0)
            continue
        reverse_success = False
        min_val = X[cell_idx, sorted_indices[cell_idx][0]]
        for assume_min_value in range(1, 101):
            coef = min_val / assume_min_value
            reversed_x = X[cell_idx] / coef
            round_reversed_X = np.round(reversed_x)
            is_near_integer = np.isclose(reversed_x, round_reversed_X, atol=int_tol)
            if np.all(is_near_integer):
                res.append(round_reversed_X)
                scaling_factors.append(coef)
                reverse_success = True
                break
        if not reverse_success:
            raise ValueError(
                f"Failed to reverse raw counts for cell {cell_idx}."
            )
    logger.info("Successfully reversed back to raw counts.")
    reversed_raw_counts_X = np.vstack(res).astype(np.float32)
    adata.X = csr_matrix(reversed_raw_counts_X)

    if return_scaling_factors:
        return adata, scaling_factors
    else:
        return adata