import logging
import scanpy as sc
from tqdm.auto import tqdm
from datasets import Dataset, concatenate_datasets

logger = logging.getLogger(__name__)


def tokenize_adata_to_hf_dataset(
    adata: sc.AnnData,
    attr_map_dict: dict[str, str],
    x_key: str = "x",
    chunk_size: int = 20000,
) -> Dataset:
    """Convert AnnData to HuggingFace Dataset in chunks.

    Args:
        adata: AnnData object.
        attr_map_dict: Mapping from adata.obs / adata.obsm keys to HF dataset column names.
        x_key: Column name for adata.X in the output dataset.
        chunk_size: Number of observations per chunk.
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    all_chunks: list[Dataset] = []
    n_obs = adata.n_obs
    for i in tqdm(range(0, n_obs, chunk_size), desc="Tokenizing to HF Dataset"):
        chunk_slice = slice(i, min(i + chunk_size, n_obs))

        chunk_dict: dict[str, list] = {}
        chunk_dict[x_key] = X[chunk_slice]

        for src_key, dst_key in attr_map_dict.items():
            in_obs = src_key in adata.obs
            in_obsm = src_key in adata.obsm
            if in_obs and in_obsm:
                raise KeyError(f"{src_key} found in both adata.obs and adata.obsm, ambiguous mapping")
            elif in_obs:
                chunk_dict[dst_key] = adata.obs[src_key].iloc[chunk_slice].tolist()
            elif in_obsm:
                chunk_dict[dst_key] = adata.obsm[src_key][chunk_slice].tolist()
            else:
                raise KeyError(f"{src_key} not found in adata.obs or adata.obsm")

        all_chunks.append(Dataset.from_dict(chunk_dict))

    return concatenate_datasets(all_chunks)

