import logging
import chemprop
import numpy as np
import pandas as pd
import scanpy as sc
from rdkit import Chem
from typing import overload
from rdkit.Chem import CanonSmiles

logger = logging.getLogger(__name__)


def check_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid and chemically sound."""
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        logger.info('invalid SMILES')
        return False
    else:
        try:
            Chem.SanitizeMol(m)
        except:
            logger.info('invalid chemistry')
            return False
    return True


@overload
def remove_invalid_smiles(
    dataframe: pd.DataFrame, 
    smiles_key: str = 'SMILES', 
    return_condition: bool = False
) -> pd.DataFrame: ...


@overload
def remove_invalid_smiles(
    dataframe: pd.DataFrame, 
    smiles_key: str = 'SMILES', 
    return_condition: bool = True
) -> pd.Series: ...


def remove_invalid_smiles(
    dataframe: pd.DataFrame, 
    smiles_key: str = 'SMILES', 
    return_condition: bool = False
) -> pd.DataFrame | pd.Series:
    r"""
    - return_condition=False: 
        Remove rows with invalid SMILES strings from the dataframe.
    - return_condition=True: 
        Return a boolean Series indicating which rows have valid SMILES strings.
    """
    unique_drugs = pd.Series(np.unique(dataframe[smiles_key]))
    valid_drugs = unique_drugs.apply(check_smiles)
    logger.info(f"A total of {(~valid_drugs).sum()} have invalid SMILES strings")
    _validation_map = dict(zip(unique_drugs, valid_drugs))
    cond = dataframe[smiles_key].apply(lambda x: _validation_map[x])
    if return_condition: 
        return cond
    dataframe = dataframe[cond].copy()
    return dataframe


def unify_smiles(
    dataframe: pd.DataFrame, 
    smiles_key: str = 'SMILES'
) -> pd.DataFrame:
    """Unify SMILES strings in the dataframe."""
    dataframe[smiles_key] = dataframe[smiles_key].apply(CanonSmiles)
    return dataframe


def compute_rdkit_embeddings(
    adata: sc.AnnData,
    smiles_key: str = 'SMILES',
    skip_variance_filter: bool = False,
    apply_z_score: bool = True
) -> None:
    if hasattr(chemprop, 'featurizers'):
        # chemprop v2.x
        from chemprop.featurizers import V1RDKit2DNormalizedFeaturizer
        _featurizer = V1RDKit2DNormalizedFeaturizer()
        features = {}
        for smi in adata.obs[smiles_key].unique():
            mol = Chem.MolFromSmiles(smi)
            features[smi] = _featurizer(mol)
    else:
        # chemprop v1.x
        features = {}
        for smi in adata.obs[smiles_key].unique():
            features[smi] = chemprop.features.features_generators.rdkit_2d_normalized_features_generator(smi)

    features_df = pd.DataFrame.from_dict(features).T
    features_df = features_df.fillna(0)

    if not skip_variance_filter:
        threshold = 0.001
        features_df = features_df.iloc[:, np.where(features_df.std() > threshold)[0]]

    normalized_df = (
        (features_df - features_df.mean()) / features_df.std() 
        if apply_z_score 
        else features_df
    )

    features_cells = np.zeros((adata.shape[0], normalized_df.shape[1]))
    for mol, rdkit_2d in normalized_df.iterrows():
        features_cells[adata.obs["SMILES"].isin([mol])] = rdkit_2d.values

    adata.obsm["rdkit2d"] = features_cells