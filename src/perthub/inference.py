import torch
import logging
import inspect
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from datasets import Dataset
from torchmetrics import R2Score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_r2(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor
) -> float:
    """
    Computes the r2 score for `y_true` and `y_pred`,
    returns `-1` when `y_pred` contains nan values
    """
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    metric = R2Score().to(y_true.device)
    metric.update(y_pred, y_true)  # same as sklearn.metrics.r2_score(y_true, y_pred)
    return metric.compute().item()
    

@torch.no_grad()
def biolord_inference(
    model: nn.Module,
    test_ds: Dataset,
    batch_size: int = 512,
    test_dl_num_workers: int = 4,
    test_dl_pin_memory: bool = True,
    device: torch.device = torch.device("cuda:0")
):
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=test_dl_num_workers,
        pin_memory=test_dl_pin_memory
    )

    model = model.to(device)
    model.eval()
    # model_forward_keys = list(inspect.signature(model.forward).parameters.keys())
    predictions = []
    for batch in tqdm(test_dl, desc="inference"):
        # filtered_batch = {k: v.to(device) for k, v in batch.items() if k in model_forward_keys}
        filtered_batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**filtered_batch)
        predictions.append(outputs.means.detach().cpu().numpy())

    return np.concatenate(predictions, axis=0)
