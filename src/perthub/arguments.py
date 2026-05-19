from typing import Optional
from wppkg import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class BiolordDataArguments:
    train_ds: str = field(
        default="../../data/ds_train",
        metadata={
            "help": "Path to the tokenized HuggingFace training dataset."
        }
    )

    valid_ds: Optional[str] = field(
        default="../../data/ds_valid",
        metadata={
            "help": "Path to the tokenized HuggingFace validation dataset."
        }
    )

    attributes_map: str = field(
        default="../../data/attributes_map.json",
        metadata={
            "help": (
                "Path to the JSON file containing categorical and ordered attributes mapping" 
                "`categorical_attributes_map` and `ordered_attributes_map`."
            )
        }
    )


@dataclass
class BiolordTrainingArguments(TrainingArguments):
    alpha_mse_loss: float = field(
        default=10000,
        metadata={
            "help": (
                "Weighting coefficient for the MSE term in the reconstruction loss. "
                "Total reconstruction loss = gaussian_nll_loss + alpha_mse_loss * mse_loss."
            )
        }
    )

    unknown_attribute_penalty: float = field(
        default=0.1,
        metadata={
            "help": (
                "L2 regularization coefficient for the unknown attribute embeddings. "
                "Penalizes the squared L2 norm of unknown attribute latent vectors to "
                "encourage explaining variation through known attributes."
            )
        }
    )