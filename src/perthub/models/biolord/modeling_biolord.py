from sklearn.cluster import MeanShift
import torch
from torch import nn
from typing import List
from dataclasses import dataclass
from torch.distributions import Normal
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .configuration_biolord import BiolordConfig
from transformers.modeling_outputs import ModelOutput


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    bias: bool = True,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU(),
    add_layernorm: bool = False,
    add_batchnorm: bool = False,
    final_linear_only: bool = True,
) -> nn.Sequential:
    """
    Build an MLP with `n_layers` layers from `in_dim` to `out_dim`.

    If `final_linear_only=True`, the last layer is strictly:
        Linear
    (no normalization, no activation, no dropout)
    """

    if add_layernorm and add_batchnorm:
        raise ValueError("Cannot use both LayerNorm and BatchNorm.")
    
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    def make_block(
        in_d: int,
        out_d: int,
        use_norm: bool,
        use_act: bool,
    ) -> List[nn.Module]:
        layers = [nn.Linear(in_d, out_d, bias=bias)]

        if use_norm:
            if add_layernorm:
                layers.append(nn.LayerNorm(out_d))
            elif add_batchnorm:
                layers.append(nn.BatchNorm1d(out_d))

        if use_act:
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        return layers

    layers: List[nn.Module] = []

    # ===== single layer =====
    if n_layers == 1:
        use_extra = not final_linear_only
        layers += make_block(in_dim, out_dim, use_norm=use_extra, use_act=use_extra)

    # ===== multiple layers =====
    else:
        # first layer
        layers += make_block(
            in_dim,
            hidden_dim,
            use_norm=True,
            use_act=True,
        )

        # middle layers
        for _ in range(n_layers - 2):
            layers += make_block(
                hidden_dim,
                hidden_dim,
                use_norm=True,
                use_act=True,
            )

        # final layer
        use_extra = not final_linear_only
        layers += make_block(
            hidden_dim,
            out_dim,
            use_norm=use_extra,   # no norm if final_linear_only
            use_act=use_extra,    # no act/dropout if final_linear_only
        )

    return nn.Sequential(*layers)


class RegularizedEmbedding(nn.Module):
    """Regularized embedding module."""

    def __init__(self, config: BiolordConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.n_samples,
            embedding_dim=config.n_latent,
        )
        self.embed = config.unknown_attributes
        self.sigma = config.unknown_attribute_noise_param if self.embed else 0

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass."""
        x_ = self.embedding(x)
        if self.training and self.sigma != 0:
            noise = torch.zeros_like(x_)
            noise.normal_(mean=0, std=self.sigma)

            x_ = x_ + noise
        x_ = x_ * self.embed
        return x_


class BiolordPreTrainedModel(PreTrainedModel):
    config_class = BiolordConfig
    base_model_prefix = "biolord"

    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()


@dataclass
class BiolordModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    gaussian_nll_loss: torch.FloatTensor = None
    mse_loss: torch.FloatTensor = None
    reconstruction_loss: torch.FloatTensor = None
    unknown_attribute_penalty_loss: torch.FloatTensor = None
    means: torch.FloatTensor = None
    variances: torch.FloatTensor = None
    samples: torch.FloatTensor = None


class BiolordModel(BiolordPreTrainedModel):
    def __init__(
        self, 
        config: BiolordConfig,
        ordered_attributes_map: dict,  # {"rdkit2d_dose": 174, ...}; (rdkit2d: 173 + dose: 1 = 174)
        categorical_attributes_map: dict,  # {"cell_type": {"A549": 0, "K562": 1, "MCF7": 2}, ...}
        alpha_mse_loss: float = 10000,
        unknown_attribute_penalty: float = 0.1
    ):
        super().__init__(config)
        self.ordered_attributes_map = ordered_attributes_map
        self.categorical_attributes_map = categorical_attributes_map

        # Determine `attribute_nn_depth` for each ordered attribute
        self.attribute_nn_depth = self._normalize_ordered_attributes_nn_param(config.attribute_nn_depth)
        # Determine `attribute_nn_width` for each ordered attribute
        self.attribute_nn_width = self._normalize_ordered_attributes_nn_param(config.attribute_nn_width)
        # Determine `attribute_dropout_rate` for each ordered attribute
        self.attribute_dropout_rate = self._normalize_ordered_attributes_nn_param(config.attribute_dropout_rate)

        # 1. Ordered attributes encoder
        self.ordered_networks = nn.ModuleDict()
        for attribute_, len_ in self.ordered_attributes_map.items():
            self.ordered_networks[attribute_] = build_mlp(
                in_dim=len_,
                out_dim=self.n_latent_attribute_ordered,
                hidden_dim=self.attribute_nn_width[attribute_],
                n_layers=self.attribute_nn_depth[attribute_],
                bias=False,
                dropout=self.attribute_dropout_rate[attribute_],
                activation=ACT2FN[config.attribute_nn_activation],
                add_layernorm=False,
                add_batchnorm=True,
                final_linear_only=False,
            )

        # 2. Categorical attributes encoder
        self.categorical_embeddings = nn.ModuleDict()
        for attribute_, unique_categories in self.categorical_attributes_map.items():
            self.categorical_embeddings[attribute_] = nn.Embedding(
                len(unique_categories),
                config.n_latent_attribute_categorical,
            )

        # 3. Unknown attributes encoder
        self.latent_codes = RegularizedEmbedding(config)


        # Decoder: Use Gaussian NLL Decoder
        self.decoder = build_mlp(
            in_dim=self._compute_decoder_input_size(config),
            out_dim=config.n_genes * 2,  # mean and variance
            hidden_dim=config.decoder_width,
            n_layers=config.decoder_depth + 1,
            bias=True,
            dropout=config.decoder_dropout_rate,
            activation=ACT2FN[config.decoder_activation],
            add_layernorm=False,
            add_batchnorm=False,
            final_linear_only=False,
        )

        # Loss Functions
        self.alpha_mse_loss = alpha_mse_loss
        self.unknown_attribute_penalty = unknown_attribute_penalty
        self.gaussian_nll_loss_fn = nn.GaussianNLLLoss()
        self.mse_loss_fn = nn.MSELoss()
        
        # Initialize weights and apply final processing
        self.post_init()    
    
    def _normalize_ordered_attributes_nn_param(self, param: dict | int) -> dict:
        """If `param` is an int, convert it to a dict with the same value for all ordered attributes."""
        return (
            param
            if isinstance(param, dict)
            else {attribute_: param for attribute_ in self.ordered_attributes_map}
        )

    def _compute_decoder_input_size(self, config: BiolordConfig) -> int:
        return config.n_latent + (
            config.n_latent_attribute_categorical * len(self.categorical_attributes_map)
            + config.n_latent_attribute_ordered * len(self.ordered_attributes_map)
        )

    def forward(
        self,
        x: torch.Tensor = None,
        sample_indices: torch.Tensor = None,
        cell_type: torch.Tensor = None,  # name consistent with categorical_attributes_map
        rdkit2d_dose: torch.Tensor = None  # name consistent with ordered_attributes_map
    ):
        input_kwargs = locals()

        # {"cell_type": torch.tensor([...])}
        categorical_attribute_dict = {}
        for attribute_ in self.categorical_attributes_map:
            categorical_attribute_dict[attribute_] = input_kwargs[attribute_].view(-1)

        # {"rdkit2d_dose": torch.tensor([...])}
        ordered_attribute_dict = {}
        for attribute_ in self.ordered_attributes_map:
            ordered_attribute_dict[attribute_] = input_kwargs[attribute_]
        
        return self._forward(
            x=x,
            sample_indices=sample_indices,
            categorical_attribute_dict=categorical_attribute_dict,
            ordered_attribute_dict=ordered_attribute_dict
        )

    def _forward(
        self,
        x: torch.Tensor = None,
        sample_indices: torch.Tensor = None,
        categorical_attribute_dict: dict = None,
        ordered_attribute_dict: dict = None
    ):
        # 1. encode unknown attributes
        latent_unknown_attributes: torch.Tensor = self._get_latent_unknown_attributes(sample_indices)

        # 2. encode known attributes (categorical + ordered)
        latent_known_attributes: dict = self._get_latent_known_attributes(
            categorical_attribute_dict, 
            ordered_attribute_dict
        )

        # concatenate all latent representations
        latent_vecs = [latent_unknown_attributes.squeeze()]
        for key_, latent_ in latent_known_attributes.items():
            latent_vecs.append(latent_)
        latent = torch.cat(latent_vecs, dim=-1)  # latent and latent_unknown_attributes are useful.

        # 3. decode to get distribution parameters
        decoder_output: dict = self._get_decoder_output(latent)

        # 4. compute losses
        losses: dict[str, torch.Tensor] = self.compute_loss(
            x=x, 
            latent_unknown_attributes=latent_unknown_attributes, 
            decoder_output=decoder_output
        )

        return BiolordModelOutput(
            loss=losses["loss"],
            gaussian_nll_loss=losses["gaussian_nll_loss"],
            mse_loss=losses["mse_loss"],
            reconstruction_loss=losses["reconstruction_loss"],
            unknown_attribute_penalty_loss=losses["unknown_attribute_penalty_loss"],
            means=decoder_output["means"],
            variances=decoder_output["variances"],
            samples=decoder_output["samples"]
        )
    
    def _get_latent_unknown_attributes(
        self, 
        sample_indices: torch.Tensor
    ) -> torch.Tensor:
        """Get the biolord's latent unknown attributes representation."""
        latent_unknown_attributes = self.latent_codes(sample_indices)

        return latent_unknown_attributes
    
    def _get_latent_known_attributes(
        self,
        categorical_attribute_dict: dict,
        ordered_attribute_dict: dict
    ) -> torch.Tensor:
        """Get the biolord's latent attribute embeddings. (categorical + ordered)"""
        attr2emb = {}  # {attribute_name: latent_embedding}

        # Get latent embeddings for categorical attributes
        for attribute_, embedding_ in self.categorical_embeddings.items():
            latent_i = embedding_(categorical_attribute_dict[attribute_].long())
            attr2emb[attribute_] = latent_i

        # Get latent embeddings for ordered attributes
        for attribute_, network_ in self.ordered_networks.items():
            latent_i = network_(ordered_attribute_dict[attribute_])
            attr2emb[attribute_] = latent_i

        return attr2emb
    
    def _get_decoder_output(
        self,
        concatenated_latent: torch.Tensor
    ) -> dict:
        mean, logvar = torch.chunk(
            self.decoder(concatenated_latent), chunks=2, dim=-1
        )
        var = torch.exp(logvar)
        px = Normal(loc=mean, scale=var.sqrt())

        return {
            "means": px.loc,
            "variances": px.variance,
            "distribution": px,
            "samples": px.sample(),
        }
    
    def unknown_attribute_penalty_loss(
        self, 
        latent_unknown_attributes: torch.Tensor
    ) -> torch.Tensor:
        """Computes the content penalty term in the loss."""
        return torch.sum(latent_unknown_attributes**2, dim=1).mean()

    def compute_loss(
        self,
        x: torch.Tensor,
        latent_unknown_attributes: torch.Tensor,
        decoder_output: dict
    ):
        means = decoder_output["means"]
        variances = decoder_output["variances"]
        gaussian_nll_loss = self.gaussian_nll_loss_fn(
            input=means, target=x, var=variances
        )
        mse_loss = self.mse_loss_fn(input=means, target=x)
        reconstruction_loss = gaussian_nll_loss + self.alpha_mse_loss * mse_loss
        unknown_attribute_penalty_loss = self.unknown_attribute_penalty_loss(
            latent_unknown_attributes=latent_unknown_attributes
        )
        # total loss
        loss = reconstruction_loss + self.unknown_attribute_penalty * unknown_attribute_penalty_loss
        return {
            "loss": loss,
            "gaussian_nll_loss": gaussian_nll_loss,
            "mse_loss": mse_loss,
            "reconstruction_loss": reconstruction_loss,
            "unknown_attribute_penalty_loss": unknown_attribute_penalty_loss
        }
        