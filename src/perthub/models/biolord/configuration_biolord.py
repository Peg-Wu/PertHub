from transformers import PreTrainedConfig

"""Cell Meta Information  -->  Cell Expression Profile"""

# TrainingArguments (loss-related):
    # reconstruction_penalty: 10000
    # unknown_attribute_penalty: 0.1
    # total_loss = 
    #   \ gaussian_nll_loss + reconstruction_penalty * mse_loss + \
    #   \ unknown_attribute_penalty * unknown_attribute_penalty_loss

class BiolordConfig(PreTrainedConfig):
    model_type = "biolord"
    def __init__(
        self,
        # NOTE: Unknown attributes related parameters
        ## total number of samples (train + val + test).
        ## sample_idx -> nn.Embedding -> unknown_attribute_embedding.
        n_samples: int = 354640,  
        ## latent size (final hidden_size).
        n_latent: int = 256,
        ## whether to include learning for unknown attributes.
        unknown_attributes: bool = True,
        ## noise strength added to encoding of unknown attributes. 
        ## sigma of the normal distribution.
        unknown_attribute_noise_param: float = 20,
        # NOTE: Categorical attributes related parameters
        ## latent size (final hidden_size).
        n_latent_attribute_categorical: int = 3,
        # NOTE: Ordered attributes related parameters
        ## latent size (final hidden_size).
        n_latent_attribute_ordered: int = 256,
        ## number of layers.
        ## all ordered attributes are concatenated and passed through the encoder.
        attribute_nn_depth: dict | int = 2,
        ## width of each layer. (middle hidden_size)
        attribute_nn_width: dict | int = 2048,
        ## dropout rate.
        attribute_dropout_rate: dict | float = 0.1,
        ## activation function.
        attribute_nn_activation: str = "relu",
        ## whether to use bias.
        attribute_nn_bias: bool = False,
        # NOTE: Decoder related parameters
        ## number of genes to output. (final output size)
        n_genes: int = 2000,
        ## number of layers.
        decoder_depth: int = 4,
        ## width of each layer. (middle hidden_size)
        decoder_width: int = 4096,
        ## dropout rate.
        decoder_dropout_rate: float = 0,
        ## activation function.
        decoder_activation: str = "relu",
        ## whether to use bias.
        decoder_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.n_latent = n_latent
        self.unknown_attributes = unknown_attributes
        self.unknown_attribute_noise_param = unknown_attribute_noise_param
        self.n_latent_attribute_categorical = n_latent_attribute_categorical
        self.n_latent_attribute_ordered = n_latent_attribute_ordered
        self.attribute_nn_depth = attribute_nn_depth
        self.attribute_nn_width = attribute_nn_width
        self.attribute_nn_activation = attribute_nn_activation
        self.attribute_nn_bias = attribute_nn_bias
        self.attribute_dropout_rate = attribute_dropout_rate
        self.n_genes = n_genes
        self.decoder_depth = decoder_depth
        self.decoder_width = decoder_width
        self.decoder_dropout_rate = decoder_dropout_rate
        self.decoder_activation = decoder_activation
        self.decoder_bias = decoder_bias