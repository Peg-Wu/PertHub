import datasets
from wppkg import read_json
from datasets import load_from_disk
from perthub.trainer import BiolordTrainer
from perthub.logging import set_verbosity_warning
from transformers import HfArgumentParser, set_seed
from perthub.models import BiolordConfig, BiolordModel
from perthub.arguments import BiolordDataArguments, BiolordTrainingArguments

def main():
    parser = HfArgumentParser((BiolordDataArguments, BiolordTrainingArguments))
    data_args, train_args = parser.parse_args_into_dataclasses()
    data_args: BiolordDataArguments
    train_args: BiolordTrainingArguments

    # seed everything
    set_seed(train_args.seed)

    # suppress perthub's verbose logging to reduce console output
    set_verbosity_warning()
    datasets.logging.set_verbosity_warning()

    # create dataset (train and valid)
    train_ds = load_from_disk(data_args.train_ds)
    valid_ds = load_from_disk(data_args.valid_ds) if data_args.valid_ds else None

    # create model
    config = BiolordConfig(
        n_samples=read_json(data_args.attributes_map)["n_samples"],
        n_genes=len(train_ds[0]["x"])
    )
    model = BiolordModel(
        config=config,
        ordered_attributes_map=read_json(data_args.attributes_map)["ordered_attributes_map"],
        categorical_attributes_map=read_json(data_args.attributes_map)["categorical_attributes_map"],
        alpha_mse_loss=train_args.alpha_mse_loss,
        unknown_attribute_penalty=train_args.unknown_attribute_penalty
    )

    # train
    trainer = BiolordTrainer(
        args=train_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds
    )

    trainer.train()


if __name__ == "__main__":
    main()