from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping 
from lightning.pytorch.loggers import WandbLogger
import lightning as L
import torch
import os
import click
import datetime

from src.data_loading.datamodule_embeddings import DataModuleEmbeddings
from src.architecture.decoder_sae_linear import DecoderSAELinear
from src.classification.litmodule import LitModule
from src.utils.data import compute_pos_weight_from_dataset



@click.command()
# general
@click.option("--seed", default=42, help="Seed for reproducibility")
@click.option("--use_wandb", type=bool, default=False, help="Use Wandb for logging")
@click.option("--wandb_project", default="domain_training", help="Wandb project name")

# data
@click.option("--metadata_path", default="data/metadata.csv", help="Path to the metadata CSV file")
@click.option("--embedding_dir", default="data/embeddings", help="Directory containing the embeddings")
@click.option("--domain_list", default="domain1,domain2", help="List of domains to use")
@click.option("--label_col", default="domain", help="Label column for training")

# model
@click.option("--embedding_dim", default=768, help="Dimensionality of the input embeddings")
@click.option("--expansion_factor", default=5, help="Factor to expand the embedding dimension in the SAE")
@click.option("--num_classes", default=1, help="Number of classes for classification")
@click.option("--sae_ckpt_path", default="models_sae/2026-02-23_12-04-04/checkpoints/best_model.ckpt", help="Path to pre-trained SAE checkpoint")

# training
@click.option("--learning_rate", default=1e-4, help="Learning rate for training")
@click.option("--max_epochs", default=50, help="Maximum number of epochs for training")
@click.option("--patience", default=5, help="Patience for early stopping callback")
@click.option("--batch_size", default=32, help="Batch size for training")
@click.option("--batch_size_val", default=32, help="Batch size for validation")
@click.option("--num_workers", default=0, help="Number of workers for data loading")
@click.option("--pin_memory", type=bool, default=True, help="Whether to pin memory in DataLoader")
@click.option("--weight_decay", default=1e-2, help="Weight decay for the optimizer")

def main(**config): 
    L.seed_everything(config["seed"], workers=True)

    # Define data module
    datamodule = DataModuleEmbeddings(
        data_dir = config["embedding_dir"],
        metadata_path = config["metadata_path"],
        domain_list = config["domain_list"],
        num_workers = config["num_workers"],
        batch_size = config["batch_size"],
        batch_size_val = config["batch_size_val"],
        label_col = config["label_col"]
    )
    datamodule.setup(stage="fit")
    # calculate pos_weight for BCE
    pos_weight = compute_pos_weight_from_dataset(datamodule.train_dataset)

    # Define path for model
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join("models_domain", current_time)
    os.makedirs(model_dir, exist_ok=True)

    # Define model
    model = DecoderSAELinear(
        d_in = config["embedding_dim"],
        d_sae = config["embedding_dim"] * config["expansion_factor"],
        num_classes = config["num_classes"],
        sae_ckpt_path = config["sae_ckpt_path"]
    )

    # Define litmodule
    litmodule = LitModule(
        model=model,
        config=config,
        pos_weight=pos_weight
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(model_dir, "checkpoints"),
        filename="best_model",
        monitor="val_auroc", 
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=config["patience"])
    callbacks=[early_stopping_callback, checkpoint_callback]

    # Define Wandblogger
    if config["use_wandb"]:
        logger = WandbLogger(
            project=config["wandb_project"],
            save_dir=model_dir,
            config=config,
        )        
        logger.experiment.define_metric("val_auroc", summary="max")
        logger.experiment.define_metric("val_acc", summary="max")

        # Instantiate the trainer with Wandb logger, checkpoint callback, and other hyperparameters
        trainer = L.Trainer(
            accelerator="auto", 
            devices="auto", 
            max_epochs=config["max_epochs"], 
            callbacks=callbacks,
            logger=logger,
            num_sanity_val_steps=2,
            log_every_n_steps=2,
        )
    else:
        # Instantiate the trainer with Wandb logger, checkpoint callback, and other hyperparameters
        trainer = L.Trainer(
            accelerator="auto", 
            max_epochs=config["max_epochs"], 
            callbacks=callbacks,
            num_sanity_val_steps=2,
            log_every_n_steps=2,
        )
    

    # Fit the model
    trainer.fit(model=litmodule, datamodule=datamodule)

    ckpt_path = checkpoint_callback.best_model_path
    print(f"Saved best model weights to {ckpt_path}")


if __name__ == "__main__":    
    main()