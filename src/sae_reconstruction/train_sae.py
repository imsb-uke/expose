from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping 
from lightning.pytorch.loggers import WandbLogger
import lightning as L
import torch
import os
import click
import datetime

from src.data_loading.datamodule_embeddings import DataModuleEmbeddings
from src.architecture.sparse_autoencoder import SAE
from src.sae_reconstruction.litmodule import LitModule



@click.command()
# general
@click.option("--seed", default=42, help="Seed for reproducibility")
@click.option("--use_wandb", type=bool, default=False, help="Use Wandb for logging")
@click.option("--wandb_project", default="sae_training", help="Wandb project name")

# data
@click.option("--metadata_path", default="data/metadata.csv", help="Path to the metadata CSV file")
@click.option("--embedding_dir", default="data/embeddings", help="Directory containing the embeddings")
@click.option("--domain_list",  default="domain1,domain2", help="List of domains to use")

# model
@click.option("--embedding_dim", default=768, help="Dimensionality of the input embeddings")
@click.option("--expansion_factor", default=3, help="Factor to expand the embedding dimension in the SAE")

# training
@click.option("--learning_rate", default=1e-4, help="Learning rate for training")
@click.option("--max_epochs", default=50, help="Maximum number of epochs for training")
@click.option("--patience", default=5, help="Patience for early stopping callback")
@click.option("--batch_size", default=32, help="Batch size for training")
@click.option("--batch_size_val", default=32, help="Batch size for validation")
@click.option("--num_workers", default=0, help="Number of workers for data loading")
@click.option("--pin_memory", type=bool, default=True, help="Whether to pin memory in DataLoader")
@click.option("--weight_decay", default=1e-2, help="Weight decay for the optimizer")
@click.option("--l1_weight", default=1e-3, help="Weight for L1 regularization in the loss function")

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
    )
    datamodule.setup(stage="fit")

    # Define path for model
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join("models_sae", current_time)
    os.makedirs(model_dir, exist_ok=True)

    # Define model
    model = SAE(
        d_in = config["embedding_dim"],
        d_sae = config["embedding_dim"] * config["expansion_factor"],
    )

    # Define litmodule
    litmodule = LitModule(
        model=model,
        config=config,
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(model_dir, "checkpoints"),
        filename="best_model",
        monitor="val_sae_metric_epoch", 
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
        logger.experiment.define_metric("val_loss_epoch", summary="min") 
        logger.experiment.define_metric("val_sae_metric_epoch", summary="max") 

        # Instantiate the trainer with Wandb logger, checkpoint callback, and other hyperparameters
        trainer = L.Trainer(
            accelerator="auto",  
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