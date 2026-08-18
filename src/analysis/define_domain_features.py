from pathlib import Path

import click
import torch
import yaml

from src.architecture.decoder_sae_linear import DecoderSAELinear


@click.command()
@click.option("--embedding_dim", default=768, help="Dimensionality of the input embeddings")
@click.option("--expansion_factor", default=5, help="Factor to expand the embedding dimension in the SAE")
@click.option("--num_classes", default=1, help="Number of classes for classification")
@click.option("--domain_classifier_ckpt_path", default="models_domain/2026-08-18_11-49-21/checkpoints/best_model.ckpt", help="Path to pre-trained domain classifier checkpoint")
@click.option("--output_path", default="data/domain_weights.pt", help="Path to save the domain weights")

def main(**config):
    model = DecoderSAELinear(
        d_in=config["embedding_dim"],
        d_sae=config["embedding_dim"] * config["expansion_factor"],
        num_classes=config["num_classes"],
        sae_ckpt_path=None,
    )

    ckpt = torch.load(config["domain_classifier_ckpt_path"], map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cleaned_state_dict = {
        k.replace("model.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {missing}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {unexpected}")

    weights = model.clas_head.weight.detach().cpu().squeeze()
    if weights.ndim == 0:
        weights = weights.unsqueeze(0)

    torch.save(weights, config["output_path"])
    print(f"Saved {weights.numel()} feature weights to {config['output_path']}")


if __name__ == "__main__":
    main()
