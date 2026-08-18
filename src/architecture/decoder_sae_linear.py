from pytorch_lightning import LightningModule
import torch.nn as nn
import torch

from src.architecture.sparse_autoencoder import SAE


class DecoderSAELinear(LightningModule):
    def __init__(
        self,
        d_in,
        d_sae,
        num_classes,
        sae_ckpt_path,
        mask_domain_weights=False,
        domain_weights_path=None,
        topk=0,
        use_largest=True
    ):
        super().__init__()
        if sae_ckpt_path:
            self.sae = self._load_pretrained_sae(
                sae_ckpt_path,
                d_in,
                d_sae   
            )
        else:
            self.sae = SAE(d_in, d_sae)

        # Freeze
        for p in self.sae.parameters():
            p.requires_grad = False

        self.topk_mask = None
        if mask_domain_weights:
            # load domain weights and compute topk domain features
            domain_weights = torch.load(domain_weights_path)
            topk_domain_idx = domain_weights.topk(topk, largest=use_largest).indices 
                
            self.topk_mask = torch.ones(d_sae)
            self.topk_mask[topk_domain_idx] = 0

            print(f"Masking {topk_domain_idx.shape[0]} / {d_sae} domain features for training.")
        
        self.clas_head = nn.Linear(
            in_features=d_sae, 
            out_features=num_classes)


    def forward(self, x): 
        # transform embeddings
        x = self.sae.forward_encoder(x)

        if self.topk_mask is not None:
            # mask topk domain features: B x N x L -> B x N x L
            x = x * self.topk_mask.to(x.device)    

        # B x N x L -> B x N x 1
        x = x.mean(dim=1)

        # classification head: B x 1 x L
        x = self.clas_head(x)

        return x
    

    def _load_pretrained_sae(self, ckpt_path, input_dim, sparse_dim):
        sae = SAE(input_dim, sparse_dim)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Remove possible "model." prefix (Lightning sometimes adds it)
        cleaned_state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
        }

        # Keep ONLY encoder weights and encoder_bias and decoder_bias
        encoder_state_dict = {
            k: v
            for k, v in cleaned_state_dict.items()
            if k.startswith("encoder.")
            or k == "encoder_bias"
            or k == "decoder_bias"
        }

        # Load only those weights
        sae.load_state_dict(encoder_state_dict, strict=False)
        return sae