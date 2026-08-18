import torch
from lightning import LightningModule


class LitModule(LightningModule):
    def __init__(
            self, 
            config=None,
            model=None,
            
    ):
        super().__init__()
        self.model = model

        self.lr = config["learning_rate"]
        self.l1_weight = config["l1_weight"]
        self.sae_hidden_dim = config["embedding_dim"] * config["expansion_factor"]


    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, prefix="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, prefix="test")
        mse, l1, loss, l0, explained_variance, dead_features_fraction, sae_metric = self._calculate_logged_metrics(batch)

        output = {
            "idx": batch_idx,
            "mse": mse.item(),
            "l1": l1.item(),
            "loss": loss.item(),
            "l0": l0.item(),
            "explained_variance": explained_variance.item(),
            "dead_fraction": dead_features_fraction,
            "sae_metric": sae_metric.item(),
        }
        return output


    def step(self, batch, batch_idx, prefix):
        mse, l1, loss, l0, explained_variance, dead_features_fraction, sae_metric = self._calculate_logged_metrics(batch)

        self.log(f"{prefix}_mse_loss", mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_l1_loss", l1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_l0", l0, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_explained_variance", explained_variance, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_dead_features_fraction", dead_features_fraction, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_sae_metric", sae_metric, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def _calculate_logged_metrics(self, batch):
        x = batch
        sae_hidden, sae_out = self(x)      

        mse, l1, loss = self._calculate_loss(sae_hidden, sae_out, x)
        l0, explained_variance = self._calculate_metrics(sae_hidden, sae_out, x)
        dead_features_fraction = self._calculate_dead_fraction(sae_hidden)

        # best SAE: low number of activations, high explained variance, low number of dead_features
        sae_metric = (1 - (l0 / self.sae_hidden_dim)) * explained_variance * (1 - dead_features_fraction)
        return mse, l1, loss, l0, explained_variance, dead_features_fraction, sae_metric


    def _calculate_loss(self, sae_hidden, sae_out, sae_in):
        # https://github.com/dynamical-inference/cytosae/blob/main/src/sae_training/sparse_autoencoder.py#L89
        mse_loss = (
            torch.pow((sae_out - sae_in.float()), 2)
            / (sae_in**2).sum(dim=-1, keepdim=True).sqrt()
        )
        mse_loss = mse_loss.mean()
        sparsity = torch.abs(sae_hidden).sum(dim=-1).mean(dim=(0,))
        l1_loss = self.l1_weight * sparsity

        loss = mse_loss + l1_loss 

        return mse_loss, l1_loss.mean(), loss.mean()


    def _calculate_metrics(self, sae_hidden, sae_out, sae_in):
        l0 = (sae_hidden > 0).float().sum(-1).mean(-1).mean() # sum over sparse dims, mean over patches and batches 

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).mean().squeeze()
        total_variance = sae_in.pow(2).sum(-1).mean()
        explained_variance = 1 - per_token_l2_loss / total_variance
        explained_variance = explained_variance.mean()  
        # clip to [0,1]
        explained_variance = explained_variance.clamp(0, 1)    

        return l0, explained_variance

    
    def _calculate_dead_fraction(self, sae_hidden: torch.Tensor, threshold=1e-6):
        """
        Count dead neurons in a ReLU layer with shape (batch, num_patches, sae_hidden_dim).
        
        A neuron is dead if it outputs <= threshold **for all patches in all batches**.
        """
        # Combine batch and patch dimensions
        B, P, D = sae_hidden.shape
        flattened = sae_hidden.reshape(B * P, D)
        
        # A neuron is dead if all activations are <= threshold
        dead_mask = (flattened.abs() <= threshold).all(dim=0)
        dead_neurons = dead_mask.sum().item()
        dead_fraction = dead_neurons / D
        return dead_fraction
            

    def on_train_epoch_end(self):
        self.shared_on_epoch_end()

    def on_validation_epoch_end(self):
        self.shared_on_epoch_end()

    def shared_on_epoch_end(self):
        if self.trainer.sanity_checking:
            return

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
        )

        return {
            "optimizer": optimizer, 
        }