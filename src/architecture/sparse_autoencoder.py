from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch



class SAEEncoder(LightningModule):
    def __init__(self, d_in, d_sae):
        super().__init__()
        self.linear = nn.Linear(d_in, d_sae, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_sae))
        self.activation = nn.ReLU()

    def forward(self, x, decoder_bias):
        x = x - decoder_bias # input centering
        h_pre_act = F.linear(x, self.linear.weight, self.bias)
        h = self.activation(h_pre_act)
        return h


class SAEDecoder(LightningModule):
    def __init__(self, d_sae, d_in):
        super().__init__()
        self.linear = nn.Linear(d_sae, d_in, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_in))

    def forward(self, x):
        x_hat = self.linear(x)
        x_hat = x_hat + self.bias
        return x_hat


class SAE(LightningModule):
    def __init__(self, d_in, d_sae):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = SAEEncoder(d_in, d_sae)
        self.decoder = SAEDecoder(d_sae, d_in)

    def forward(self, x):
        latents = self.forward_encoder(x)
        recons = self.forward_decoder(latents)
        return latents, recons

    def forward_encoder(self, x):
        return self.encoder(x, self.decoder.bias)

    def forward_decoder(self, x):
        return self.decoder(x)


    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.decoder.weight.data /= torch.norm(self.decoder.weight.data, dim=1, keepdim=True)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        self.set_decoder_norm_to_unit_norm()