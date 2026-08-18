import torch
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger

import wandb
from torchmetrics.classification import AUROC, Accuracy, F1Score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class LitModule(L.LightningModule):
    def __init__(
            self, 
            model,
            config,
            pos_weight
    ):
        super().__init__()
        self.model = model

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float)
        )

        self.lr = config["learning_rate"]
        self.num_classes = config["num_classes"]
        self.max_epochs = config["max_epochs"]
        self.batch_size = config["batch_size"]
        self.weight_decay = config["weight_decay"]

        self.use_wandb = config["use_wandb"]

        task = "binary"
        self.train_auroc = AUROC(task=task)
        self.val_auroc = AUROC(task=task)
        self.test_auroc = AUROC(task=task)

        self.train_acc = Accuracy(task=task)
        self.val_acc = Accuracy(task=task)
        self.test_acc = Accuracy(task=task)

        self.train_f1 = F1Score(task=task)
        self.val_f1 = F1Score(task=task)
        self.test_f1 = F1Score(task=task)

        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight) # sigmoid + bce 



    def on_fit_start(self):
        self.step_outputs = {
            "train": {"targets": [], "preds": []},
            "val": {"targets": [], "preds": []},
            "test": {"targets": [], "preds": []},
        }
    
    def on_test_start(self):
        self.test_outputs = []
        self.step_outputs = {
            "train": {"targets": [], "preds": []},
            "val": {"targets": [], "preds": []},
            "test": {"targets": [], "preds": []},
        }
       

    def forward(self, inputs):
        return self.model(inputs)


    def training_step(self, batch, batch_idx):
        probs, preds, y, loss = self.step(batch, batch_idx, prefix="train")
        self.train_auroc(probs, y)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        return loss


    def validation_step(self, batch, batch_idx):
        probs, preds, y, loss = self.step(batch, batch_idx, prefix="val")
        self.val_auroc(probs, y)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)


    def test_step(self, batch, batch_idx):
        probs, preds, y, loss = self.step(batch, batch_idx, prefix="test")
        self.test_auroc(probs, y)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)
        
        output = {
                "probs": probs.cpu().numpy().item(),
                "preds": preds.cpu().numpy().item(),
                "target": y.cpu().numpy().item(),
                "test_loss": loss.item(),
                "test_auroc": self.test_auroc.compute().item(),
                "test_acc": self.test_acc.compute().item(),
                "test_f1": self.test_f1.compute().item(),
                "batch_idx": batch_idx,
            }

        self.test_outputs.append(output)
        return output


    def step(self, batch, batch_idx, prefix):
        x, y = batch
        logits = self(x).squeeze(1)       
        probs = torch.sigmoid(logits)        
        preds = (probs > 0.5).long()         
        loss = self.loss(logits, y.float())
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.step_outputs[prefix]["targets"].append(y)
        self.step_outputs[prefix]["preds"].append(preds)
        return probs, preds, y, loss
        

    def on_train_epoch_end(self):
        self.shared_on_epoch_end(mode="train")

    def on_validation_epoch_end(self):
        self.shared_on_epoch_end(mode="val")

    def on_test_epoch_end(self):
        self.shared_on_epoch_end(mode="test")
    
    def shared_on_epoch_end(self, mode):
        if self.trainer.sanity_checking:
            return

        outputs = self.step_outputs[mode]

        if isinstance(self.logger, WandbLogger):
            self._log_confusion_matrix(outputs, mode, "confusion_matrix_bin_clas")
        
        self.step_outputs[mode] = {"targets": [], "preds": []}

    
    def _log_confusion_matrix(self, outputs, mode, title="confusion_matrix_bin_clas"):
        fig, ax = plt.subplots()
        y_test = self.cat_outputs(outputs, "targets").numpy()
        y_pred = self.cat_outputs(outputs, "preds").numpy()

        colors = {
            "train": "Blues",
            "val": "Oranges",
            "test": "Greens",
        }

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap=colors[mode],
            normalize="true",
        )

        # set scaling to 0-1
        disp.im_.set_clim(vmin=0, vmax=1)

        if self.use_wandb:
            self.logger.experiment.log(
                {
                    f"{title}_{mode}": wandb.Image(ax),
                },
            )
        else:
            plt.savefig(f"{self.results_dir}/{title}_{mode}.png")    
        plt.close()

    def cat_outputs(self, outputs, key: str, to_cpu: bool = True) -> torch.Tensor:
        """Concatenates tensor values of given key from the list of dicts ('outputs') received by the
        on_epoch_end function of the LightningModule."""
        out = torch.cat(outputs[key])
        return out.detach().cpu() if to_cpu else out


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs
        )            
        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler, 
        }