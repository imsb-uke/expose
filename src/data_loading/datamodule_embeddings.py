import lightning as L
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torchvision.transforms import v2
import pandas as pd

from src.data_loading.dataset_embeddings import DatasetEmbeddings

class DataModuleEmbeddings(L.LightningDataModule):
    def __init__(
            self, 
            data_dir,
            metadata_path,
            domain_list,
            num_workers,
            batch_size=1,
            batch_size_val=1,
            label_col=None
        ):
        super().__init__()

        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.domain_list = domain_list
        if isinstance(self.domain_list, str):
            self.domain_list = domain_list.replace(' ', '').split(",")
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val    
        self.label_col = label_col    


    def setup(self, stage):
        # Load metadata 
        self.metadata_df = (
            pd.read_csv(self.metadata_path)
            .query("domain in @self.domain_list")
        )

        if stage == "fit":
            self.metadata_train_df = self.metadata_df.loc[lambda df_: df_["split"] == "train"]
            self.metadata_val_df = self.metadata_df.loc[lambda df_: df_["split"] == "val"]

            self.train_dataset = DatasetEmbeddings(
                metadata_df=self.metadata_train_df,
                embedding_dir = self.data_dir,
                label_col = self.label_col
            )
            self.val_dataset = DatasetEmbeddings(
                metadata_df=self.metadata_val_df,
                embedding_dir = self.data_dir,
                label_col = self.label_col
            )
            
            print(f"Using {self.metadata_train_df.shape[0]} train / {self.metadata_val_df.shape[0]} val images. {self.metadata_df.shape[0]} images in total.")

        elif stage == "test":
            print("Using test dataset")
            self.metadata_test_df = self.metadata_df.loc[lambda df_: df_["split"] == "test"]
            self.test_dataset = DatasetEmbeddings(
                metadata_df=self.metadata_test_df,
                embedding_dir = self.data_dir,
                label_col = self.label_col
            )
            print(f"Using {self.metadata_test_df.shape[0]} test images. {self.metadata_df.shape[0]} images in total.")


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=False,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        if self.label_col is None:
            data = [item for item in batch]
            return pad_sequence(data, batch_first=True)
        else:
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            target = torch.tensor(target)
            return [pad_sequence(data, batch_first=True), target]