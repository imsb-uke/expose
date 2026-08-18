from torch.utils.data import Dataset
import os
import torch
import h5py

from src.utils.transforms import to_float_tensor


class DatasetEmbeddings(Dataset):
    def __init__(self, metadata_df, embedding_dir, label_col):
        self.metadata_df = metadata_df
        self.embedding_dir = embedding_dir
        self.label_col = label_col

        # Store sample ids individually
        self.sample_ids = self.metadata_df.sample_id.tolist()
        
    def __len__(self):
        return len(self.sample_ids)
    
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_info = self.metadata_df.loc[sample_id]

        # load embeddings for sample
        embedding_path = os.path.join(self.embedding_dir, sample_info.embedding_path) 
        with h5py.File(embedding_path,'r') as file:
            features = file['patch_features'][:] # N x embedding_dim
            features = torch.Tensor(features)

        if self.label_col is not None:
            label = sample_info[self.label_col]
            label = to_float_tensor(label)
            return features, label
        
        return features