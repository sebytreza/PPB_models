import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

num_classes = 11255


class ClusteringDataset(Dataset):
    def __init__(self, metadata):

        self.metadata = metadata
        self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
        self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        survey_id = self.metadata.surveyId[idx]
        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(num_classes)  
        for species_id in species_ids:
            label_id = species_id
            label[label_id] = 1  # Set the corresponding class index to 1 for each species

        return label


class TrainDataset(Dataset):
    def __init__(self, data_dir, cluster, cluster_dt, N_cluster, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = cluster_dt.metadata
        self.cluster_dt = cluster_dt
        self.cluster = cluster
        self.N_cluster = N_cluster

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]
        sample = torch.load(os.path.join(self.data_dir, f"GLC24-PA-{self.subset}-bioclimatic_monthly_{survey_id}_cube.pt"))
        label = torch.zeros(self.N_cluster)
        label[self.cluster[idx]] = 1.

        # Ensure the sample is in the correct format for the transform
        if isinstance(sample, torch.Tensor):
            sample = sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()


        if self.transform:
            sample = self.transform(sample)

        return sample, label, survey_id
    
class TestDataset(TrainDataset):
    def __init__(self, data_dir, metadata, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata
        
    def __getitem__(self, idx):
        
        survey_id = self.metadata.surveyId[idx]
        sample = torch.load(os.path.join(self.data_dir, f"GLC24-PA-{self.subset}-bioclimatic_monthly_{survey_id}_cube.pt"))
        
        if isinstance(sample, torch.Tensor):
            sample = sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample, survey_id

