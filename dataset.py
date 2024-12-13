import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

num_classes = 11255


class SpeciesDataset(Dataset):
    def __init__(self, metadata, concat = False):

        self.concat = concat
        if concat :
            self.metadata = metadata.dropna(subset="speciesId").reset_index(drop=True)
            self.label_dict = metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()

        else :
            self.metadata = metadata
            self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
            self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
            self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
            self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)
            

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]

        if self.concat :
            species_ids = np.array(self.label_dict.get(survey_id, '')[0].split(' ')).astype(int)
        else :
            species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(num_classes)  
        for species_id in species_ids:
            label_id = int(species_id)
            label[label_id] = 1  # Set the corresponding class index to 1 for each species

        return survey_id, label
    

class PostBERTDataset(Dataset):
    def __init__(self, metadata,survey_id):

        self.metadata = metadata.loc[metadata.surveyId.isin(survey_id)]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.metadata.iloc[idx,1:].values

        return label

class LexicoDataset(Dataset):
    def __init__(self, metadata, dictionnary):
        self.metadata = metadata
        self.dictionnary = dictionnary
        self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
        self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        self.order = list(self.metadata['speciesId'].value_counts().index)
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]
        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        species_ids = sorted(species_ids, key = self.order.index)
        label = []
        for species_id in species_ids:
            label_id = int(species_id)
            label.append(self.dictionnary.species[label_id])  # Create the list of species name

        return survey_id, label
    

    


class TrainDataset(Dataset):
    def __init__(self, data_dir, survey_id, spec_dt, N_cluster, cluster = None, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.label_dt = spec_dt
        self.cluster = cluster
        self.N_cluster = N_cluster
        self.survey_id = survey_id
        self.spec_dt = spec_dt


    def __len__(self):
        return len(self.survey_id)

    def __getitem__(self, idx):
        survey_id = self.survey_id[idx]
        sample = torch.load(os.path.join(self.data_dir, f"GLC24-PA-train-bioclimatic_monthly_{survey_id}_cube.pt"))


        # Ensure the sample is in the correct format for the transform
        if isinstance(sample, torch.Tensor):
            sample = sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()


        if self.transform:
            sample = self.transform(sample)
        
        species = self.spec_dt[idx]

        if self.cluster is None :
            return sample, species
        
        label = torch.zeros(self.N_cluster)
        label[self.cluster[idx]] = 1


        return sample, label, species
    
class TestDataset(TrainDataset):
    def __init__(self, data_dir, metadata, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata


    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        
        survey_id = self.metadata.surveyId[idx]
        sample = torch.load(os.path.join(self.data_dir, f"GLC24-PA-test-bioclimatic_monthly_{survey_id}_cube.pt"))
        
        if isinstance(sample, torch.Tensor):
            sample = sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample, survey_id

