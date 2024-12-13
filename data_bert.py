from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
import torch
import numpy as np
import pandas as pd
import random
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import MiniBatchKMeans
from torch_kmeans import KMeans
from tqdm import tqdm

from dataset import TrainDataset, TestDataset, ClusteringDataset, LexicoDataset
from model import ModifiedResNet18
from train import Run,Run_baseline
from clustering import Clustering, HClustering, MClustering
from functions import assembly
from functions import dist1




dict_path = "data/speciesIdTab_PA_PO.csv"
dictionnary = pd.read_csv(dict_path, sep = ';', usecols = ['species','speciesId'])

train_data_path = "data/cubes/GLC24-PA-train-bioclimatic_monthly/" 
train_metadata_path = 'data/metadata/GLC24-PA-metadata-train.csv'
train_metadata = pd.read_csv(train_metadata_path)

label_dt = LexicoDataset(train_metadata, dictionnary)

device= torch.device('cuda')

config = BertConfig.from_pretrained("/home/gigotleandri/Documents/plantbert_space/plantbert_text_classification_model", output_hidden_states=True, output_attentions=True)
model = BertForSequenceClassification.from_pretrained("/home/gigotleandri/Documents/plantbert_space/plantbert_text_classification_model", config = config)

model.to(device)

tokenizer = AutoTokenizer.from_pretrained("/home/gigotleandri/Documents/plantbert_space/plantbert_text_classification_model", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token = '<pad>'



lt_space = np.zeros((len(label_dt), 768))
surveyid = np.zeros((len(label_dt),1))
for i in tqdm(range(len(label_dt))):
    id, species = label_dt[i]
    input = tokenizer(text = species, padding=True, return_tensors='pt',is_split_into_words=True)
    while np.shape(input.input_ids)[1] > 512:
        species = species[:-1]
        input = tokenizer(text = species, padding=True, return_tensors='pt',is_split_into_words=True)

    with torch.no_grad() :
        output = model.forward(**input.to(device))
    pooled_output = torch.cat(tuple([output.hidden_states[i] for i in [-1]]), dim=-1)
    pooled_output = pooled_output[:, 0, :]


    lt_space[i] = pooled_output.cpu()
    surveyid[i,0] = id

pd.DataFrame(np.concatenate((surveyid, lt_space), axis = 1)).rename(columns = {0 : 'surveyId'}).to_csv("data/latent_space.csv", index = False)


