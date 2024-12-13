import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from functions import f1_score
from dataset import ClusteringDataset, TrainDataset, TestDataset

transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 64
learning_rate = 0.00002
num_epochs = 10
positive_weigh_factor = 1.0
num_classes = 11255

seed = 42
# Set seed for Python's built-in random number generator
torch.manual_seed(seed)
# Set seed for numpy
np.random.seed(seed)
# Set seed for CUDA if available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # Set cuDNN's random number generator seed for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("DEVICE = CUDA")



# Load Training metadata
train_data_path = "data/cubes/GLC24-PA-train-bioclimatic_monthly/" 
train_metadata_path = 'data/metadata/GLC24-PA-metadata-train.csv'
train_metadata = pd.read_csv(train_metadata_path)
cluster_dataset = ClusteringDataset(train_metadata)
cluster_dataloader = DataLoader(cluster_dataset,batch_size = len(cluster_dataset))
train_dataset = TrainDataset(train_data_path, None, cluster_dataset, None, subset="train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True, num_workers=4)

# Load Test metadata
test_data_path = "data/cubes/GLC24-PA-test-bioclimatic_monthly/"
test_metadata_path = 'data/metadata/GLC24-PA-metadata-test.csv'
test_metadata = pd.read_csv(test_metadata_path)
test_dataset = TestDataset(test_data_path, test_metadata, subset="test")
test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False, num_workers=4)


class Auto_encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            nn.Linear(num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
            nn.ReLU(),
            nn.Linear(36, 18)
        )
         
        self.decoder = torch.nn.Sequential(
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = Auto_encoder(num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=25)



##   TRAINING   ##
print(f"Training for {num_epochs} epochs started.")
p =  0.1 # part of the validation dataset
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    validation = False
    Loss = 0

    for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total = len(train_loader)):

        data = targets.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()


        if batch_idx > (1-p)*len(train_loader) and not validation :
            model.eval()
            validation = True
            F1Pred = 0
            F1Score = 0
            F1Pred2 = 0
            F1Pred3 = 0
            AccScore = 0
            Len = 0

        if not validation:
            outputs = model(data)
            positive_weight_factor = 1

            pos_weight = targets*positive_weight_factor  # All positive weights are equal to 10
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            loss = criterion(outputs, targets)
            Loss+= 0
            loss.backward()
            optimizer.step()
        
        else :
            with torch.no_grad():
                outputs = torch.sigmoid(model(data))
                N = len(outputs)
                for i in range(N):
                    F1Pred  += f1_score(outputs[i].cpu(), targets[i].cpu())

                    # spec = outputs[i].cpu()/max(outputs[i].cpu())
                    # norm = np.linalg.norm(spec)

                    # spec_ord = np.argsort(spec)
                    # assemblage = np.zeros_like(spec_ord)
                    # id = 0
                    # f1max, f1  = 0, 0
                    # while f1 >= f1max :
                    #     f1max = f1
                    #     assemblage_max = assemblage.copy()

                    #     spec_id = spec_ord[-id-1]
                    #     assemblage[spec_id] = 1
                    #     f1 = f1_score(assemblage,spec/(id + 1 + norm**2))
                    #     id += 1
                    
                    # F1Pred2 += f1_score(assemblage_max, targets[i].cpu())
                    # topk = np.zeros_like(spec)
                    # topk[spec_ord[-25:]] = 1
                    # F1Pred3 += f1_score(topk, targets[i].cpu())

                Len += len(targets)

        
    scheduler.step()

    print(f'Dice coeff : {F1Pred/Len:.2f}')
    print(f'Loss : {Loss}')
    # print(f'F1 closest : {F1Pred2/Len:.2f}')
    # print(f'F1 top25 : {F1Pred3/Len:.2f}')
    # F1.append(F1Pred/Len)
    # F1_2.append(F1Pred2/Len)
    # F1_3.append(F1Pred3/Len)



# Save the trained model
model.eval()
torch.save(model.state_dict(), "auto_encoder.pth")





