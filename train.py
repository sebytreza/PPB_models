import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from functions import f1_score

class Run():
    def __init__(self,model,optimizer,scheduler,device):
        
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
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device


    def train(self, train_loader,num_epochs,Ck):

        print(f"Training for {num_epochs} epochs started.")
        reclustering = None
        for epoch in range(num_epochs):
            self.model.train()
            for batch_idx, (data, targets, _) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)
                
                positive_weight_factor = 1
                pos_weight = targets*positive_weight_factor  # All positive weights are equal to 10
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = criterion(outputs, targets)
                #print(np.shape(targets), np.shape(outputs))
                loss.backward()
                self.optimizer.step()

                if batch_idx % 278 == 0:
                    score = 0
                    for output, target in zip(outputs,targets):
                        score += f1_score(Ck[torch.argmax(outputs, dim = 1).cpu()], Ck[torch.argmax(targets, dim = 1).cpu().numpy()])
                    print('mean F1 score :' , score/len(targets))
                    #print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

            self.scheduler.step()
            #print("Scheduler:",self.scheduler.state_dict())
            if reclustering != None :
                print('Estimated percentage of category change :', (torch.sum(torch.argmax(outputs, dim = 1).cpu() == reclustering)/len(outputs)).item())
            reclustering = torch.argmax(outputs, dim = 1).cpu()

        # Save the trained model
        self.model.eval()
        torch.save(self.model.state_dict(), "models/resnet18-with-bioclimatic-cubes.pth")

    def test(self,test_loader, Ck_spec):

        with torch.no_grad():
            surveys = []
            Spec = None
            for data, surveyID in tqdm(test_loader, total=len(test_loader)):
                
                data = data.to(self.device)
                
                outputs = self.model(data)
                predictions = torch.sigmoid(outputs).cpu().numpy()
                
                ck_spec = Ck_spec[np.argmax(predictions, axis = 1)]

                if Spec is None:
                    Spec = ck_spec
                else:
                    Spec = np.concatenate((Spec, ck_spec), axis = 0)

                surveys.extend(surveyID.cpu().numpy())



        data_concatenated = [' '.join(map(str, np.where(row == 1)[0])) for row in Spec]

        pd.DataFrame({'surveyId': surveys, 'predictions': data_concatenated,}).to_csv("submissions/prediction.csv", index = False)



    


