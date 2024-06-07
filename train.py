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
        Reclustering = None
        p =  0.1 # part of the validation dataset
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.model.train()
            validation = False

            for batch_idx, (data, targets, _) in tqdm(enumerate(train_loader), total = len(train_loader)):
                data = data.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                if batch_idx > (1-p)*len(train_loader) and not validation :
                    self.model.eval()
                    validation = True
                    F1Score = 0
                    AccScore = 0
                    Len = 0
                    Max_cluster = torch.Tensor([])

                if not validation:
                    outputs = self.model(data)
                    
                    positive_weight_factor = 1
                    pos_weight = targets*positive_weight_factor  # All positive weights are equal to 10
                    criterion = torch.nn.CrossEntropyLoss() #BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = criterion(outputs, targets)
                    #print(np.shape(targets), np.shape(outputs))
                    loss.backward()
                    self.optimizer.step()
                
                else :
                    with torch.no_grad():
                        outputs = self.model(data)
                        for output, target in zip(outputs,targets):
                            F1Score  += f1_score(Ck[torch.argmax(output, dim = 0).cpu()], Ck[torch.argmax(target, dim = 0).cpu().numpy()])
                        AccScore += sum(torch.argmax(outputs, dim = 1).cpu() == torch.argmax(targets, dim = 1).cpu())
                        Len += len(targets)
                        Max_cluster = torch.cat((Max_cluster, torch.argmax(outputs, dim = 1).cpu()))
                    

                
            self.scheduler.step()

            print(f'mean F1 score : {F1Score/Len:.2f}')
            print(f'mean Accuracy : {AccScore/Len:.2f}')

            if Reclustering != None :
                print(f'Estimated percentage of category changes :{torch.sum( Max_cluster == Reclustering).item()/len(Max_cluster)*100 :.2f} %')
            Reclustering = Max_cluster

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
                predictions = torch.sigmoid(outputs).cpu().numpy() #sigmoid useless here,
                
                ck_spec = Ck_spec[np.argmax(predictions, axis = 1)]

                if Spec is None:
                    Spec = ck_spec
                else:
                    Spec = np.concatenate((Spec, ck_spec), axis = 0)

                surveys.extend(surveyID.cpu().numpy())



        data_concatenated = [' '.join(map(str, np.where(row == 1)[0])) for row in Spec]

        pd.DataFrame({'surveyId': surveys, 'predictions': data_concatenated,}).to_csv("submissions/prediction.csv", index = False)



    


