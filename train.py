import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from functions import f1_score,weighted_assignment
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

class Run():
    def __init__(self,model,optimizer,scheduler,device):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device


    def train(self, train_loader,validation_loader, num_epochs,Ck, save = None,save_loss = None):
        print(f"Training for {num_epochs} epochs started.")
        Reclustering = None
        n_Ck = torch.Tensor(normalize(Ck, axis = 1))
        F1 = []
        F1S = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            ## TRAINING ##
            self.model.train()
            for _, (data, targets, species) in tqdm(enumerate(train_loader), total = len(train_loader)):
                data = data.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()


                outputs = self.model(data)
                
                criterion = torch.nn.CrossEntropyLoss() 
                loss = criterion(outputs, targets)
                #print(np.shape(targets), np.shape(outputs))
                loss.backward()
                self.optimizer.step()

            ## VALIDATION ##
            self.model.eval()
            F1Pred = 0
            F1Score = 0
            F1Soft = 0
            AccScore = 0
            Len = 0
            Max_cluster = torch.Tensor([])                

            for _ , (data,targets, species) in tqdm(enumerate(validation_loader), total = len(validation_loader)):
        
                data = data.to(self.device)
                targets = targets.to(self.device)
                species = species.to(self.device)
                with torch.no_grad():
                    outputs = self.model(data)
                    predictions = torch.nn.functional.softmax(outputs, dim= -1).cpu()
                    N = len(outputs)
                    for i in range(N):
                        soft_prediction = weighted_assignment(predictions[i],n_Ck)
                        F1Soft += f1_score(soft_prediction, species[i].cpu().numpy())
                        F1Pred  += f1_score(Ck[torch.argmax(outputs[i], dim = 0).cpu()], Ck[torch.argmax(targets[i], dim = 0).cpu().numpy()])
                        F1Score += f1_score(Ck[torch.argmax(outputs[i], dim = 0).cpu()], species[i].cpu().numpy())


                    AccScore += sum(torch.argmax(outputs, dim = 1).cpu() == torch.argmax(targets, dim = 1).cpu())
                    Len += len(targets)
                    Max_cluster = torch.cat((Max_cluster, torch.argmax(outputs, dim = 1).cpu()))

                
            self.scheduler.step()

            print(f'Score F1: {F1Score/Len:.2f}')
            print(f'Dice coeff : {F1Pred/Len:.2f}')
            print(f'Score F1 soft : {F1Soft/Len:.2f}')
            F1.append(F1Score/Len)
            F1S.append(F1Soft/Len)
            

            if Reclustering != None :
                print(f'Estimated percentage of category changes :{torch.sum( Max_cluster == Reclustering).item()/len(Max_cluster)*100 :.2f} %')
            Reclustering = Max_cluster

        # Save the trained model
        if save is not None :
            self.model.eval()
            torch.save(self.model.state_dict(), "models/" + save + ".pth")
        if save_loss is not None :
            np.save("loss/" + save_loss +".npy", np.array(F1))
            np.save("loss/" + save_loss +"_soft.npy", np.array(F1S))

    def test(self,test_loader, Ck_spec):

        with torch.no_grad():
            n_Ck = torch.Tensor(normalize(Ck_spec, axis = 1))
            surveys = []
            Spec = None
            Spec_soft = None
            for data, surveyID in tqdm(test_loader, total=len(test_loader)):
                
                data = data.to(self.device)
                
                outputs = self.model(data)
                predictions = torch.nn.functional.softmax(outputs, dim= -1).cpu()
                
                ck_spec = Ck_spec[np.argmax(predictions, axis = 1)]

                if Spec is None:
                    Spec = ck_spec
                else:
                    Spec = np.concatenate((Spec, ck_spec), axis = 0)

                for i in range(len(data)):
                    soft_prediction = weighted_assignment(predictions[i],n_Ck)

                    if Spec_soft is None:
                        Spec_soft = [soft_prediction]
                    else:
                        Spec_soft = np.concatenate((Spec_soft, [soft_prediction]), axis = 0)

                surveys.extend(surveyID.cpu().numpy())



        data_concatenated = [' '.join(map(str, np.where(row == 1)[0])) for row in Spec]
        pd.DataFrame({'surveyId': surveys, 'predictions': data_concatenated,}).to_csv("submissions/prediction_classif.csv", index = False)


        data_concatenated = [' '.join(map(str, np.where(row == 1)[0])) for row in Spec_soft]
        pd.DataFrame({'surveyId': surveys, 'predictions': data_concatenated,}).to_csv("submissions/prediction_classif_soft.csv", index = False)



class Run_baseline():
    def __init__(self,model,optimizer,scheduler,device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device


    def train(self, train_loader,validation_loader, num_epochs,Ck, save = None, save_loss = None):
        F1 =  []
        F1_2 = []
        F1_3 = []
        TRUE = []
        PRED = []

        print(f"Training for {num_epochs} epochs started.")
        p =  0.1 # part of the validation dataset
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")


            ## TRAINING ##
            self.model.train()

            for _, (data, targets, species) in tqdm(enumerate(train_loader), total = len(train_loader)):
                data = data.to(self.device)
                targets = species.to(self.device) 
                species = species.to(self.device)
                self.optimizer.zero_grad()


                outputs = self.model(data)
                positive_weight_factor = 1
                pos_weight = targets*positive_weight_factor  # All positive weights are equal to 10
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))

                loss = criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()
            
            ## VALIDATION ##
            self.model.eval()
            F1Pred = 0
            F1Pred2 = 0
            F1Pred3 = 0
            Len = 0
            for _, (data, targets, species) in tqdm(enumerate(validation_loader), total = len(validation_loader)):
                
                data = data.to(self.device)
                species = species.to(self.device)
                targets = species



                with torch.no_grad():
                    outputs = torch.sigmoid(self.model(data))
                    N = len(outputs)

                    TRUE.append(species[0].cpu().numpy())
                    PRED.append(outputs[0].cpu().numpy())
                
                    
                    for i in range(N):
                        F1Pred  += f1_score(outputs[i].cpu(), targets[i].cpu())

                        spec = outputs[i].cpu()/max(outputs[i].cpu())
                        proj = np.zeros_like(spec)
                        mask = np.where(spec > 0)[0]
                        spec = spec[mask]
                        spec_ord = np.argsort(spec)
                        assemblage = np.zeros_like(spec_ord)
                        id = 0
                        f1max, f1  = 0, 0
                        while f1 >= f1max :
                            f1max = f1

                            spec_id = spec_ord[-id-1]
                            assemblage[spec_id] = 1
                            f1 = f1_score(assemblage,spec)
                            id += 1         
                        proj[mask[spec_ord[-id:]]]  = 1
                        F1Pred2 += f1_score(proj, targets[i].cpu())
                        topk = np.zeros_like(proj)
                        topk[mask[spec_ord[-25:]]] = 1
                        F1Pred3 += f1_score(topk, targets[i].cpu())

                    Len += len(targets)

                
            self.scheduler.step()

            print(f'Dice coeff : {F1Pred/Len:.2f}')
            print(f'F1 closest : {F1Pred2/Len:.2f}')
            print(f'F1 top25 : {F1Pred3/Len:.2f}')
            F1.append(F1Pred/Len)
            F1_2.append(F1Pred2/Len)
            F1_3.append(F1Pred3/Len)

            np.save("data/true.npy", TRUE)
            np.save("data/pred.npy",PRED)
  

        
        # Save the trained model
        if save is not None :
            self.model.eval()
            torch.save(self.model.state_dict(), "models/" + save + ".pth")
        if save_loss is not None :
            np.save("loss/" + save_loss +"ct.npy", np.array(F1))
            np.save("loss/" + save_loss +"proj.npy", np.array(F1_2))
            np.save("loss/" + save_loss +"topk.npy", np.array(F1_3))

    def test(self,test_loader):
        PRED_T = []
        with torch.no_grad():
            surveys = []
            Spec = None
            Spec_topk = None
            Spec_bis = None
            for data, surveyID in tqdm(test_loader, total=len(test_loader)):
                
                data = data.to(self.device)
                
                outputs = self.model(data)
                outputs = torch.sigmoid(outputs).cpu().numpy()
                
                N = len(outputs)
                for i in range(N):
                    spec = outputs[i]
                    PRED_T.append(outputs[i])
                    assemblage_max = np.zeros_like(spec)
                    mask = np.where(spec > 0)[0]
                    spec = spec[mask]
                    spec_ord = np.argsort(spec)
                    assemblage = np.zeros_like(spec_ord)
                    id = 0
                    f1max, f1  = 0, 0
                    while f1 >= f1max :
                        f1max = f1

                        spec_id = spec_ord[-id-1]
                        assemblage[spec_id] = 1
                        f1 = f1_score(assemblage,spec)
                        id += 1         
                    assemblage_max[mask[spec_ord[-id+1:]]]  = 1

                    topk = np.zeros_like(assemblage_max)
                    topk[mask[spec_ord[-25:]]] = 1



                    if Spec is None:
                        Spec = [assemblage_max]
                    else:
                        Spec = np.concatenate((Spec, [assemblage_max]), axis = 0)

                    if Spec_topk is None:
                        Spec_topk = [topk]
                    else:
                        Spec_topk = np.concatenate((Spec_topk, [topk]), axis = 0)
                    


                    spec = outputs[i]/max(outputs[i])
                    proj = np.zeros_like(assemblage_max)
                    spec_ord = np.argsort(spec)
                    assemblage = np.zeros_like(spec_ord)
                    id = 0
                    f1max, f1  = 0, 0
                    while f1 >= f1max :
                        f1max = f1
                        id += 1
                        spec_id = spec_ord[-id]
                        assemblage[spec_id] = 1
                        f1 = f1_score(assemblage,spec)
                               
                    proj[mask[spec_ord[-id+1:]]]  = 1

                    if Spec_bis is None:
                        Spec_bis = [proj]
                    else:
                        Spec_bis = np.concatenate((Spec_bis, [proj]), axis = 0)



                surveys.extend(surveyID.cpu().numpy())

        
        np.save('data/pred_test.npy', PRED_T)
                
        data_concatenated = [' '.join(map(str, np.where(row == 1)[0])) for row in Spec]
        pd.DataFrame({'surveyId': surveys, 'predictions': data_concatenated,}).to_csv("submissions/prediction.csv", index = False)

        data_concatenated = [' '.join(map(str, np.where(row == 1)[0])) for row in Spec_topk]
        pd.DataFrame({'surveyId': surveys, 'predictions': data_concatenated,}).to_csv("submissions/prediction_topk.csv", index = False)

        data_concatenated = [' '.join(map(str, np.where(row == 1)[0])) for row in Spec_bis]
        pd.DataFrame({'surveyId': surveys, 'predictions': data_concatenated,}).to_csv("submissions/prediction_bis.csv", index = False)
    


