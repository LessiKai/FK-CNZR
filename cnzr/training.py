
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from random import randint
from colorama import Fore, Back, Style
from torch import nn
from pathlib import Path




class ModelV1(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, out_features),
        )

    def forward(self, x):
        x = self.network(x)
        x = torch.softmax(x, dim=1).to(torch.float)
        return x



def train_model(x, y, model, n_classes, print_debug = True, epochs = 200, learning_rate = 0.0001618243, batch_size = 32, seed_nn = 0 ):
    
    # Gewichtung berücksichtigen
    cancer_type, cancer_count = np.unique(y, return_counts= True)
    sorted_inds = np.argsort(cancer_type)
    cancer_count = cancer_count[sorted_inds]
    cancer_type = cancer_type[sorted_inds]
    cancer_loss_weights = 1 / cancer_count
    cancer_loss_weights = cancer_loss_weights / np.sum(cancer_loss_weights)
    print(cancer_loss_weights)
    
    print(f"Modelparameter: lr= {learning_rate}, batch_size = {batch_size}.")
    if(batch_size>len(x)/2): batch_size = len(x)

    # Data Preprocessing
    x = torch.tensor(x[0:len(x) -len(x) % batch_size], dtype=torch.float32)        # Arraytrunkierung des Rests (Vereinfacht Algorithmus)
    y = torch.tensor(y[0:len(y) -len(y) % batch_size], dtype=torch.float32)
    n_batches = int(len(x) / batch_size)
    
    N = len(x)
    inds = np.arange(0, N, 1)
    
    print(f"\n→ Initiating Modeltraining. Total Epochs: {epochs}")

    # Setup
    loss_fn = nn.CrossEntropyLoss() # Setup Loss Function
    torch.manual_seed(seed_nn)
    optim   = torch.optim.Adam(model.parameters(), lr=learning_rate)     # Setup optimizer
    model.train()

    writer = SummaryWriter()

    for epoch in range(epochs):  
        if epoch == 0 or epochs < 10: 
            print_command = True
        else: 
            if (epoch % (epochs/20) == 0): 
                print_command = True
            else:
                print_command = False
        if print_command: print(f"{'↪': ^2} | Epoch: {epoch: ^4}     Advanced {int(epoch/epochs*100): ^4}%     ", end='')
        
        np.random.shuffle(inds)
        x_batches, y_batches = np.split(x, n_batches), np.split(y, n_batches)
        
        for i in range(n_batches):
            batch_inds = inds[i *batch_size: (i+1) *batch_size -1 ]
            prob = model(x[batch_inds])
            loss    = loss_fn(prob, y[batch_inds])
                                   
            optim.zero_grad()
            loss.backward()
            optim.step()

        if print_command: 
            batch_inds = inds[0: 3*batch_size -1 ]
            y_pred = model(x[batch_inds])
            loss = loss_fn(y_pred, y[batch_inds])
            
            hit_arr = np.zeros((len(batch_inds), y.shape[1]))
            for obs in range(len(batch_inds)):
                if torch.argmax(y_pred[obs]) == np.argmax(y[obs]):
                    hit_arr[obs][torch.argmax(y_pred[obs])] = 1
            hits = np.sum(hit_arr == 1)
            accuracy = hits/len(batch_inds)
            print(f"Batch[0] avg. loss: {loss:.4f}")
            writer.add_scalar('charts/train_loss', loss, epoch)
            writer.add_scalar('charts/train_acc', accuracy, epoch)

    writer.close()
    print("Model has been Trained 💪")
    return model
    



def test_model(x, y, model):
    
    model.eval()
    writer = SummaryWriter()
    run_dir    = Path("./runs").resolve()
    run = len([d for d in run_dir.iterdir()])

    y_pred = model(torch.tensor(x, dtype=torch.float))
    
    #accuracy
    hit_arr = np.zeros((len(x), y.shape[1]))
    for obs in range(len(x)):
        if torch.argmax(y_pred[obs]) == np.argmax(y[obs]):
            hit_arr[obs][torch.argmax(y_pred[obs])] = 1
    hits = np.sum(hit_arr == 1)
    accuracy = hits/len(x)

    y_pred = model(torch.tensor(x[0:10], dtype=torch.float))
    print(f"{'Treffercounts': ^47}")
    print(f"{'C1': ^14} {'C2': ^14} {'C3': ^14}")
    print(f"{np.sum(hit_arr.T[0] == 1): ^13}   {np.sum(hit_arr.T[1] == 1): ^13}   {np.sum(hit_arr.T[2] == 1): ^13}   - Treffer")
    print(f"{int(np.sum(hit_arr.T[0] == 1)/hits): ^13}   {int(np.sum(hit_arr.T[1] == 1)/hits*100): ^13}   {int(np.sum(hit_arr.T[2] == 1)/hits*100): ^13}   - Trefferanteil")
    print()
    print(f"{int(np.sum(y.T[0] == 1)/len(x)*100): ^13}   {int(np.sum(y.T[1] == 1)/len(x)*100): ^13}   {int(np.sum(y.T[2] == 1)/len(x)*100): ^13}   - Klassenanteil im Testset")
    
    writer.add_scalar('charts/test_acc', accuracy, run)
    print(f"▶ Auswertung abgeschlossen. Von {len(x)} Klassen wurden {hits} richtig erkannt. Die Genauigkeit beträgt {accuracy*100:.2f} %.")
    writer.close()

    # tensorboard --logdir='runs'
    return