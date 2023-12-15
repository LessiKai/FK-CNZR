
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
from colorama import Fore, Back, Style
from torch import nn



class ModelV1(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, out_features),
        )

    def forward(self, x):
        x = self.network(x)
        x = torch.softmax(x, dim=1)
        return x



def train_model(x, y, model, n_classes, print_debug = True):
    # Controll Variables for the NN
    epochs              = 100
    elements_per_batch  = 20
    learning_rate       = 0.00005
    if(elements_per_batch>len(x)/2): elements_per_batch = len(x)

    # Data Preprocessing
    ## Sample X-y into Batches
    x = x[0:len(x) -len(x) % elements_per_batch]        # Arraytrunkierung des Rests (Vereinfacht Algorithmus)
    y = y[0:len(y) -len(y) % elements_per_batch]
    n_batches = int(len(x) / elements_per_batch)
    if(print_debug == True): print(f"Length X = {len(x)}")

    ## TENSOR Konstruktion für NN
    x_batches = torch.tensor(np.split(x             , n_batches), dtype=torch.float32)
    y_batches = torch.tensor(np.split(y , n_batches))
    
    # Model Training
    print(f"\n→ Initiating Modeltraining. Total Epochs: {epochs}")
    
    epoch_print = 0
    loss_fn = nn.CrossEntropyLoss() # Setup Loss Function

    torch.manual_seed(42)
    for epoch in range(epochs):  
        if epoch == 0 or epochs < 10: 
            print_command = True
        else: 
            if (epoch % (epochs/20) == 0): 
                print_command = True
            else:
                print_command = False
        if print_command: print(f"{"↺": ^2} | Epoch: {epoch: ^4}     Advanced {int(epoch/epochs*100): ^4}%     ", end='')
        
        model.train()
        for idx, x_batch in enumerate(x_batches):
            y_batch = y_batches[idx]
            y_pred  = model(x_batch)
            optim   = torch.optim.Adam(model.parameters(), lr=learning_rate)     # Setup optimizer
            loss    = loss_fn(y_pred, y_batch)
            loss.backward()
            optim.step()
            optim.zero_grad()
        if print_command: 
            y_pred = model(x_batches[0])
            print(f"Batch[0] avg. loss: {loss_fn(y_pred, y_batches[0]):.4f}     ")
    
        
    print("Model has been Trained 💪")
    return model
    


def test_model(x, y, model):
    model.eval()
    y_pred = model(torch.tensor(x[0:10], dtype=torch.float32))
    print(f"{"y": ^15} {"y_pred": ^15}")
    for i,data in enumerate(y_pred):
        if str(y[0:10][i].tolist()) == str(torch.round(y_pred[i]).tolist()):
            print(f"{Fore.GREEN}{str(y[0:10][i].tolist()): ^13} {str(torch.round(y_pred[i]).tolist()): ^13}")
        else: 
            print(f"{Fore.RED}{str(y[0:10][i].tolist()): ^13} {str(torch.round(y_pred[i]).tolist()): ^13}", end="")
            print(Fore.RESET)

    # 1-1, 1-0, 0-1, 0-0
    # Right Trues, Right False, Wrong Trues, 

    return