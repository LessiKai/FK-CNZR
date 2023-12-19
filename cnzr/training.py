
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
from random import randint
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
        x = torch.softmax(x, dim=1).to(torch.float)
        return x



def shuffle_indices(batch_indices, n_batches, N):
    # Ein Array mit vielen einsen. dann Auswahl von x random Elementen, setze dann skalar auf 0
    indice_server = np.zeros(N)
    for batch, arrays in enumerate(batch_indices):
        for index_pointer, array in enumerate(batch_indices[batch]):
            while batch_indices[batch][index_pointer] == 0:
                # solange noch kein Wert zugewiesen wurde.
                new_index = randint(1, N)
                if indice_server[new_index-1] == 0:
                    # Indize Werte + 1, damit 0 marker für unbelegt bleiben kann. Spätere Reudktion.
                    batch_indices[batch][index_pointer] = new_index  
                    indice_server[new_index-1] = 1
    # wieder um eins reduzieren
    for batch, arrays in enumerate(batch_indices):
        for index_pointer, array in enumerate(batch_indices[batch]):
            while batch_indices[batch][index_pointer] == 0:
                batch_indices[batch][index_pointer] -= 1 
    return batch_indices
        


def assign_batches(batches, batch_indices, data):
    """langsamer Prozess: optimierbar"""
    feature_depth = data.shape[1]
    for batch, arrays in enumerate(batch_indices):
        for observation_index, array in enumerate(batch_indices[batch]):
            for feature in range(feature_depth):
                batches[batch][observation_index][feature] = data[observation_index][feature]
                assert (isinstance(batches[batch][observation_index][feature], np.float64)), f"Falscher Datentyp mit {datentyp}"
    return batches


def train_model(x, y, model, n_classes, print_debug = True):
    """Neu: In jeder Epoche werden die Batches Random gemacht"""
    # Controll Variables for the NN
    epochs              = 20
    batch_size          = 20
    learning_rate       = 0.0002
    if(batch_size>len(x)/2): batch_size = len(x)

    # Data Preprocessing
    ## Sample X-y into Batches
    x = x[0:len(x) -len(x) % batch_size]        # Arraytrunkierung des Rests (Vereinfacht Algorithmus)
    y = y[0:len(y) -len(y) % batch_size]
    n_batches = int(len(x) / batch_size)
    if(print_debug == True): print(f"Length X = {len(x)}")
    batch_indices = np.zeros((n_batches, batch_size))

    ## Sample X-y into Batches
    x_batches = np.zeros((n_batches, batch_size, x.shape[1]))
    y_batches = np.zeros((n_batches, batch_size, y.shape[1]))
    print(f"Batch Indexing...")
    ## TENSOR Konstruktion für NN
    batch_indices = shuffle_indices(batch_indices, n_batches, len(x))
    print(f"x Batch Assignment by Indices...")
    x_batches = assign_batches(x_batches, batch_indices, x)
    print(x_batches.shape)
    print(f"y Batch Assignment by Indices...")
    y_batches = assign_batches(y_batches, batch_indices, y)
    x_batches_tensor = torch.tensor(x_batches, dtype=torch.float)
    y_batches_tensor = torch.tensor(y_batches, dtype=torch.float)

    
    # Model Training
    print(f"\n→ Initiating Modeltraining. Total Epochs: {epochs}")
    epoch_print = 0
    loss_fn = nn.CrossEntropyLoss() # Setup Loss Function
    torch.manual_seed(42)
    optim   = torch.optim.Adam(model.parameters(), lr=learning_rate)     # Setup optimizer
    for epoch in range(epochs):  
        if epoch == 0 or epochs < 10: 
            print_command = True
        else: 
            if (epoch % (epochs/20) == 0): 
                print_command = True
            else:
                print_command = False
        if print_command: print(f"{"↺": ^2} | Epoch: {epoch: ^4}     Advanced {int(epoch/epochs*100): ^4}%     ", end='')
              
        ## Sample X-y into Batches
        batch_indices = shuffle_indices(batch_indices, n_batches, len(x))
        x_batches_tensor = torch.tensor(assign_batches(x_batches, batch_indices, x), dtype=torch.float)
        y_batches_tensor = torch.tensor(assign_batches(y_batches, batch_indices, y), dtype=torch.float)
        model.train()
        for idx, x_batch in enumerate(x_batches_tensor):
            y_batch = y_batches_tensor[idx]
            y_pred  = model(x_batch)
            loss    = loss_fn(y_pred, y_batch)
            loss.backward()
            optim.step()
            optim.zero_grad()
        if print_command: 
            y_pred = model(x_batches_tensor[0])
            print(f"Batch[0] avg. loss: {loss_fn(y_pred, y_batches_tensor[0]):.4f}     ")
    
    
    print("Model has been Trained 💪")
    return model
    



def test_model(x, y, model):
    model.eval()
    y_pred = model(torch.tensor(x[0:10], dtype=torch.float))
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