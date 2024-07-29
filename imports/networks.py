import torch
import torch.nn.functional as F
import time
import numpy as np
from ray import train
import json
from imports.utility import *
from imports.architectures import get_architecture

### pytorch dataset object which automatically shuffles and divides into batches
class customDataSet(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def save_best_model(p, m):
    """Saves the current model if the validation loss is better than the saved one."""
    name_addon = ""
    if p["simtype"] != "single":
        name_addon += "_"+p["simtype"]
    if p["noisy"] == True:
        name_addon += "_"+"noisy"
    if p["channel"] != "2chan":
        name_addon += "_"+p["channel"]
    modelname = p_to_filename(p) + name_addon
    
    ### see if any saved model already exists
    try:
        filepath = open(p['model_path']+modelname+".json", 'r')
        bestjson = json.load(filepath)
        bestvalloss = np.min(bestjson["vallosses"])
    except:
        with open(p['model_path'] + modelname + ".json", 'w') as filepath:
            json.dump(p, filepath, indent=4)
        bestvalloss = np.min(p["vallosses"])
    ### if this model is an improvement, save model and parameters
    if p["vallosses"][-1] < bestvalloss:
        torch.save(m, p["model_path"] + modelname + ".pt")
        with open(p['model_path'] + modelname + ".json", 'w') as filepath:
            json.dump(p, filepath, indent=4)

def train_network(p, verbose=2, report=False):
    """Main training loop. Includes loading, training, saving and reporting."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### load model architecture given the hyperparameters in p 
    p["architecture"] = get_architecture(p)
    model = CustomCNN(p["architecture"]).to(device)

    ### load optimizer and set learning rate annealing factor such that it is 1e-6 after 300 epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["L2"])
    gamma = (1e-6/p["lr"])**(1./300.)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    
    ### set loss function
    lossfn = RMSE

    ### load training and validation data and create pytorch dataset objects
    _, _, trainx, trainy, valx, valy, _, _, _, _ = load_nn_dataset(p)
    trainx = np.concatenate(trainx)
    trainy = np.concatenate(trainy)
    valx = np.concatenate(valx)
    valy = np.concatenate(valy)
    trainloader = torch.utils.data.DataLoader(customDataSet(trainx, trainy), batch_size=p["batch_size"])
    valloader = torch.utils.data.DataLoader(customDataSet(valx, valy), batch_size=p["batch_size"])

    ### main loop
    trainloss_hist = []
    valloss_hist = []
    lr_hist = []
    nr_batches = len(trainloader)
    nr_valbatches = len(valloader)
    for epoch in range(p["nr_epochs"]):
        epoch_start = time.time()
        train_losses = 0
        for batch, datapairs in enumerate(trainloader):
            trainx, trainy = datapairs
            y_pred = model(trainx.float().to(device)).squeeze(1)
            y_true = trainy.to(device)
            loss = lossfn(y_pred, y_true)
            with torch.no_grad():
                train_losses += np.float64(loss.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ### validation must also be looped to prevent memory overflow
        val_losses = 0
        for val_batch, datapairs in enumerate(valloader):
            valx, valy = datapairs
            with torch.no_grad():
                val_pred = model(valx.float().to(device)).squeeze(1)
                val_true = valy.to(device)
                val_loss = lossfn(val_pred, val_true)
                val_losses += np.float64(val_loss.cpu())
        
        ### update learning rate
        scheduler.step()

        ### save losses and lr
        trainloss_hist.append(train_losses/nr_batches)
        valloss_hist.append(val_losses/nr_valbatches)
        lr_hist.append(np.float64(scheduler._last_lr[0]))
        p["trainlosses"] = trainloss_hist
        p["vallosses"] = valloss_hist
        p["lrs"] = lr_hist

        ### When using ray tune we must report, which also saves if the model has improved
        if report:
            save_best_model(p, model)
            train.report({"val loss": val_losses/nr_valbatches, "loss":train_losses/nr_batches})#, checkpoint=ray.train.Checkpoint.from_directory("/home/s2041340/data1/checkpoints/"))
        
        if verbose == 0:
            pass
        elif verbose == 1:
            print(f"Epoch: {epoch}, done in {time.time() - epoch_start:.2f} seconds")
        elif verbose == 2:
                print(f"Epoch: {epoch}, done in {time.time() - epoch_start:.2f} seconds")
                print(f"Validation loss: {val_losses/nr_valbatches}. Train loss: {train_losses/nr_batches}", flush=True)

    return model, p

def MSE(pred, true):
    return (pred - true).square().mean()

def RMSE(pred, true):
    return (pred - true).square().mean().sqrt()


class CustomCNN(torch.nn.Module):
    """Creates pytorch model object from list of dictionaries with all layers and parameters"""
    def __init__(self, architecture):
        super(CustomCNN, self).__init__()
        ### Note: skip-connections not used in project
        self.layers, self.skip_connection_indices = self._build_layers(architecture)

    def _build_layers(self, layer_configs):
        layers = torch.nn.ModuleList()
        skip_connection_indices = []

        for idx, l in enumerate(layer_configs):
            layer_type = l['type']

            if layer_type == 'conv':
                layers.append(torch.nn.Conv2d(l['in_channels'], l['out_channels'], l['kernel_size'], l['stride'], l['padding']))

                # Check if there's a skip connection
                if 'skip_connection' in l and l['skip_connection']:
                    skip_connection_indices.append(idx)

            elif layer_type == 'pool':
                layers.append(torch.nn.MaxPool2d(l['kernel_size'], stride=l['stride'], padding=l['padding']))

            elif layer_type == 'fc':
                layers.append(torch.nn.Linear(l['in_features'], l['out_features']))

            elif layer_type == 'batch_norm':
                layers.append(torch.nn.BatchNorm2d(l['nr_features'], l['momentum']))
            
            elif layer_type == 'batch_norm_1d':
                layers.append(torch.nn.BatchNorm1d(l['nr_features'], l['momentum']))

            elif layer_type == 'relu':
                layers.append(torch.nn.ReLU())

            elif layer_type == 'leaky_relu':
                layers.append(torch.nn.LeakyReLU(l["slope"]))    
            
            elif layer_type == 'flatten':
                layers.append(torch.nn.Flatten())
            
            elif layer_type == 'dropout':
                layers.append(torch.nn.Dropout(l["p"]))

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        return layers, skip_connection_indices

    def forward(self, x):
        skip_connections = []
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            if layer_idx in self.skip_connection_indices:
                skip_connections.append(x)
        return x#, skip_connections


