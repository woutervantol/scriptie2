import torch
import torch.nn.functional as F
import time
import numpy as np
from ray import train
import json
from imports.utility import *
from imports.architectures import get_architecture

class customDataSet(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def save_best_model(p, m):
    name_addon = ""
    if p["simtype"] != "single":
        name_addon = "_"+p["simtype"]
    modelname = p_to_filename(p) + name_addon
    try:
        filepath = open(p['model_path']+modelname+".json", 'r')
        bestjson = json.load(filepath)
        bestvalloss = np.min(bestjson["vallosses"])
    except:
        with open(p['model_path'] + modelname + ".json", 'w') as filepath:
            json.dump(p, filepath, indent=4)
        bestvalloss = np.min(p["vallosses"])
    if p["vallosses"][-1] < bestvalloss:
        torch.save(m, p["model_path"] + modelname + ".pt")
        with open(p['model_path'] + modelname + ".json", 'w') as filepath:
            json.dump(p, filepath, indent=4)

def train_network(p, verbose=2, report=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p["architecture"] = get_architecture(p)
    model = CustomCNN(p["architecture"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["L2"])
    gamma = (1e-6/p["lr"])**(1./300.)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    lossfn = MSE

    _, _, trainx, trainy, valx, valy, _, _, _, _ = make_nn_dataset(p)
    trainloader = torch.utils.data.DataLoader(customDataSet(trainx, trainy), batch_size=p["batch_size"])
    valloader = torch.utils.data.DataLoader(customDataSet(valx, valy), batch_size=p["batch_size"])

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
        val_losses = 0
        for val_batch, datapairs in enumerate(valloader):
            valx, valy = datapairs
            with torch.no_grad():
                val_pred = model(valx.float().to(device)).squeeze(1)
                val_true = valy.to(device)
                val_loss = lossfn(val_pred, val_true)
                val_losses += np.float64(val_loss.cpu())
        scheduler.step()

        trainloss_hist.append(train_losses/nr_batches)
        valloss_hist.append(val_losses/nr_valbatches)
        lr_hist.append(np.float64(scheduler._last_lr[0]))
        p["trainlosses"] = trainloss_hist
        p["vallosses"] = valloss_hist
        p["lrs"] = lr_hist

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

    if report:
        # modelname = p_to_filename(p)
        # torch.save(model, "/home/s2041340/data1/checkpoints/" + modelname + time.gmtime() + ".pt")
        # train.report({"val loss": val_losses/nr_valbatches, "loss":train_losses/nr_batches})#, checkpoint=ray.train.Checkpoint.from_directory("/home/s2041340/data1/checkpoints/"))
        pass
    else:
        return model, p

def make_nn_dataset(p):
    filename =  p_to_filename(p)
    testx = np.empty((0, 2, p["resolution"], p["resolution"]))
    testy = np.empty((0))
    valx = np.empty((0, 2, p["resolution"], p["resolution"]))
    valy = np.empty((0))
    trainx = np.empty((0, 2, p["resolution"], p["resolution"]))
    trainy = np.empty((0))
    if p["simtype"] == "single":
        data_x = np.load(p["data_path"] + filename + ".npy")
        data_y = np.load(p["data_path"] + filename + "_masses.npy")
        test_split = int(len(data_y)*p["test_size"])
        val_split = test_split + int(len(data_y)*p["val_size"])
        testx = data_x[:test_split]
        testy = data_y[:test_split]
        valx = data_x[test_split:val_split]
        valy = data_y[test_split:val_split]
        trainx = data_x[val_split:]
        trainy = data_y[val_split:]
    elif p["simtype"] == "extremes":
        for model in ["HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
            p_temp = p.copy()
            p_temp["model"] = model
            filename = p_to_filename(p_temp)
            data_x = np.load(p["data_path"] + filename + ".npy")
            data_y = np.load(p["data_path"] + filename + "_masses.npy")
            test_split = int(len(data_y)*p["test_size"])
            val_split = test_split + int(len(data_y)*p["val_size"])
            testx = np.append(testx, data_x[:test_split], axis=0)
            testy = np.append(testy, data_y[:test_split], axis=0)
            valx = np.append(valx, data_x[test_split:val_split], axis=0)
            valy = np.append(valy, data_y[test_split:val_split], axis=0)
            trainx = np.append(trainx, data_x[val_split:], axis=0)
            trainy = np.append(trainy, data_y[val_split:], axis=0)
    else:
        for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
            if p["simtype"] == "all_but" and p["model"] == model:
                continue
            p_temp = p.copy()
            p_temp["model"] = model
            filename = p_to_filename(p_temp)
            data_x = np.load(p["data_path"] + filename + ".npy")
            data_y = np.load(p["data_path"] + filename + "_masses.npy")
            test_split = int(len(data_y)*p["test_size"])
            val_split = test_split + int(len(data_y)*p["val_size"])
            testx = np.append(testx, data_x[:test_split], axis=0)
            testy = np.append(testy, data_y[:test_split], axis=0)
            valx = np.append(valx, data_x[test_split:val_split], axis=0)
            valy = np.append(valy, data_y[test_split:val_split], axis=0)
            trainx = np.append(trainx, data_x[val_split:], axis=0)
            trainy = np.append(trainy, data_y[val_split:], axis=0)
    # data_x = np.empty((0, 2, p["resolution"], p["resolution"]))
    # data_y = np.empty((0))
    # if p["simtype"] == "single":
    #     data_x = np.load(p["data_path"] + filename + ".npy")
    #     data_y = np.load(p["data_path"] + filename + "_masses.npy")
    # elif p["simtype"] == "extremes":
    #     for model in ["HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
    #         p_temp = p.copy()
    #         p_temp["model"] = model
    #         filename = p_to_filename(p_temp)
    #         data_x = np.append(data_x, np.load(p["data_path"] + filename + ".npy"), axis=0)
    #         data_y = np.append(data_y, np.load(p["data_path"] + filename + "_masses.npy"), axis=0)
    #         print(f"{model} loaded", flush=True)
    # else:
    #     for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
    #         if p["simtype"] == "all_but" and p["model"] == model:
    #             continue
    #         p_temp = p.copy()
    #         p_temp["model"] = model
    #         filename = p_to_filename(p_temp)
    #         data_x = np.append(data_x, np.load(p["data_path"] + filename + ".npy"), axis=0)
    #         data_y = np.append(data_y, np.load(p["data_path"] + filename + "_masses.npy"), axis=0)
    #         print(f"{model} loaded", flush=True)


    shuffled_indices = np.arange(len(testy))
    np.random.shuffle(shuffled_indices)
    testx = testx[shuffled_indices]
    testy = testy[shuffled_indices]
    shuffled_indices = np.arange(len(valy))
    np.random.shuffle(shuffled_indices)
    valx = valx[shuffled_indices]
    valy = valy[shuffled_indices]
    shuffled_indices = np.arange(len(trainy))
    np.random.shuffle(shuffled_indices)
    trainx = trainx[shuffled_indices]
    trainy = trainy[shuffled_indices]

    testx = np.log10(testx)
    testy = np.log10(testy)
    valx = np.log10(valx)
    valy = np.log10(valy)
    trainx = np.log10(trainx)
    trainy = np.log10(trainy)

    std_x = np.std(trainx, axis=(0, 2, 3))
    std_y = np.std(trainy)
    mean_x = np.mean(trainx, axis=(0, 2, 3))
    mean_y = np.mean(trainy)

    #scale and shift data for better nn training
    testx = (testx - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis]
    testy = (testy - mean_y) / std_y
    valx = (valx - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis]
    valy = (valy - mean_y) / std_y
    trainx = (trainx - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis]
    trainy = (trainy - mean_y) / std_y

    #Select the correct image for single channel runs
    if p["channel"]=="low": 
        testx = testx[:,:1,:,:]
        valx = valx[:,:1,:,:]
        trainx = trainx[:,:1,:,:]
    elif p["channel"]=="high": 
        testx = testx[:,1:,:,:]
        valx = valx[:,1:,:,:]
        trainx = trainx[:,1:,:,:]
    else:
        pass

    # test_split = int(len(data_y)*p["test_size"])
    # val_split = test_split + int(len(data_y)*p["val_size"])
    # testx = data_x[:test_split]
    # testy = data_y[:test_split]
    # valx = data_x[test_split:val_split]
    # valy = data_y[test_split:val_split]
    # trainx = data_x[val_split:]
    # trainy = data_y[val_split:]

    return testx, testy, trainx, trainy, valx, valy, mean_x, mean_y, std_x, std_y

def MSE(pred, true):
    return (pred - true).square().mean()

def RMSE(pred, true):
    return (pred - true).square().mean().sqrt()

class CustomCNN(torch.nn.Module):
    def __init__(self, architecture):
        super(CustomCNN, self).__init__()

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
        # skip_connections = []
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            # if layer_idx in self.skip_connection_indices:
            #     skip_connections.append(x)
        return x#, skip_connections


