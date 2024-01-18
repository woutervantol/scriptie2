import numpy as np
import torch
from imports.utility import *
from ray import train

def ray_train(config, net, p):
    trainx, trainy, valx, valy = ray_get_data(p)
    model = net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["L2"])
    for epoch in range(config["nr_epochs"]):
        nr_batches = int(len(trainy)/config["batch_size"])
        trainx, trainy = ray_shuffle(trainx, trainy)
        train_losses = 0
        for batch in range(nr_batches):
            batch_start = batch*config["batch_size"]
            batch_stop = (batch+1)*config["batch_size"]
            y_pred = model(torch.Tensor(trainx[batch_start:batch_stop]).to(device)).squeeze(1)
            target = torch.Tensor(trainy[batch_start:batch_stop]).to(device)
            loss = ray_MSE(y_pred, target)
            with torch.no_grad():
                train_losses += np.float64(loss.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        with torch.no_grad():
            val_pred = model(torch.Tensor(valx).to(device)).squeeze(1)
            val_true = torch.Tensor(valy).to(device)
            val_loss = np.float64(ray_MSE(val_pred, val_true).cpu())
            train.report({"val loss": val_loss, "loss":loss})

        # self.scheduler.step(val_loss)
        
        # self.epochs.append(epoch)
        # self.losses.append(train_losses/nr_batches)
        # self.lrs.append(self.scheduler._last_lr[0])
        # print(f"Epoch {epoch} finished. Current val loss: {val_loss}")

def ray_shuffle(datax, datay):
    indices = np.arange(len(datay))
    np.random.shuffle(indices)
    return datax[indices], datay[indices]

def ray_MSE(pred, true):
    return (pred - true).square().mean()


def ray_get_data(p):
    filename =  p_to_filename(p) + "big2"
    data_x = np.load(p["data_path"] + filename + ".npy")
    data_y = np.load(p["data_path"] + filename + "_masses.npy")

    data_x = np.log10(data_x)
    data_y = np.log10(data_y)
    std_x = np.std(data_x, axis=(0, 2, 3))
    std_y = np.std(data_y)
    mean_x = np.mean(data_x, axis=(0, 2, 3))
    mean_y = np.mean(data_y)

    #scale and shift data for better nn training
    data_x = (data_x - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis]
    data_y = (data_y - mean_y) / std_y
    
    #Select the correct image for single channel runs
    if p["channel"]=="low": 
        data_x = data_x[:,:1,:,:]
    elif p["channel"]=="high": 
        data_x = data_x[:,1:,:,:]
    else:
        pass

    test_split = int(len(data_y)*p["test_size"])
    val_split = test_split + int(len(data_y)*p["val_size"])
    valx = data_x[test_split:val_split]
    valy = data_y[test_split:val_split]
    trainx = data_x[val_split:]
    trainy = data_y[val_split:]
    
    return trainx, trainy, valx, valy