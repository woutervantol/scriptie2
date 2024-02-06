import torch
import torch.nn.functional as F
import time
import numpy as np


class Model():
    def __init__(self, p):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = p
        self.model = None
        self.optimizer = None
        self.lr = p["lr"]
        self.batch_size = p["batch_size"]
        self.lossfn = MSE
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.lrs = []


    def set_linear_model(self, nr_inputs):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(nr_inputs, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )

    def set_convolutional_model(self):
        self.model = CustomCNN(self.p["architecture"])
        self.model = self.model.to(self.device)


    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.p["L2"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.p["lrfactor"], patience=self.p["lrpatience"])


    def train(self, data="", verbose=2):
        for epoch in range(self.p["nr_epochs"]):
            epoch_start = time.time()
            nr_batches = int(len(data.trainy)/self.batch_size)
            trainx, trainy = self.shuffle(data.trainx, data.trainy)
            train_losses = 0
            for batch in range(nr_batches):
                batch_start = batch*self.batch_size
                batch_stop = (batch+1)*self.batch_size
                y_pred = self.model(torch.Tensor(trainx[batch_start:batch_stop]).to(self.device)).squeeze(1)
                target = torch.Tensor(trainy[batch_start:batch_stop]).to(self.device)
                loss = self.lossfn(y_pred, target)
                with torch.no_grad():
                    train_losses += np.float64(loss.cpu())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                val_pred = self.model(torch.Tensor(data.valx).to(self.device)).squeeze(1)
                val_true = torch.Tensor(data.valy).to(self.device)
                val_loss = np.float64(self.lossfn(val_pred, val_true).cpu())
                self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            
            self.epochs.append(epoch)
            self.losses.append(train_losses/nr_batches)
            self.lrs.append(self.scheduler._last_lr[0])


            if verbose == 0:
                pass
            elif verbose == 1:
                print(f"Epoch: {epoch}, done in {time.time() - epoch_start:.2f} seconds")
            elif verbose == 2:
                    print(f"Epoch: {epoch}, done in {time.time() - epoch_start:.2f} seconds")
                    print(f"Validation loss: {val_loss}. Train loss: {train_losses/nr_batches}")
        
    
    
    def shuffle(self, datax, datay):
        indices = np.arange(len(datay))
        np.random.shuffle(indices)
        return datax[indices], datay[indices]

def MSE(pred, true):
    return (pred - true).square().mean()


# def ray_train(config):
#         for epoch in range(self.p["nr_epochs"]):
#             epoch_start = time.time()
#             print(self.batch_size)
#             nr_batches = int(len(data.trainy)/self.batch_size)
#             trainx, trainy = self.shuffle(data.trainx, data.trainy)
#             train_losses = 0
#             for batch in range(nr_batches):
#                 batch_start = batch*self.batch_size
#                 batch_stop = (batch+1)*self.batch_size
#                 y_pred = self.model(torch.Tensor(trainx[batch_start:batch_stop]).to(self.device)).squeeze(1)
#                 target = torch.Tensor(trainy[batch_start:batch_stop]).to(self.device)
#                 loss = self.lossfn(y_pred, target)
#                 with torch.no_grad():
#                     train_losses += np.float64(loss.cpu())
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#             with torch.no_grad():
#                 val_pred = self.model(torch.Tensor(data.valx).to(self.device)).squeeze(1)
#                 val_true = torch.Tensor(data.valy).to(self.device)
#                 val_loss = np.float64(self.lossfn(val_pred, val_true).cpu())
#                 self.val_losses.append(val_loss)
#             self.scheduler.step(val_loss)
            
#             self.epochs.append(epoch)
#             self.losses.append(train_losses/nr_batches)
#             self.lrs.append(self.scheduler._last_lr[0])


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
                layers.append(torch.nn.BatchNorm1d(l['nr_features']))

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


