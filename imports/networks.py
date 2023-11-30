import torch
import torch.nn.functional as F
import time
import numpy as np



class Model():
    def __init__(self, p):
        self.p = p
        self.model = None
        self.optimizer = None
        self.lr = p["lr"]
        self.batch_size = p["batch_size"]
        self.lossfn = MSE
        self.epochs = []
        self.losses = []
        self.val_losses = []


    def set_linear_model(self, nr_inputs):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(nr_inputs, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )

    def set_convolutional_model(self):
        self.model = CustomCNN(self.p["architecture"])


    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self, data, verbose=2):
        for epoch in range(self.p["nr_epochs"]):
            epoch_start = time.time()
            nr_batches = int(len(data.trainy)/self.batch_size)
            trainx, trainy = self.shuffle(data.trainx, data.trainy)
            train_losses = 0
            for batch in range(nr_batches):
                batch_start = batch*self.batch_size
                batch_stop = (batch+1)*self.batch_size
                y_pred = self.model(torch.Tensor(trainx[batch_start:batch_stop])).squeeze(1)
                target = torch.Tensor(trainy[batch_start:batch_stop])
                loss = self.lossfn(y_pred, target)
                with torch.no_grad():
                    train_losses += np.float64(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.epochs.append(epoch)
            self.losses.append(train_losses/nr_batches)


            if verbose == 0:
                pass
            elif verbose == 1:
                print(f"Epoch: {epoch}, done in {time.time() - epoch_start:.2f} seconds")
            elif verbose == 2:
                with torch.no_grad():
                    val_pred = self.model(torch.Tensor(data.valx)).squeeze(1)
                    val_true = torch.Tensor(data.valy)
                    val_loss = np.float64(self.lossfn(val_pred, val_true))
                    self.val_losses.append(val_loss)
                    print(f"Epoch: {epoch}, done in {time.time() - epoch_start:.2f} seconds")
                    print(f"Validation loss: {val_loss}. Train loss: {train_losses/nr_batches}")
        
    def shuffle(self, datax, datay):
        indices = np.arange(len(datay))
        np.random.shuffle(indices)
        return datax[indices], datay[indices]


    def test(self, data):
        prediction = self.model(torch.Tensor(data.testx[:64]))
        print(prediction[:10])
        print(data.testy[:10])



def MSE(pred, true):
    return (pred - true).square().mean()



def predict_mass_linear(l, band="low"):
    poly = np.poly1d(np.load(f"/home/tol/Documents/Thesis/models/linear_fit_{band}_6.npy"))
    return 10**poly(np.log10(l))





# class NeuralNetwork(torch.nn.Module):
#     def __init__(self, p, in_channels=2):
#         super().__init__()
#         res = p['resolution']
#         hidden_channels = 20
#         kernel = 3

#         self.conv1 = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel, padding=int(kernel/2), padding_mode="zeros")
#         self.bn1 = torch.nn.BatchNorm2d(hidden_channels)
#         self.conv2 = torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel, padding=int(kernel/2), padding_mode="zeros")
#         self.bn2 = torch.nn.BatchNorm2d(hidden_channels)
#         self.flatten = torch.nn.Flatten()
#         self.out = torch.nn.Linear(res*res*hidden_channels, 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         # x = self.bn1(x)
#         x = F.relu(x)

#         x = self.conv2(x)
#         # x = self.bn2(x)
#         x = F.relu(x)
        
#         x = self.flatten(x)
#         x = self.out(x)
#         return x




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
                layers.append(torch.nn.MaxPool2d(l['kernel_size'], l['stride'], l['padding']))

            elif layer_type == 'fc':
                layers.append(torch.nn.Linear(l['in_features'], l['out_features']))

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


