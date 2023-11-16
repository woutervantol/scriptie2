import torch
import time
import numpy as np

class NeuralNetwork():
    def __init__(self, lr=0.001, batch_size=64):
        self.model = None
        self.optimizer = None
        self.lr = lr
        self.batch_size = batch_size
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

    def set_convolutional_model(self, in_channels=2):

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=(3, 3), stride=1, padding_mode="zeros"),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(76880, 1)
        )


    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self, data, nr_epochs=10, verbose=2):
        for epoch in range(nr_epochs):
            epoch_start = time.time()
            nr_batches = int(len(data.trainy)/self.batch_size)
            train_losses = 0
            for batch in range(nr_batches):
                batch_start = batch*self.batch_size
                batch_stop = (batch+1)*self.batch_size
                y_pred = self.model(torch.Tensor(data.trainx[batch_start:batch_stop])).squeeze(1)
                target = torch.Tensor(data.trainy[batch_start:batch_stop])
                loss = self.lossfn(y_pred, target)
                with torch.no_grad():
                    train_losses += loss
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
                    val_loss = self.lossfn(val_pred, val_true)
                    self.val_losses.append(val_loss)
                    print(f"Epoch: {epoch}, done in {time.time() - epoch_start:.2f} seconds")
                    print(f"Validation loss: {val_loss}. Train loss: {train_losses/nr_batches}")
                


    def test(self, data):
        prediction = self.model(torch.Tensor(data.testx[:64]))
        print(prediction[:10])
        print(data.testy[:10])



def MSE(pred, true):
    return (pred - true).square().mean()



def predict_mass_linear(l, band="low"):
    poly = np.poly1d(np.load(f"/home/tol/Documents/Thesis/models/linear_fit_{band}_6.npy"))
    return 10**poly(np.log10(l))