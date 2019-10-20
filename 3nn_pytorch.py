# Developed by Brad Cho
# This is 3-layer Neural Network built with Pytorch
# This will classify microRNA as positive, and pseudo-hairpin RNA as negative using 48 features of RNA
# 48 features are of three different categories of data: 1). Sequential, 2). 2-D structural, 3). Thermodynamic
# This is built as a part of my graduate research in which I focused on applying deep learning on microRNA for classification
# Input data consist of 1384 RNA sequences (691 positive, 693 negative)

import torch
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
from torch.utils import data
import matplotlib.pyplot as plt

# reading data saved in csv file
inputs = pd.read_csv("48-features_inputs.csv", header=None)
label = pd.read_csv("48-features_labels.csv", header=None)


# input dimension 1383 x 48, label dimension 1383
# input data converted into tensors
inputs = torch.Tensor(inputs.values)
label = torch.Tensor(label.values)
label = label.view(-1)
label = label.long()

# randomizing the orders before training
randomize = torch.randperm(len(inputs))
inputs = inputs[randomize]
label = label[randomize]

# train : validation : test ratio = 70 : 15 : 15
in1 = int(len(inputs)*0.7)
in2 = int(len(inputs)*0.15)

#split data into train, validation, and test
inputs_train = inputs[:in1]
inputs_val = inputs[in1:in1+in2]
inputs_test = inputs[in1+in2:]

label_train = label[:in1]
label_val = label[in1:in1+in2]
label_test = label[in1+in2:]

# created dictionary objects for preparing mini batches for training
batch_inputs_train = dict()
batch_inputs_val = dict()
batch_inputs_test = dict()

batch_label_train = dict()
batch_label_val = dict()
batch_label_test = dict()

# 24 mini batches of input data
b = 24
n_batch_training = int(len(inputs_train)/b)
n_batch_valtes = int(len(inputs_val)/b)

# adding to training, validation, test mini-batches
for i in range(n_batch_training):
    if i < n_batch_training-1:
        batch_inputs_train[i] = inputs_train[i * b : (i+1) * b]
        batch_label_train[i] = label_train[i * b : (i+1) * b].long()
    if i == n_batch_training-1:
        batch_inputs_train[i] = inputs_train[i * b :]
        batch_label_train[i] = label_train[i * b :].long()
for j in range(n_batch_valtes):
    if j < n_batch_valtes-1:
        batch_inputs_val[j] = inputs_val[j * b : (j+1) * b]
        batch_inputs_test[j] = inputs_test[j * b : (j+1) * b]
        batch_label_val[j] = label_val[j * b : (j+1) * b].long()
        batch_label_test[j] = label_test[j * b : (j+1) * b].long()
    if j == n_batch_valtes-1:
        batch_inputs_val[j] = inputs_val[j * b :]
        batch_inputs_test[j] = inputs_test[j * b :]
        batch_label_val[j] = label_val[j * b :].long()
        batch_label_test[j] = label_test[j * b :].long()


class Classifier_3nn(nn.Module):
    #Neural Network classifier class
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(48,128) # 48 features input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,2)

        #drop-out module with 0.2 probability, doesn't have to be used but can be if needed
        self.dropout = nn.Dropout(p=0.2)

    #Forward propagation
    def forward(self, x):
        ############################################
        # use if flattening is needed
        # x = x.view(x.shape[0], -1)
        ############################################

        ############################################
        # use if using dropout
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))
        # x = F.log_softmax(self.fc5(x),dim=1)
        ############################################

        # 3 fully connected layers with softmax in output layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

# this is a function to actually execute the training, validating, testing the model
# accepts two parameters when called: learning rate, and weight decay
def classify(learning,decay):
    model = Classifier_3nn()
    # used for validation
    criterion = nn.NLLLoss()
    # Adam used for optimization of the model
    optimizer = optim.Adam(model.parameters(), lr=learning,weight_decay=decay)
    epochs = 45
    train_losses, val_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for i in range(len(batch_inputs_train)):
            optimizer.zero_grad()
            log_prob = model(batch_inputs_train[i])
            # Forward pass, get our logits
            loss = criterion(log_prob, batch_label_train[i])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            # validation and accuracy needes to be set to zero at new epoch before validation and test
            val_loss = 0
            accuracy = 0

            # need to use model.eval() and model.train() if you are using drop out.
            # this is necessary for turning off drop-out for evaluation, and test
            with torch.no_grad():
                #set model to evaluation mode which disables dropout
                #model.eval()
                for j in range(len(batch_inputs_test)):
                    #validation set
                    log_ps_val = model(batch_inputs_val[j])
                    val_loss += criterion(log_ps_val, batch_label_val[j])

                    #test set
                    log_ps_test = model(batch_inputs_test[j])
                    ps = torch.exp(log_ps_test)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == batch_label_test[j].view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            #set the model back to train mode, enables dropout
            #model.train()

            train_losses.append(running_loss / len(batch_inputs_train))
            val_losses.append(val_loss / len(batch_inputs_val))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Validation Loss: {:.3f}.. ".format(val_losses[-1]),
                  "Test Accuracy: {:.3f}".format(accuracy / len(batch_inputs_test)))

# training performance plot: Training Loss and Validation Loss

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

# training with learning rate: 3e-4, and weight decay: 0
classify(3e-4, 0)
