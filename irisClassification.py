import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in the Iris dataset available for download at https://jovian.ai/outlink?url=https%3A%2F%2Farchive.ics.uci.edu%2Fml%2Fmachine-learning-databases%2Firis%2Firis.data
dataset = pd.read_csv("iris.data")

dataset.columns = ["sepal length (cm)", 
                   "sepal width (cm)", 
                   "petal length (cm)", 
                   "petal width (cm)", 
                   "species"]

# Remap the species data into a numerical value
mappings = {
   "Iris-setosa": 0,
   "Iris-versicolor": 1,
   "Iris-virginica": 2
}
dataset["species"] = dataset["species"].apply(lambda x: mappings[x])

# Separate dataset into inputs and outputs
X = dataset.drop("species",axis=1).values
y = dataset["species"].values

# Use sklearn to split the data into a training and testing set and turn into tensors
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Initialize the ReLU activated, fully connected neural network and optimization parameters
class NeuralNet(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=20, hidden_layer2=20, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hidden_layer1)                  
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)                  
        self.out = nn.Linear(hidden_layer2, output_features)      
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
model = NeuralNet()

lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
losses = []

# Train the model using the training set
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = lossFunc(y_pred, y_train)
    losses.append(loss.detach())
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad() #zero out the gradient buffer because someone designed backpropogation to accumulates for some reason
    loss.backward()
    optimizer.step()
    
# Optional plotting of the training period
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');

# Evaluate the model 
preds = []
with torch.no_grad():
    for val in X_test:
        y_hat = model.forward(val)
        preds.append(y_hat.argmax().item())
preds = torch.tensor(preds)
accuracy = torch.sum(torch.eq(y_test,preds)).item()/y_test.nelement()

print(f'The Iris classification model has a {100*accuracy:0.3f} percent accuracy')