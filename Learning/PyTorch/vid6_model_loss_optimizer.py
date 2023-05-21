#this is implementation of linear regression using pytorch

# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training Loop
#   -forward pass: compute prediction
#   -backward pass: gradients
#   -update weights

import torch
import torch.nn as nn


#LMAO convert all to 2d arrays
X = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype = torch.float32)

X_test = torch.tensor([5], dtype = torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

#Model banao
model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)   
    
     
model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}') #using the forward function to make a prediction for the input value of 5. The result is then formatted to display with three decimal places

#Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss() #this is mean square error loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   #Stochastic Gradient Descent

for epoch in range(n_iters):
    y_pred = model(X)
    
    l = loss(Y, y_pred)
    
    #gradient = basically backward pass/ back propogation
    l.backward() #dl/dw
    
    optimizer.step() #updates weights
    
    optimizer.zero_grad()
        
    if epoch % 10 == 0:          # %1 matlab every epoch. if we have more eppchs, we can write %10 and shit so the program doesnt have to print something for each epoch
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')  #print syntax you weren't familiar with. if u forget, chatGPT it. simple hi hai
        
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')    
        
        




