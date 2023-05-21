#this is implementation of linear regression using pytorch

import torch

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

#here if we want to write y as w*x, w = 2

w = torch.tensor(0.0, dtype = torch.float32, requires_grad= True) #initializing

#Model prediction
def forward(x):
    return w*x

#loss = MSE(Mean Squared Error)

def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#gradient
'''def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()''' #Pytorch does this part for you

print(f'Prediction before training: f(5) = {forward(5):.3f}') #using the forward function to make a prediction for the input value of 5. The result is then formatted to display with three decimal places



#Training
learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
    y_pred = forward(X)
    
    l = loss(Y, y_pred)
    
    #gradient = basically backward pass/ back propogation
    l.backward() #dl/dw
    
    with torch.no_grad():           #you do this because you don't want what is inside to affect your gradient calculation
        w -= learning_rate * w.grad
    
    w.grad.zero_()
        
    if epoch % 10 == 0:          # %1 matlab every epoch. if we have more eppchs, we can write %10 and shit so the program doesnt have to print something for each epoch
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')  #print syntax you weren't familiar with. if u forget, chatGPT it. simple hi hai
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')    
        
        




