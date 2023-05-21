import torch

x = torch.randn(3, requires_grad=True)
print(x)
y = x+2 #creates computation graph y = x + 2
#since requires_grad = true, pytorch generates backward fn automatically
print(y) #prints grad fn also AddBackwards

z = y*y*2
z = z.mean()
print(z)

z.backward()  #dz/dx
print(x.grad)
#if requires_grad = False, then we won't be able to run z.backward()

v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32)
z.backward(v)   #we're passing v in the backward propogation
print(x.grad) #this will now work even if requires_grad = false

#Prevent torch from tracking gradients
''' 
- x.requires_grad_(False) # again, _ indicates changing the variable
- x.detach()
- with torch.no_grad():
'''

weights = torch.ones(4, requires_grad= True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() #will stop it gradients from being added at every epoch


