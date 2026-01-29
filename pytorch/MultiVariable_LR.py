import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
 

num_samples = 1000
num_features = 3 
X=torch.randn(num_samples,num_features)

true_weights = torch.tensor([2.0,-3.4,5.0])#一维的数组
true_bias = 10.0
y=torch.matmul(X,true_weights)+true_bias
y+=torch.randn(y.shape)*0.1
y=y.view(-1,1)#在这里，y变成[1000,1]的张量

print(f"Input shape:{X.shape}")
print(f"Target shape:{y.shape}")

class LinearRegressionModel(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear=nn.Linear(in_dim,out_dim)
        #w的形状：[out_dim,in_dim]
    
    def forward(self,x):
        out=self.linear(x)
        return out

model = LinearRegressionModel(in_dim=3,out_dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train=X.to(device)
y_train=y.to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

num_epochs=1000
for epoch in range(num_epochs):
    y_pred=model(X_train)
    loss = criterion(y_pred,y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1)%50==0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n===Training finished===")
with torch.no_grad():
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    print("\nTrue Parameters:")
    print(f"Weights: {true_weights}")
    print(f"Bias: {true_bias}")