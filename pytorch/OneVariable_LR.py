import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device:{device}")

x_values=np.linspace(0,10,100).astype(np.float32).reshape(-1,1)
y_values=3*x_values+0.8+np.random.randn(100,1)*1.5

x_train=torch.from_numpy(x_values).to(device).float()
y_train=torch.from_numpy(y_values).to(device).float()

class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear=nn.Linear(input_dim,output_dim)

    def forward(self,x):
        out=self.linear(x)
        return out

input_dim=1
output_dim=1
model=LinearRegressionModel(input_dim,output_dim).to(device)

learning_rate = 0.005
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
epochs = 5000
for epoch in range(epochs):
    outputs = model(x_train)
    loss =criterion(outputs,y_train)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if(epoch+1)%50 == 0 :
        print(f"Epoch[{epoch+1}/{epochs}],loss:{loss.item():.4f}")

model.eval()
with torch.no_grad(): # 预测阶段不需要计算梯度
    predicted = model(x_train).cpu().data.numpy()

# 打印最终学到的参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"学到的 {name}: {param.data}")

# 可视化
plt.scatter(x_values, y_values, label='Original Data')
plt.plot(x_values, predicted, color='red', label='Fitted Line')
plt.legend()
plt.show()

    