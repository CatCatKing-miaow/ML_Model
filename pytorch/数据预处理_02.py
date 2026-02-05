import pandas as pd
import torch
data = pd.read_csv('./data/house_tiny.csv')
print(data)

inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

#对于 inputs 中的类别值或离散值，我们将'NaN'视为一个类别，即dummy_na=True
inputs = pd.get_dummies(inputs,dummy_na=True,dtype=float)
print(inputs)

x,y=torch.tensor(inputs.values,dtype=torch.float32),torch.tensor(outputs.values,dtype=torch.float32)
#在创建tensor类型时再指定数据类型float32。最稳妥
print(x,y)