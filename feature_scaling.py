import torch
import torch.nn as nn
import torch.optim as optim

# ================= 1. 准备数据 =================
# 3个样本，2个特征 (面积, 房龄)
# Y 是房价 (万元)
X_train_raw = torch.tensor([[100.0, 10.0], [150.0, 5.0], [80.0, 20.0]])
y_train = torch.tensor([[500.0], [800.0], [400.0]]) # 标签通常不归一化，除非数值极大

# ================= 2. 预处理 (Pre-processing) =================
# 计算并保存统计量 (相当于保存了 C++ 的 config)
# dim=0: 计算每一列的均值和方差
train_mean = X_train_raw.mean(dim=0)
train_std = X_train_raw.std(dim=0)

# 执行归一化
X_train_norm = (X_train_raw - train_mean) / train_std

print(f"统计量已保存: Mean={train_mean}, Std={train_std}")

# ================= 3. 训练模型 =================
model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1) # 归一化后 LR 可以大一点

for epoch in range(100):
    # 【重点】喂给模型的是 X_train_norm，但标签依然是原始 y_train
    y_pred = model(X_train_norm)
    loss = nn.MSELoss()(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ================= 4. 推理/预测 (Inference) =================
# 假设来了一个新房子：120平米，8年房龄
X_new_raw = torch.tensor([[120.0, 8.0]])

# 【关键步骤】必须用训练集的统计量进行同样的变换
# 绝对不能用 X_new_raw.mean()，因为这就一个样本，方差是 NaN
X_new_norm = (X_new_raw - train_mean) / train_std

# 喂给模型
with torch.no_grad():
    y_predict = model(X_new_norm)

# 【结论】因为我们没有归一化 y_train，所以出来的直接就是房价
print(f"预测房价: {y_predict.item():.2f} 万元")