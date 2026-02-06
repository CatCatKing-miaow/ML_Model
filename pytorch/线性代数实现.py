import torch
# x=torch.tensor([3.0])
# y=torch.tensor([2.0])
# print(x+y)
# print(x*y)
# print(x/y)
# print(x**y)

# z=torch.arange(4)
# print(z)
# print(z[3])
# print(len(z))
# print(z.shape)

# A =torch.arange(20).reshape(5,4)#创建矩阵A
# print(A)
# print(A.T)#转置

# #对称矩阵：等于自身转置
# B=torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
# print(B==B.T)

# #可以构建具有更多轴的数据结构
# X=torch.arange(24).reshape(2,3,4)
# print(X)

# #相同形状的任何两个张量，任何按元素二元运算的结果形状不变
# A=torch.arange(20,dtype=torch.float32).reshape(5,4)
# B=A.clone()
# print(A)
# print(A+B)

# #两个矩阵按元素的乘法称作哈达玛积
# print(A*B)

# #标量与矩阵的运算
# a=2
# X=torch.arange(24).reshape(2,3,4)
# print(a+X,(a*X).shape)

# #计算元素和
# x=torch.arange(4,dtype=torch.float32)
# print(x,x.sum())

#表示任意形状张量的元素和
A=torch.arange(20*2).reshape(2,5,4)
print(A)
print(A.shape,A.sum())
#指定维度求和
A_sum_axis0=A.sum(axis=0)#第一维度
print(A_sum_axis0,A_sum_axis0.shape)
A_sum_axis1=A.sum(axis=1)#第二维度
print(A_sum_axis1,A_sum_axis1.shape)
#对多维度求和
print(A.sum(axis=[0,1]).shape)

#一个与求和相关的量是平均值（mean）

#计算总和或均值时保持轴数不变
sum_A=A.sum(axis=1,keepdims=True)
print(sum_A)
#通过广播将A除以sum_A
print(A/sum_A)
#累加求和
print(A.cumsum(axis=0))

#torch.dot表示向量点积
x=torch.ones(4,dtype=torch.float32)
y=torch.arange(4,dtype=torch.float32)
print(x,'\n',y,'\n',torch.dot(x,y))
#也可以通过按元素乘法再求和表示点积
print(torch.sum(x*y)) 
#矩阵向量积Ax
A=torch.arange(20,dtype=torch.float32).reshape(-1,4)
print(A.shape,'\n',x.shape,'\n',torch.mv(A,x).shape)
#矩阵乘积AB
B=torch.ones(4,3)
print(B,'\n',torch.mm(A,B))

#=====范数（类似长度）=====
#L2范数
u=torch.tensor([3.0,-4.0])
print(torch.norm(u))
#L1范数
print(torch.abs(u).sum())
#矩阵的F范数
print(torch.norm(torch.ones(4,9)))


