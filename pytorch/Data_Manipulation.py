import torch

x=torch.arange(12,dtype=torch.float32)
x=x.reshape(3,4)

print(x)
y=torch.zeros((2,3,4))
print(y)
z=torch.ones((2,3))
print(z)

x1=torch.tensor([[2,1,3,4],[2,4,5,6],[3,4,5,6]])
print(x1)

x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)#求幂运算
print(torch.exp(x))

x=torch.arange(12,dtype=torch.float32).reshape((3,4))
y=torch.tensor([[2.,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((x,y),dim=0))#dim 表示进行合并的维数
print(torch.cat((x,y),dim=1))
print(x==y)
print(x.sum())

#broadcasting
a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
print(a,b)
print(a+b)
#notation
# 对齐：把 Shape 写下来，右对齐。
# 检查：上下对比，要么相等，要么有一个是 1，要么上面缺了（视为1）。
# 原理：Stride = 0，数据没动，只是索引方式变了。
# 禁忌：不要对“小”张量做 In-place 的广播写入。

print(x[-1])#最后一行
print(x[1:3])#第1，2行（从第0行开始）
print(x)
x[1,2]=9
print(x)

x[0:2,:]=12 #前闭后开，选中0，1行；冒号全选列
print(x)

before=id(x)
x=x+1
print(id(x)==before)
#运行一些操作可能会导致为新结果分配新内存

z=torch.zeros_like(x)
print('id(z):',id(z))
z[:]=x+1
print('id(z):',id(z))
#执行原地操作；或者使用x+=1

a=x.numpy()
b=torch.tensor(a)
print(type(a),type(b))
#转化为Numpy张量

a=torch.tensor([3.5])
print(a,a.item(),float(a),int(a))
#将0维张量转化为python标量