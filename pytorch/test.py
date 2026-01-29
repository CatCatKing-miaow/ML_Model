import torch
import sys

def verify_installation():
    print("--- PyTorch 环境检测 ---")
    
    # 1. 检测 Python 版本
    print(f"Python 版本: {sys.version}")
    
    # 2. 检测 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 3. 检测 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        # 4. 获取 GPU 信息
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        cuda_version = torch.version.cuda
        
        print(f"当前 GPU 设备: {device_name}")
        print(f"CUDA 运行库版本: {cuda_version}")
        
        # 5. 测试张量运算
        print("\n--- 正在进行 GPU 运算测试 ---")
        try:
            # 创建两个随机张量并移动到 GPU
            x = torch.rand(5, 3).to('cuda')
            y = torch.rand(5, 3).to('cuda')
            
            # 执行加法
            z = x + y
            
            print("张量加法测试成功！")
            print(f"结果张量设备: {z.device}")
        except Exception as e:
            print(f"运算测试失败: {e}")
    else:
        print("\n[警告] CUDA 不可用！请检查显卡驱动是否安装正确。")

if __name__ == "__main__":
    verify_installation()