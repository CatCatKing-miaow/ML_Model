# GitHub 仓库管理指南 
这份指南总结了如何在 Windows (VS Code) 环境下，使用 Git Bash 管理机器学习项目的完整流程。\
1. 核心工作流环境推荐配置：VS Code 双终端模式\
终端 1 (PowerShell/CMD): 专门用于运行 Python 代码、训练模型 (python train.py)。\
终端 2 (Git Bash): 专门用于执行 Git 指令 (git commit 等)，保持环境隔离。

2. 首次安装与身份配置 (One-Time Only)在新电脑上安装 Git 后，必须执行一次身份登记。
# 1. 设置用户名 (建议使用英文名)
git config --global user.name "你的名字"

# 2. 设置邮箱 (必须与 GitHub 注册邮箱一致，否则没有绿格子)
git config --global user.email "你的邮箱@example.com"
3. 新项目初始化流程 (Start a New Project)当你开始一个新的项目文件夹时，按以下顺序操作：3.1 初始化与连接# 1. 初始化仓库
git init

# 2. 强制将分支名设为 main (现代化标准)
git branch -M main

# 3. 关联 GitHub 远程仓库 (URL 在 GitHub 网页创建仓库后获取)
git remote add origin [https://github.com/你的用户名/仓库名.git](https://github.com/你的用户名/仓库名.git)\
3.2 配置黑名单 (.gitignore)策略： 采用“白名单模式”，只允许上传代码和笔记，过滤数据集和杂物。在项目根目录创建 .gitignore 文件，内容如下：
# 1. 拒绝所有文件
*

# 2. 允许访问子文件夹 (必须加这一行，否则无法扫描子目录)
!*/

# 3. 放行特定格式文件 (根据需要添加)
!*.py\
!*.ipynb\
!*.md\
!.gitignore\
4. 日常开发“三板斧” (Daily Workflow)这是每天写代码时重复频率最高的操作。
步骤指令作用
1. 装车\
git add .扫描所有变化（新增/修改/删除/移动），放入暂存区。
2. 存档\
git commit -m "描述信息"生成版本快照。描述信息必填，例如："完成线性回归作业"。
3. 发货\
git push将本地存档同步到 GitHub 云端。\
注意： 第一次推送时如果报错，请使用 git push -u origin main。之后的日常推送只需 git push。
4. 常见问题与技巧\
    Q1: 遇到 LF / CRLF 警告怎么办？warning: LF will be replaced by CRLF in ...处理： 直接忽略。这是 Git 在自动处理 Windows (CRLF) 和 Linux (LF) 的换行符差异，不影响代码运行。\
    Q2: 想要整理/移动文件？操作： 直接在 VS Code 文件管理器中拖拽移动或重命名。Git 处理： 移动完后，直接执行日常“三板斧” (add -> commit -> push)，Git 会自动识别出这是“移动”操作，而非“删除再新建”。\
    Q3: 为什么空文件夹没上传？机制： Git 不追踪空文件夹。解决： 只有当文件夹里有文件时，它才会被上传。

祝代码无 Bug，GitHub 全绿！ 🟩🟩🟩