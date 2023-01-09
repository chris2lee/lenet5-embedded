# LenetDeploy :bowtie: 

#### 介绍
- 用C语言部署PyTorch训练结果，以lenet-5为范例
- Deploy the PyTorch training results in C, taking lenet-5 as an example.


#### 软件架构

-  **matoperation.c/.h**  矩阵操作相关
-  **cnnoperation.c/.h**  卷积神经网络的基本算子（卷积、池化、平化、全连接运算）
-  **improvedlenet5.c/.h**  使用cnnoperation搭建lenet5神经网络
-  **lenet5params.h**  神经网络训练后得到的各参数
-  **main.c**  说明如何使用搭建完成的神经网络（以improvedlenet5为例）

# lenet5-embedded
