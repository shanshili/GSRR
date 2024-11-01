import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# 加载数据集

"""
transforms.Compose
pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
Normalize：Normalized an tensor image with mean and standard deviation
           使用平均值和标准差对张量图像进行归一化
ToTensor：convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
"""
# Define the transformation to apply to the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load the MNIST dataset
"""
MNIST是一个手写体数字的图片数据集
https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
https://blog.csdn.net/tony_vip/article/details/118735261
"""
trainset = MNIST(root='./data', train=True, download=True, transform=transform)
testset = MNIST(root='./data', train=True, download=True, transform=transform)

# Set hyperparameters
"""
batch_size: 表示单次传递给程序用以训练的数据（样本）个数。训练时所使用的内存量会比较小、训练数据集的分块训练，提高训练的速度。劣势：
使用少量数据训练时可能因为数据量较少而造成训练中的梯度值较大的波动。

"""
batch_size = 64
learning_rate = 0.001
num_epochs = 10