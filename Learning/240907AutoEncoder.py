import torch
from sympy.codegen import Print
from torch import nn, optim
import pandas as pd
# from _240810data import get_data
from _240830GraphConstruct import location_graph, topological_features_construct, data_color_graph
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
epoch_range = 850
"""
Train node quality based on node topology characteristics (no tem data)
"""


BJ_position = pd.read_csv('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/BJ_position.csv')
TJ_position = pd.read_csv('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/TJ_position.csv')
# Path1 = '../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/BJ'
# test_1 = get_data(Path1)
# Path2 = '../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/TJ'
# test_2 = get_data(Path2)
# test = pd.concat([test_1, test_2])
dataset_location = pd.concat([BJ_position, TJ_position])
lat = dataset_location['lat'].values
lon = dataset_location['lon'].values
# NO = dataset_location['NO'].values
# Label = dataset_location['label'].values
data_location = np.transpose(np.vstack((lat, lon)))


"""
data:需要进行分割的数据集
random_state:设置随机种子，保证每次运行生成相同的随机数
test_size:将数据分割成训练集的比例
"""
train_set, test_set = train_test_split(data_location, test_size=0.2, random_state=42)

"""
首先获得节点的拓扑特征，再用特征学习节点质量（754*1）
"""
location_g,A = location_graph(train_set)
topological_features = topological_features_construct(location_g)

# Validation set
location_g2,A2 = location_graph(test_set)
topological_features2 = topological_features_construct(location_g2)

"""
"""
data = np.array(topological_features) ###### data不对
# 归一化
norm_scalar = MinMaxScaler() ######？
data = np.transpose(norm_scalar.fit_transform(data))
# data = np.transpose(np.array(topological_features))
#print(data)
# print('X矩阵形状：', data.shape)      #  (754, 6)
data = torch.tensor(data, dtype=torch.float32)

# Validation set
data2 = np.array(topological_features2)
norm_scalar = MinMaxScaler()
data2 = np.transpose(norm_scalar.fit_transform(data2))
data2 = torch.tensor(data2, dtype=torch.float32)

"""
"""
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.encoder = nn.Sequential(
            nn.Linear(6, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 6),
            nn.Tanh()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fun = nn.MSELoss()

loss_history = []
for epoch in range(epoch_range):
    encoded, decoded = autoencoder(data)
    loss = loss_fun(decoded, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() ##
    loss_history.append(loss.item()) ##
    print('epoch:{}, loss:{}'.format(epoch, loss))

# Validation set
loss_history2 = []
for epoch in range(epoch_range):
    encoded2, decoded2 = autoencoder(data2)
    loss = loss_fun(decoded2, data2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() ##
    loss_history2.append(loss.item()) ##
    print('epoch:{}, loss:{}'.format(epoch, loss))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(loss_history)), loss_history)
ax1.plot(range(len(loss_history2)), loss_history2) # Validation set
plt.ylabel('Loss')
plt.xlabel('Epoch : {}'.format(epoch_range))
plt.title('Training Loss')
plt.text(0,loss_history[0],loss_history[0])
plt.text(epoch_range,loss_history[(epoch_range-1)],loss_history[(epoch_range-1)],horizontalalignment = 'left')
plt.text(epoch_range,loss_history2[(epoch_range-1)],loss_history2[(epoch_range-1)],horizontalalignment = 'left') # Validation set
plt.savefig('Training_Loss_epoch'+str(epoch_range)+'.svg', format='svg')
plt.show()
#plt.annotate(format(loss_history[(epoch_range-1)],'.7f'),(epoch_range,loss_history[(epoch_range-1)]),(epoch_range,loss_history[epoch_range]+100))

print(encoded.detach().numpy())
# print(encoded.detach().numpy()[[314,315,316,317,318,677,678,679,680],[0,0,0,0,0,0,0,0,0]])
print(encoded.detach().numpy()[[460,560],[0,0]])
# print('epoch:{}, loss:{}'.format(0, loss_history[0]))
# print('epoch:{}, loss:{}'.format(epoch_range, loss_history[epoch_range-1]))
data_color_graph(encoded.detach().numpy(),location_g,train_set,epoch_range)
plt.show()

torch.save(autoencoder, './model_save/autoencoder.pth')
