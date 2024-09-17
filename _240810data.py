from pickletools import float8

import pandas as pd
import codecs
import csv
import numpy as np
from tqdm import tqdm
import os
import re
import matplotlib.pyplot as plt

# BJ_position = pd.read_csv('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/BJ_position.csv')
# TJ_position = pd.read_csv('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/TJ_position.csv')
# dataset_location = pd.concat([BJ_position, TJ_position])
# print(dataset_location)

"""
with codecs.open('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/BJ_position.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f, skipinitialspace=True):
        print(row)
f.close()
"""
"""
数据对应坐标
"""
def get_data2(path):
    dataset_list2 = []
    lat = []
    lon = []
    tem = []
    # print(os.listdir(path))
    # print(os.path.basename(path))
    for file in tqdm(os.listdir(path)):  ##进度条
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        lat.append(df.loc[0,'lat'])
        lon.append(df.loc[0,'lon'])
        tem.append(df.loc[:,'TEM'].astype(str))
        # tem = np.vstack((tem,df.loc[:,'TEM'].values))
        #TEM_data = np.vstack((tem, np.transpose(df.loc[0,'lon'])))
    data_location = np.transpose(np.vstack((lat, lon)))
    # tem_data = np.asarray(tem)
    # print(type(df.loc[:,'TEM'].values))
    # print(tem)
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(tem)], axis=1)
    # print(data_location)
    return dff.fillna(0).values.T,data_location
"""
只有数据，没有对应坐标
"""
def get_data(path):
    dataset_list = []
    # print(os.listdir(path))
    # print(os.path.basename(path))
    for file in tqdm(os.listdir(path)):  ##进度条
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        full_filename = os.path.basename(file_path) # 带后缀
        filename = full_filename.split('.')[0] # 不带后缀
        # print(re.findall(r"\d+", str(filename))[0])
        df['NO'] = int(re.findall(r"\d+", str(filename))[0])
        if os.path.basename(path) == 'BJ':
            df['label'] = 1
        else:
            df['label'] = 2
        dataset_list.append(df)
    df = pd.concat(dataset_list)
    return df

"""
BJ :
Label 1
sensor(NO) 453
pieces 8760

TJ :
Label 2
sensor(NO) 301
pieces 8760
"""
# Path1 = '../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/BJ'
# test_1 = get_data(Path1)
# Path2 = '../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/TJ'
# test_2 = get_data(Path2)
# test_df = pd.concat([test_1, test_2])
# print(test_df)


"""
location graph 与 data 如何关联
"""
# 要把数据构成图
# 把温度数据和图上的节点（即邻接矩阵，关联起来）
# 所以一般来说，节点的数据都是什么样的？
# tem_data = test.values[:,(1,5,6)] # data NO label
# print(tem_data)
# Data Associations （非常慢 不合理）
# for tem in tem_data:
#     for location in data_location:
#         if tem[1] == location[2] :
#             if tem[2] == location[3] :
#                 location = np.hstack((location, tem[0]))
# print(data_location)

# def node_data_associations(location_g,dataset):
