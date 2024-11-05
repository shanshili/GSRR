import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

"""
根据重要性依次选择选择节点加入，构成1-450点的450个图，观察各性能曲线趋势
很多鲁棒性的指标就是反着来，依次删掉节点，也是要对比450个图
先写衡量指标，依次计算450个数值的代码，然后画在一张图里

设定一个指标步距，比如450/30,15个点看趋势有没有明显的区分
"""

# 全局修改字体
config = {
            "font.family": 'Times New Roman',
            "font.size": 10.5,
            # "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            # "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)



x = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135]

# MSE
"""
多个节点特征怎么汇总
图特征怎么获得
"""
def mean_squared_error(y_true, y_pred):
    """
    计算均方误差 (MSE)
    参数:
    y_true (array-like): 真实值
    y_pred (array-like): 预测值
    返回:
    float: 均方误差
    """
    # 确保输入是 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 计算误差平方
    squared_errors = (y_true - y_pred) ** 2
    # 计算均方误差
    mse = np.mean(squared_errors)
    return mse

# 示例数据
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# 计算 MSE
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}")

#



fig, ax = plt.subplots()
p = ax.twinx()
ax.axhline(0.05, c='#E89B9E', ls='--')
# p.axhline(0.3, c='#ffccc3', ls='--')

line_mse = ax.plot(x, mse, marker='<', markerfacecolor='white', label='MSE', color='#d92523')
line_aec = p.plot(x, mse, marker='>', markerfacecolor='white', label='AEC', color='#2e7ebb')
ax.set_ylabel('MSE')
p.set_ylabel('AEC')

# 设置两轴颜色
# ax.spines['left'].set_color('#1d2e58')
# p.spines['left'].set_color('#1d2e58')
# ax.spines['right'].set_color('#821e26')
# p.spines['right'].set_color('#821e26')
# 刻度颜色
# ax.tick_params(axis='y', labelcolor='#1d2e58', color='#1d2e58')
# p.tick_params(axis='y', labelcolor='#821e26', color='#821e26')


ax.legend()
# 设置坐标轴名
ax.set_xlabel('节点数目')
# plt.savefig('./图3-9-1.png',dpi=600)
plt.show()

