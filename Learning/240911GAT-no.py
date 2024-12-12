from _240830GraphConstruct import location_graph
from _240810data import get_data2
import numpy as np

Path1 = '../dataset/北京-天津气象2021/北京-天津气象2021'
TEM, data_location = get_data2(Path1)
location_g,A = location_graph(data_location)
print(np.shape(TEM))


"""
GAT的原理，需要什么
"""

