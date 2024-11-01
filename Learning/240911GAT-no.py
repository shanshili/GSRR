from _240830GraphConstruct import location_graph, topological_features_construct, data_color_graph
from _240810data import get_data2
from DAEGC2.daegc import DAEGC,target_distribution
from DAEGC2.utils import get_M
from DAEGC2.evaluation import eva
import pandas as pd
import numpy as np
from torch import nn, optim
import torch.nn.functional as F


Path1 = '../dataset/北京-天津气象2021/北京-天津气象2021'
TEM, data_location = get_data2(Path1)
location_g,A = location_graph(data_location)
print(np.shape(TEM))


"""
GAT的原理，需要什么
"""

