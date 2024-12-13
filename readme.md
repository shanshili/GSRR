# GSRR: Graph Similarity and Resilience Ranking

![Gp75_Contribution_weights_2](https://github.com/shanshili/GSRR/blob/855eee837c98962f57efd8bce4fb8e3251ae6936/readme.assets/Gp75_Contribution_weights_2.svg)![select-indicators6](https://github.com/shanshili/GSRR/blob/855eee837c98962f57efd8bce4fb8e3251ae6936/readme.assets/select-indicators6.svg)

![Gp75_MGC-RMselect14](https://github.com/shanshili/GSRR/blob/855eee837c98962f57efd8bce4fb8e3251ae6936/readme.assets/Gp75_MGC-RMselect14.svg)

![t1+CE+t4_softsort_normal_epoch_150_lr_1e-06_20241204_204900](https://github.com/shanshili/GSRR/blob/855eee837c98962f57efd8bce4fb8e3251ae6936/readme.assets/t1%2BCE%2Bt4_softsort_normal_epoch_150_lr_1e-06_20241204_204900.svg)

![3y_select_20_softsort_normal_epoch_150_lr_1e-06_20241204_204900](https://github.com/shanshili/GSRR/blob/855eee837c98962f57efd8bce4fb8e3251ae6936/readme.assets/3y_select_20_softsort_normal_epoch_150_lr_1e-06_20241204_204900.svg)



# utils

### dataprocess

### GraphConstruct

### utils

### model

GAT

### model_cuda

### model_cuda2

NodeEmbeddingModule2：modify GAT





# MGC-RM



## 1

### perturbation

Generate a perturbation graph

### perturbation2

Generate a perturbation graph

packaged into a function

## 2

### MFC_RMF

Multi-Granularity Cross Representation and Matching

for i in range(perturbed_a.shape[0]) Perturbation graph are involved in training

```python
plt.savefig
torch.save pth
```

### MFC_RMF2

1. Specify a graph pair to **train** and save the model. 

2. Add the model overload module to generate the **similarity score** of the remaining graph pair and save it as a file. (Nearly 4 hours)
3. Save prediction_loss as txt and fig to evaluate the difference in the effect of the graph

```python
np.savetxt('./similarity score/z'+ 'GP_'+str(j)+'_node_'+str(node)+'_data_'+str(data_num)+'model_'+str(formatted_time)+'.txt', z_p)

np.savetxt('./prediction_loss/GraphPair_' + str(j) + '_n' + str(node) + '_d' + str(data_num) + '_Prediction_Loss_epoch_' + str(args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(formatted_time) +'.txt', loss_history2)
```

### MFC_RMF2cuda2

## 3

### plotpredictloss

### plotscore

## 4

### PageRank2

weights PageRank

## 5

### Perform

### plotpredictloss

### plotscore

## 6

### no-readout





# Resilience

## 1

### resilience-cpu

cuda Unused

The oldest version

### resilience_cuda

CrossEntropy

Basic version

### resilience_train_test

model test:

use cuda

with plot

### resilience_eval

model test:

use cuda

## 2

### plotscore2

line charts

### plotscore-r

for Rg

### plotscore-y

for predict data

## 3

### R-Perform

Evaluate metric calculations