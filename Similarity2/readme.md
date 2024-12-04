# .

### dataprocess2

### GraphConstruct2

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

### resilience

### resilience_cuda

CrossEntropy

### resilience_test_cuda_plot

model test:

### resilience_cuda_eval

## 2

### plotscore2

line charts

### plotscore3

for Rg

### plotscore31

for predict data