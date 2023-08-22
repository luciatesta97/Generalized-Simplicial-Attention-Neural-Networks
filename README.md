# Generalized Simplicial Attention Neural Networks


![Maps](https://github.com/lrnzgiusti/Simplicial-Attention-Networks/blob/main/assets/maps.jpg)
![Layer](https://github.com/luciatesta97/Generalized-Simplicial-Attention-Neural-Networks/blob/main/architec.png)

### Abstract 

The aim of this work is to introduce Generalized Simplicial Attention Neural Networks (GSANs), i.e., novel neural architectures designed to process data defined on simplicial complexes using masked self-attentional layers. Hinging on topological signal processing principles, we devise a series of self-attention schemes capable of processing data components defined at different simplicial orders, such as nodes, edges, triangles, and beyond. These schemes learn how to weight the neighborhoods of the given topological domain in a task-oriented fashion, leveraging the interplay among simplices of different orders through the Dirac operator and its Dirac decomposition. We also theoretically establish that GSANs are permutation equivariant and simplicial-aware. Finally, we illustrate how our approach compares favorably with other methods when applied to several (inductive and transductive) tasks such as trajectory prediction, missing data imputation, graph classification, and simplex prediction.

### Organization of the code

We can find five folders:
-GSAN_SAN: Implementation for GSAN when it is reduced to 1 level 
-GSAN_DIrac: Implementation for GSAN with Dirac Operator
-GSAN Joint: GSAN Joint Implementation
-2-simplex: GSAN for 2-simplex prediction experiment
-3-simplex: GSAN for 3-simplex prediction experiment

### References

[1] Veličković, Petar, et al. **Graph Attention Networks**. arXiv preprint arXiv:1710.10903 (2017). <br>
[2] Kipf, T.N. and Welling, M., 2016. **Semi-supervised classification with graph convolutional networks**. arXiv preprint arXiv:1609.02907. <br>
[3] Ebli, Stefania, Michaël Defferrard, and Gard Spreemann. **Simplicial neural networks**. arXiv preprint arXiv:2010.03633 (2020). <br>
[4] Barbarossa, Sergio, and Stefania Sardellitti. **Topological signal processing over simplicial complexes**. IEEE Transactions on Signal Processing 68 (2020): 2992-3007.


```
