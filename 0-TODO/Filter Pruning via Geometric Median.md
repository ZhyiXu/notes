## Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration

论文地址： https://arxiv.org/abs/1811.00250

作者：Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, Yi Yang

代码地址：https://github.com/he-y/filter-pruning-geometric-median

发表：CVPR 2019 Oral

代码地址：https://github.com/he-y/filter-pruning-geometric-median



### 摘要

Previous works utilized ''smaller-norm-less-important'' criterion to prune filters with smaller norm values in a convolutional neural network. In this paper, we analyze this norm-based criterion and point out that its effectiveness depends on two requirements that are not always met:

 (1) the norm deviation of the filters should be large;

 (2) the minimum norm of the filters should be small. 

To solve this problem, we propose a novel filter pruning method, namely Filter Pruning via Geometric Median (FPGM), to compress the model regardless of those two requirements. Unlike previous methods, FPGM compresses CNN models by pruning filters with redundancy, rather than those with ''relatively less'' importance. When applied to two image classification benchmarks, our method validates its usefulness and strengths. 

Notably, on CIFAR-10, FPGM reduces more than 52% FLOPs on ResNet-110 with even 2.69% relative accuracy improvement. Moreover, on ILSVRC-2012, FPGM reduces more than 42% FLOPs on ResNet-101 without top-5 accuracy drop, which has advanced the state-of-the-art.  



### 实验结果

ResNet + Cifar10

![1587354612850](D:\Notes\raw_images\1587354612850.png)



![1587354670064](D:\Notes\raw_images\1587354670064.png)



![1587354796519](D:\Notes\raw_images\1587354796519.png)



![1587354763001](D:\Notes\raw_images\1587354763001.png)