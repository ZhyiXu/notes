### 奇异值分解 (SVD) 原理与在降维中的应用

文章地址：https://www.cnblogs.com/pinard/p/6251584.html

奇异值分解(Singular Value Decomposition，以下简称SVD) 是在机器学习领域广泛应用的算法，它不光可以用于降维算法中的特征分解，还可以用于推荐系统，以及自然语言处理等领域。是很多机器学习算法的基石。本文就对SVD的原理做一个总结，并讨论在在PCA降维算法中是如何运用运用SVD的。

#### 1. 回顾特征值和特征向量

我们首先回顾下特征值和特征向量的定义如下：
$$
Ax=\lambda x
$$
其中A是一个 $n×n$ 的实对称矩阵，$x$ 是一个 n 维向量，则我们说 λ 是矩阵 A 的一个特征值，而 x 是矩阵 A 的特征值 λ 所对应的特征向量。

求出特征值和特征向量有什么好处呢？ 就是我们可以将矩阵A特征分解。如果我们求出了矩阵A的n个特征值λ1≤λ2≤...≤λnλ1≤λ2≤...≤λn,以及这nn个特征值所对应的特征向量{w1,w2,...wn}{w1,w2,...wn}，，如果这nn个特征向量线性无关，那么矩阵A就可以用下式的特征分解表示：

A=WΣW−1A=WΣW−1

 

 

