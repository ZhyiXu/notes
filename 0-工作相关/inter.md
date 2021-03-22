**1. 给出一个矩阵**

2 4 6 

5 7 9

1 8 8

相邻位置跨越需要消耗

Cost abs(2-4) = 2

Cost abs(2-5) = 3

在给定条件 Max_cost = 3 的情况下

能否从左上走到右下？ 返回 true 或者 false。二分

```python
import numpy as np
m,n = len(matrix),len(matrix[0])
d = np.zeros_like(matrix)
d[0][0] = 1
for i in range(m):
  for j in range(n):
    d[i][j] = 
```



