# 伴随矩阵

对于任何非负整数 k，方阵 A 的伴随矩阵的 k 次幂等于 A 的 k 次幂的伴随矩阵。我用 Python 写了一个程序来验证这个定理。 

![adj(A^k)=adj(A)^k](https://github.com/easai/cofactor/blob/main/formula.png)

* 测试一 
方阵 A：
[[7, -5, -8],
 [-3, -7, -2],
 [0, -4, -8]]

* 测试二
方阵 A：
[[7, -5, -8],
 [0, -7, -2],
 [0, 0, -8]]
 
* 测试三 
方阵 A：
[[7, 0, 0],
 [-3, -7, 0],
 [0, -4, -8]]

* 测试四 
A：[1]（只有一个值）
 
* 测试五
A：[]（空数组）
 
* 测试六  
矩阵 A：
[[7, -5, -8],
 [0, 0, -8]] 
（非方阵）


## 使用法

```bash
poetry install
poetry run pytest
```

## 结果

以下是测试用例的预期输出结果及说明。

## 测试一
方阵 A：
[[7, -5, -8],
 [-3, -7, -2],
 [0, -4, -8]]

### 结果：
```bash
测试一
A:
[[ 7 -5 -8]
 [-3 -7 -2]
 [ 0 -4 -8]]
A^2:
[[64 32 18]
 [ 0 72 54]
 [12 60 72]]
adj(A^2)
[[ 1944.   648.  -864.]
 [-1224.  4392. -3456.]
 [  432. -3456.  4608.]]
adj(A)
[[ 48. -24.  12.]
 [ -8. -56.  28.]
 [-46.  38. -64.]]
adj(A)^2:
[[ 1944.   648.  -864.]
 [-1224.  4392. -3456.]
 [  432. -3456.  4608.]]
True
```
### 原因：
因为 adj(A^2) 等于 adj(A)^2，代码正确地返回 True。  

## 测试二
方阵 A：
[[7, -5, -8],
 [0, -7, -2],
 [0, 0, -8]]
### 结果：
```bash
测试二
A:
[[ 7 -5 -8]
 [ 0 -7 -2]
 [ 0  0 -8]]
A^2:
[[49  0 18]
 [ 0 49 30]
 [ 0  0 64]]
adj(A^2)
[[ 3136.    -0.     0.]
 [   -0.  3136.    -0.]
 [ -882. -1470.  2401.]]
adj(A)
[[ 56.  -0.   0.]
 [-40. -56.  -0.]
 [-46.  14. -49.]]
adj(A)^2:
[[ 3.13600000e+03  0.00000000e+00  0.00000000e+00]
 [-5.68434189e-14  3.13600000e+03  0.00000000e+00]
 [-8.82000000e+02 -1.47000000e+03  2.40100000e+03]]
True
```
### 原因：
此测试适用于上三角矩阵 。因为 adj(A^2) 等于 adj(A)^2，代码正确地返回 True。    

## 测试三
方阵 A：
[[ 7  0  0]
 [-3 -7  0]
 [ 0 -4 -8]]
 
### 结果：
```bash
测试三
A:
[[ 7  0  0]
 [-3 -7  0]
 [ 0 -4 -8]]
A^2:
[[49  0  0]
 [ 0 49  0]
 [12 60 64]]
adj(A^2)
[[ 3136.    -0.  -588.]
 [   -0.  3136. -2940.]
 [    0.    -0.  2401.]]
adj(A)
[[ 56. -24.  12.]
 [ -0. -56.  28.]
 [  0.  -0. -49.]]
adj(A)^2:
[[ 3.13600000e+03  5.68434189e-14 -5.88000000e+02]
 [ 0.00000000e+00  3.13600000e+03 -2.94000000e+03]
 [ 0.00000000e+00  0.00000000e+00  2.40100000e+03]]
True
```
### 原因：
此测试适用于下三角矩阵 。因为 adj(A^2) 等于 adj(A)^2，代码正确地返回 True。   

## 测试四
A：[1]（只有一个值）
### 结果：
```bash
测试四
A 必须是矩阵
```
### 原因：
因为 A 只有一个值，A 不是矩阵。因此代码打印错误 "A 必须是矩阵" 并终止。这边缘案例测试确保全面覆盖。

## 测试五
A：[]（空数组）
### 结果：
```bash
测试五
A 必须是矩阵
```
### 原因：
因为 A 是空数组，A 不是矩阵。因此代码打印错误 "A 必须是矩阵" 并终止。这边缘案例测试确保全面覆盖。

## # 测试六
矩阵 A：
[[7, -5, -8],
 [0, 0, -8]] 
（非方阵）
### 结果：
```bash
测试六
矩阵 A 必须是方阵
```
### 原因：
因为 A 是非方阵，代码打印错误 "矩阵 A 必须是方阵" 并终止。这边缘案例测试确保全面覆盖。 

