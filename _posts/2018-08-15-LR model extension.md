---
layout:     post
title:      LR model extension
subtitle:   分类标签为1，-1
date:       2018-08-15
author:     加华
header-img: img/machine_learning.jpg
catalog: true
tags:
    - Machine Learning
    - CS229
---

## Q

问题来源于CS229(2017Autumn)课程中，ps2的第一题，题目中给出了两组二分类的数据和一个LR的模型，关于题目中的具体问题不再详述，倒是题目中给出的模型让我花

了很长时间才搞明白，现在po出来。题目中数据http://cs229.stanford.edu/ps/ps2/ data_a.txt，模型 http://cs229.stanford.edu/ ps/ps2/lr_debug.py，

模型中的部分代码让我困惑，

```
def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta) 
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))
```

这是题目中用于模型求导的代码，这与我之前接触到的标签为0，1的LR模型不同，下面给出标签为-1，1时，LR目标函数和梯度的求解过程。

## 推导过程

这道题目中对于label为-1和1的情况下，LR的模型的推导困惑了我很长时间。

从LR回归的本质来讲，利用sigmoid函数将 $ \theta ^Tx $ 映射为0，1值，而本题目中，label则为-1和1，因此需要对sigmoid的值域做一个映射，这里采取

$ 2f(x)-1 $ 将函数值映射为 -1，1值，$$ y=\left\{\begin{matrix} 1,z>0\\ 0,z=0\\ -1,z<0 \end{matrix}\right.$$ 其中，$ z = \theta ^Tx $ 

因此，可以进行如下推导，由 $   y = \frac{2}{1+e^{-z}}-1 \tag{1}$  

由$(1)$可得到 $$   P(y=-1| z)= \frac{e^z}{1+e^z} $$ $$ P(y=1| z)= \frac{1}{1+e^z} $$  

因此，利用极大似然法，我们可以得出其对数极大似然函数为$$   L(z) = ( \frac{e^{z}}{1+e^z})^{I[y=-1]}\cdot ( \frac{1}{1+e^{z}})^{I[y=1]} \tag{2} $$ 

则对数极大似然函数为 $$   l(z) = (ln(1+exp(z))-z){I[y=-1]}\cdot ( ln(1+e^{z})){I[y=1]} \tag{3}$$  

由于label，即y的值域，$(3)$可以写为 $$   l(z) = (ln(1+e^{-yz})+yz){I[y=-1]}\cdot  ln (1+e^{-yz}){I[y=1]} \tag{4}$$   

利用$ l  n(1+e^{\mu })) = ln(1+e^{-\mu }) + \mu $   

可将$(4)$化为如下形式，$$   l(z) = ln(1+e^{yz}){I[y=-1]}\cdot ln (1+e^{yz}){I[y=1]} \tag{5}$$  

即为 $$   l(z) = ln(1+e^{yz}) \tag{6}$$   

式$(6)$即为本体中所沿用目标函数
