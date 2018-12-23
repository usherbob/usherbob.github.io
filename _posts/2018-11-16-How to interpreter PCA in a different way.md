---
layout:     post
title:      How to interpreter PCA in a different way
subtitle:   Use project point to solve for PCA
date:       2018-11-16
author:     加华
header-img: img/post-bg-ios10.jpg
catalog: true
mathjax: true
tags:
    - machine learning
    - algorithm
---

> We mostly derivate PCA method with the idea of maximize data variance. If you have ever watched machine learning video of Dr.Andrew Ng, you'll find that there's another impressive interpretation of PCA. We could understand PCA in the angle of orthogonal regression fit. Amazing, hmm. In this blog, I will refer one question of Stanford CS229 problem set and give my understanding of this question and how PCA could be derivated in this way.

## Question
![image of question](/img/pca_quest.jpg)"source(http://cs229.stanford.edu/)" 

## Derivation

### project point
There is a definition in how to get the project point of x. Actually, we could get the project point without thinking much. $ x=u^Txu $

Also, you could get the same result through solving for the optimization problem. Since we could represent $ v $ in $ f_{u}(x) $ with $ v=\alpha u $, it is easy to solve this optimization problem.
$f_u(x) = \left \|| x-\alpha u \right \||^2 = \left ( x-\alpha u \right )^T\left ( x-\alpha u \right ) = x ^T x-2\alpha u^T x + \alpha^2 u^T u$
$ \frac{\partial f_u\left ( x \right )}{\partial x} = -2 u^T x + 2u^Tu \alpha = 0$
So, we have $ \alpha = u^Tx$, thus, the project point is $ x=u^Txu $ 
