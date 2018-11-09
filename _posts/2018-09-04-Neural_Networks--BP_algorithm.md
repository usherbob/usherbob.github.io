---
layout:     post
title:      BP algorithm of fully connected network
subtitle:   easy BP algorithm derivation
date:       2018-09-04
author:     加华
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - deep learning
    - algorithm
---

>BP algorithm of fully connected  network 最近，由于在看论文的过程中，我发现自己对于CNN的基本结构不是很熟悉，单独抽了一段时间，强化一下自己的基础知识。我的主要参考文献是Neural Networks and Deep Learning, Michael Nielson, 原文http://neuralnetworksanddeeplearning.com/index.html

## 内容
本篇博客内容主要是对于全连接神经网络中的BP算法进行推导，主要给出公式的推导及其值观理解

## Preliminary
首先，我们需要一个对于neuron的高效的标号方式，
![](/img/neuron_denotion.png)

此图借鉴于Neural Networks and Deep Learning, Michael Nielson，然后，我们需要给出neuron的输入与输出间的关系
![](/img/activation_equation.png)

其中的sigma函数可以是任意激活函数，如sigmoid函数。
我们还需要给出loss函数，书中名为cost函数，此处定义为C。

## BP equations
问题的核心在于求解出根据cost函数与其z之间的偏导，这是我们后续优化神经网络中各参数调整的关键，此处我们将其定义为delta。

![](/img/BP_equations.png)
