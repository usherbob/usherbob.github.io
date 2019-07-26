---
layout:     post
title:      Point Cloud Processing
subtitle:   
date:       2019-07-07
author:     加华
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - deep learning
    - graph
    - point cloud
---

最近一段时间,我一直在探索图深度学习在点云(point cloud)数据集上的应用,在这里总结一下.

# 什么是点云?

​	点云数据作为一种常见的3D物体的表示形式,通常是由雷达在物体表面采样得到的,当然,点云数据也可以由软件生成,如ModelNet数据,就是通过在CAD模型表面采样得到的.

> A point cloud is a collection of points in space representing an object. It is a very basic discrete representation, essentially specifying the geometry of the object, by sampling it at certain positions. Typically, point clouds are generated using 3D scanners. However, point clouds can also be software-
> generated, e.g., as a result of the conversion of a CAD model to a point cloud or as a result of an operation yielding a new point cloud.[1]

​	ModelNet40数据集是[3]提出的一种3Dmesh模型数据集,而pointnet为了得到对应点云数据,通过最远点采样算法(Farthest Point Sampling)在mesh模型上采样对应点(2048个点)数据,从而得到了对应的ModelNet40点云数据集.

![point cloud sample](/img/image.HM5Q4Z.png)

​	随着pointnet[2]的提出,在点云数据集上的任务也受到了广泛关注,常见的任务有3D物体分类(3D object classification),3D物体部件分割(3D object Part Segmentation), 场景内语义分割(Semantic Segmentation in Scenes).我这里主要以3D object classification为例,综述图深度学习在3D point cloud上的研究进展.

# 问题描述

3D object classificaiton具体的任务,可以描述如下,[1]

> A point cloud is represented as a set of 3D points, 
>
> $$ \left \{P_i | i=1,...,n \right \} $$
>
> where each point $P_i$ is a vector of its $\left (x, y, z \right )$ coordinate plus extra feature channels such as color, normal etc. For simplicity and clarity, unless otherwise noted, we only use the $$ \left (x, y, z \right )$$ coordinate as our point’s channels.

> For the object classification task, the input point cloud is either directly sampled from a shape or pre-segmented from a scene point cloud. Our proposed deep network outputs $k$ scores for all the $k$ candidate classes.

# 模型

### 1.PointNet

[PointNet](https://arxiv.org/abs/1612.00593)

![pointnet architecture](/img/pointnet_architecture.png)

做3D点云任务的网络以pointnet为代表,通常会在同一个网络架构下分别实现分类与分割任务,这里主要讨论分类任务.

输入的每一个点云样本$P$均是由mesh采样得到的n个点(1024),每个点有3维坐标为初始特征.

首先是input transform模块,这一模块是借鉴于DeepMind文章Spatial Transformer Network,这一模块的目的是将输入点云样本均变换至同一坐标系下(Canonical Space ),采用的方法是通过T-Net回归得到一个3x3的矩阵$T$,利用矩阵$T$与输入点云做矩阵乘法对输入点云作变换$P=PT$, 而T-Net本身则是一个小的pointnet,在原文Joint Alignment Network中有解释. feature transform模块与input transform模块类似,是将feature在空间中对齐.note: **分类任务中,input transform与feature transform模块也可以不加,** 当然加入这两个模块,分类效果能够提升0.7个百分点左右.

然后是作者提出要解决点云无序性的问题,模型需要对于以不同顺序输入的点云样本,具有不变的输出,也就是,同样n个点,以不同顺序输入,我们的模型应该能将其分类成同一目标.作者由此提出,模型应该是具有对称性的,因此作者利用mlp去对输入点云提取特征(作者在文章中证明mlp可以逼近任意形式的函数,具体见原文4.2中Symmetry Function for Unordered Input).而具体代码实现中,作者则主要是利用了1x1的卷积实现.(note:所有数据均是利用tensorflow中的常用数据格式(batch_size, height, width, channels))

![mlp](/img/pointnet_core.jpg)

最后一个部分是一个aggregate模块,我们通过前面的mlp对每个点都提取了特征,那如何能够将这些点的特征放到一起,构成整个样本的特征?由于点云数据包括graph结构数据,缺乏有效的pooling策略,因此,最常用的特征aggregate模块就是global pooling,pointnet中利用了global maxpooling,即

`feature_afterpooling = tf.reduce_max(feature, axis=1)`

![global pooling](/img/global_pooling.png)

而后续直接将global pooling的特征接全连接层作输出即可.

### 2. EdgeConv

[Dynamic Graph CNN for Point Cloud](https://arxiv.org/abs/1801.07829)

![DGCNN architecture](/img/056.png)

这篇文章DGCNN是结合图卷积实现3D点云分类任务中,实验效果最好的,它是在pointNet基础上进行的工作,将pointNet中的MLP层换成EdgeConv+maxpooling,效果有较高提升.整个模型最关键的设计就是EdgeConv,主要针对pointnet中对每个顶点提取特征的方式进行了改进.值得注意的是,pointnet中由于对每个点提取特征时,并没有利用到其他点的信息,因此每个点之间是孤立的,而EdgeConv则是利用相邻顶点的信息构造当前顶点的特征,优势可想而知.

![a simple graph](/img/graph.png)

对于从点云构建的graph,EdgeConv按照如下方式结合邻顶点信息,

- concat邻顶点与中心顶点之差

  ```python
  for i = 1:N:
  	for j = a,b,c,d,e: # all neighbors
  		hij = concat(f(i), f(i)-f(j))
  ```

  输入为N * n维度的点云数据, 结合了$k$个邻顶点的信息,变成了N * k * 2n的矩阵edge_feature, 其中N代表一个样本中点的数量, k为邻顶点的数量, n为每个点的输入特征维度.

- 1x1卷积

  经过上一步concat后,利用1x1的卷积将中心顶点信息与邻顶点信息有机结合一下.

- maxpooling

  当然,一层edgeconv可以带来一圈(1 hop)的视野范围,模型中用到了4个EdgeConv,使得每个中心顶点可以最远结合到距离为4的顶点信息,如果不进行pooling,那4个EdgeConv就能结合$k^4$个邻顶点信息,矩阵维度则会变成N* $k^4$ *2n维度,太大了,因此,每次EdgeConv之后,需要在-2维度上进行maxpooling,如下,

  `feature = tf.reduce_max(feature, axis=-2) `

DGCNN中其余结构则与pointnet完全一致.

# References

[1] Dr. ir. D. Roose, POINT CLOUD PROCESSING USING LINEAR ALGEBRA AND GRAPH THEORY

[2] pointNet

[3] shepeNet

[4]Dynamic Graph CNN for learning on Point Cloud