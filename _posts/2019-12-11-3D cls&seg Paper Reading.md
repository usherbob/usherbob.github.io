---
layout:     post
title:      3D cls&seg Paper Reading
subtitle: 
date:       2019-12-11
author:     加华
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:

   - deep learning
   - point cloud
---

1. Linked Dynamic Graph CNN(ICRA)

   <img src="/img/ldgcnn.png" alt="选区_187" style="zoom: 50%;" />

   Contribution:

   - 不同EdgeConv之间的跳层连接,或者说是dense连接,与DenseNet的思想较为相似.

   - 重新设计了EdgeConv的卷积形式,从而能够解决旋转鲁棒性的问题,借此舍弃了T-Net

     <img src="/img/image-20191203172254220.png" alt="image-20191203172254220" style="zoom:67%;" />

   - 通过重新训练分类网络,将模型的分类准确率从91.8提升至92.9,实际在tf代码中的操作是将feature保存下来,然后利用保存的固定的feature重新训练分类器.

2. SRINet(MM)

   <img src="/img/image-20191203172600696.png" alt="image-20191203172600696" style="zoom:50%;" />

   Contributions:

   - 设计旋转不变投影模块,即Point Projection,对每一个输入点云,任意选定一组3个方向的向量,将其与原输入点云中的每个点分别作投影,从而组成一个4维的点集.作者通过理论分析证明了投影后的四元组的旋转鲁棒性.

     <img src="/img/image-20191203173214577.png" alt="image-20191203173214577" style="zoom: 67%;" />

   - 作者利用了一个key point detection的模块以及Graph Aggregation,相当于增加了注意力机制,其中key point通过每个点与邻点的surface normal信息来确定.作者通过ablation验证了这两个模块的有效性.

     <img src="/img/image-20191203175037147.png" alt="image-20191203175037147" style="zoom: 67%;" />

     <img src="/img/image-20191203175318589.png" alt="image-20191203175318589" style="zoom:50%;" />

3. Self-supervised deep learning on point clouds by reconstructing space(NeurIPS)

   <img src="/img/image-20191203191731610.png" alt="image-20191203191731610" style="zoom:50%;" />

   Contributions:

   - 提出了一个针对不同任务的预训练方法,可以提升3D object classification与3D segmentation的效果.目标在空间中被分成27个(3x3x3)大体素,预训练模型通过类似于语义分割的方法,为每个点预测对应的体素标签.

     Assumption:

     The key assumption of the proposed method is that learning to reassemble displaced point cloud segments is only possible by learning holistic representations that capture the high-level semantics of the objects in the point cloud.

4. Dynamic Points Agglomeration for Hierarchical Point Sets Learning(ICCV)

   <img src="/img/image-20191203193911880.png" alt="image-20191203193911880" style="zoom:67%;" />

   Contributions:

   - 提出了pooling模块DPAM,与GNN中的

     [DiffPool]: https://arxiv.org/abs/1806.08804

     几乎完全一致,不同的是作者提出的DPAM具体回归Agglomeration Matrix的方式.

   - 作者提出了共享DPAM模块的思想,避免参数过多.

5. KPConv(Flexible and Deformable Convolution for Point Clouds)

   <img src="/img/image-20191203211831769.png" alt="image-20191203211831769" style="zoom: 50%;" />

6. Point2Sequence

   <img src="/img/image-20191203223738037.png" alt="image-20191203223738037" style="zoom: 50%;" />

   Contributions:

   - 利用了一个局部的multi-scale,对要留下的中心点,去找他们的(32, 64, 128)最近邻,视野域不同;
   - 利用一个encoder-decoder外加attention的结构去对multi-scale的特征进行融合.

7. InterpConv

   <img src="/img/image-20191205204404553.png" alt="image-20191205204404553" style="zoom:50%;" />

   这篇文章类似于KPConv,利用一个立方体形的kernel去提取point cloud特征,利用插值方法去解决不规则问题,作者提出卷积核中的卷积位置可以是变化的,但文章大部分还是按照立方体形状设计.

8. SpiderCNN

9. Flex-Conv

   Contributions:

   - 设计flex-conv

     Processing irregular data requires a function w, which can handle an unbounded domain of arbitrary — potentially real-valued — relations between τ and l, besides retaining the ability to share parameters across different neighborhoods.在CNN中,由于卷积核中心与周围邻居的关系退化成为离散的3x3矩形,只需要有一个3x3的权重w表对应即可,但由于点的邻居的坐标是连续的,我们无法定义一个有限的权重表w对应,只能去学习一个对应关系$w'$,这个对应关系能够将邻居点的坐标对应到其卷积权重.文章拟合了一个线性的对应关系,通过优化参数$\theta_b,\theta_{b_{c}}$来调整.
     $$
     \tilde{w}\left(c, \ell, \ell^{\prime} | \theta_{c}, \theta_{b_{c}}\right)=\left\langle\theta_{c}, \ell-\ell^{\prime}\right\rangle+\theta_{b_{c}}
     $$

   - 设计点云分布密度相关的flex-pooling

     通过设计一个与density相关的点云采样方法($O(N)$)来代替FPS$(O(N^2))$,点云density的近似方法:
     $$
     \sum_{\ell^{\prime} \in \mathcal{N}_{k}(\ell)}\left\|\ell-\ell^{\prime}\right\|
     $$

10. Iterative Transformer  Network for 3D Point Cloud

    <img src="/img/image-20191210152142884.png" alt="image-20191210152142884" style="zoom:50%;" />

    Contributions:

    - 设计Iterative的Transformer,用于估计点云的旋转(姿态),提出PointNet中的T-Net会改变物体的scale,实际上是一个affine变换,而实质上针对刚体的旋转变换不应该有形变与尺度变换,应该由一个满足$RR^t=1, det(R)=1$的旋转矩阵与平移向量$l$构成,而通常用四元组去表示旋转矩阵$R$,因此Iterative Transformer每次只预测一个7维向量即可.
    - 类似Revisiting Point Cloud Classification,作者提出了一个新的数据集(仿真现实情况中的物体不同方向及遮挡造成的物体不完整).

11. A-CNN:

    <img src="/img/image-20191210194336228.png" alt="image-20191210194336228" style="zoom:50%;" />

    Contributions:

    - 类似于PointNet++的网络结构,作者重新设计了卷积与grouping,pooling仍然沿用了FPS,值得注意的是作者在文中提到了overlapped grouping带来的问题.

    - 比较有意思的一点是,作者利用normal信息对所有近邻点进行逆时针排列,之后再利用类似CNN的1x3或1x5卷积核进行卷积.

      将所有近邻点投影到normal为法向量的切平面上,然后将所有点在切平面上进行逆时针排序.

      <img src="/img/image-20191210194748960.png" alt="image-20191210194748960" style="zoom:50%;" />

12. Tangent Convolutions for Dense Prediction in 3D

    | <img src="/img/image-20191211160020862.png" alt="image-20191211160020862" style="zoom:50%;" /> | <img src="/img/image-20191211160050417.png" alt="image-20191211160050417" style="zoom:75%;" /> |
    | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | 切平面投影                                                   | 卷积设计                                                     |

    Contributions:

    - A-CNN中的切平面投影方法来自于本文章;
    - 类似于kNN graph的设计,最大的不同是本方法在切平面上寻找k最近邻.

