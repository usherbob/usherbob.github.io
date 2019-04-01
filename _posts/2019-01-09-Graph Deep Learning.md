---
layout:     post
title:      Graph Deep Learning
subtitle:   for beginners
date:       2019-01-09
author:     加华
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - deep learning
    - graph
---
> NIPS Tutorial-Geometric Deep Learning on Graphs and Manifolds [2]

## 1. Intro
 
图的应用主要是针对非欧式空间，例如3D流体的分析，社区网络分割。
![](/img/application_GDL.png 'source[2]'){:width="400px"}

图是由什么构成的？首先是图的结构，指这些顶点与顶点之间相互连接的关系，其次，是顶点上的信息，拿社区网络分析的这张图为例子，每个顶点就代表一个用户主题，顶点之间连接的关系可能表示用户之间的关系，工作关系、同事关系等，顶点上的信息则表征着这个用户自己的特征，性别、年龄。
![](/img/graph_structure.png 'source[2]'){:width="600px"}

## 2. Basic Graph Theories

### A. Laplacian Matrix

Laplacian matrix是图深度的spectra domain(类比与频域)处理方式的基础，GCN最早的形式就是通过对Laplacian matrix的特征值与特征向量作处理进行卷积操作。那么Laplacian matrix到底有什么直观的意义那？
![](/img/laplacian_matrix.png 'source[2]'){:width="600px"}

我们可以看到，我们定义的$ \Delta = D-W $算子可以很好的表征顶点与其周围邻居之间的关系，进而可以表现出图结构的smoothness。我们还可以参考博客[3]关注它的具体推推导。

### B. Orthogonal bases on graphs
如果我们想要找到某个图结构中最smooth的一个正交基底怎么办？当然是熟悉的配方，我们需要对其进行正交分解，最smooth的一个正交基底就是最大特征值对应的特征向量。
![](/img/eig_vec_lap.png 'source[2]'){:width="600px"}

在文献[1]中有一个很好的例子告诉我们不同的特征向量在图的信息上究竟有什么对应。图中的$u_0, u_1, u_{50}$分别代表Laplacian matrix的特征向量，下标表示排序。其实这就像极了欧式空间中信号在fourier变换下的直流信号以及不同的高频信号。
![](/img/eg_eigvec.png 'source[1]'){:width="600px"}

### C. Fourier analysis
要想了解卷积，我们通常会先了解Fourier变换，这是因为变换后的频域特征具有很好的性质，时域下的卷积在频域中通过乘积即可实现。我们以Fourier 级数为例，欧氏空间下的Fourier级数，可以认为是函数$f(x)$与正交基底$$en = exp\left \{ i2n\pi x \right \} $$点积得到[4]，即

$$f(x) = \sum_{k=1 }^{n}<f(x), en> \cdot en $$

其中，在Hilbert空间中的内积是这么定义的

$$<f(x),en(x)> = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)en(x)dx$$

那么在图上的Fourier变换如何实现那？我们自然希望通过Laplacian矩阵的正交变换基来实现，通过图结构在不同频率基底的变换得到其变换结果。因此，自然的，我们可以借用Laplacian matrix的正交特征向量作为变换基底，图上的Fourier变换形式就出现了

$$ f = \sum_{k=1 }^{n}<f, \phi_n> \cdot \phi_n $$

### D. Convolution
上一节中，我们引出了Fourier级数，那么在图上的卷积是如何实现的？

我们先来关注两个离散信号的卷积，$f = (f_1,...,f_n)^T, g=(g_1,...,g_n)^T$，二者的卷积可以写成如下形式，
![](/img/disc_conv.png 'source[2]'){:width="450px"}
为什么会有上面的等号那？这是因为上面那个矩阵叫circulant matrix，这种矩阵可以被离散Fourier正交基对角化。

类比Euclidean空间下的卷积性质，我们自然想到，通过图在Fourier频域下的频域信号的乘积得到，可以实现图上的卷积，自然有以下性质。
![](/img/spect_conv.png 'source[2]'){:width="400px"}


## Spectra domain Geometric Deep Learniing Methods
如何成功将深度学习引入到Graph上，最关键的两点是卷积与pooling，上面我们已经提到了如何由Euclidean空间类比设计图上的卷积，下面我们的主要任务是介绍如何设计可学习的卷积核。

首先来说，最简单的想法就是将我们在上一节D中的信号$g$替换成为一个可学习的卷积核$W$，那么便有以下形式

$$ y = \Phi W \Phi^Tf $$

其中，$W$是一个nXn的对角矩阵，因此可供学习的参数有n个。

但是这种方法会有以下几个问题，
![](/img/spect_conv_short.png 'source[2]'){:width="500px"}

根据我的理解，
1. 首先我们选取的正交基底会影响我们最后在图上卷积的结果，因为每次卷积的过程中我们都要用到这些正交基$\Phi$，因此，选取不同的正交基底对最后的卷积结果可能会有较大影响。
2. 如上面所提到的，每一层都需要学习n个参数，复杂度为$O(n)$，类比于CNN，我们通常会选取3X3或者5X5的卷积核，其复杂度为$O(1)$；
3. 每一层的计算量比较大，这些计算量主要来源于$\Phi$与$\Phi^T$；

### Smoothness

为了能够成功的通过卷积学习到图结构上信号的稳定特征，我们需要限制信号在频域中的平滑度，以此来换取在空间域中信号更好的表征，因此，在上面结果的基础上，我们需要对卷积核的设计做出改进。

第一点，就是通过一个平滑的函数来约束卷积核参数的取值，不能够为完全自由的取，因此，由下式

$$ \tau (\Delta)f = \Phi\tau(\Lambda )\Phi^Tf $$

其中，$\tau(\Delta)$为一个平滑的函数，可以利用插值等方法实现[6]。

### ChebNet

在[6]之后，Defferrard[7]和Kipf[8]同时提出了另外一种改进$\tau()$函数的方法，他们通过借鉴Chebyshev多项式[9]来改进卷积核的设计，使其既能够具有很好的locality，即表征局部特征的能力，又精简了每一层卷积核的参数量以及计算量。

通过将卷积核的形式设计成为多项式级数的形式，可以有效的解决locality的问题，同时可以降低卷积核的参数量，有下式形式

$$ \tau(\Delta) = \sum_{k=0}^{K-1} w_k \Lambda^k $$

其中，$K$代表节点$j$与$i$之间的最短距离，即hops，直意为两个节点之间需要跳几次能够跳过去。

这种形式的卷积核已经解决了两个重要的问题，但是没有能够减小每一层的计算复杂度。注意到Chebyshev级数在计算过程中可以利用递归进行计算，即有如下形式，

$$T_{n+1} = 2*T_n - T_{n-1}$$

因此，有

$$\tau(\Delta) = \sum_{k=0}^{K-1} w_k T_k(\Lambda) $$

即，

$$ 
\begin{align*} 
y &= \tau_w(L)x \\
  &= U\tau_w(\Lambda)U^Tx \\
  &= U \sum_{k=0}^{K-1} w_k T_k(\tilde{\Lambda})U^Tx \\
  &= \sum_{k=0}^{K-1} w_k T_k(\tilde L)x 
\end{align*}
$$

其中，
$$\tilde L = \frac{2L}{\lambda_{max}} - I_n$$
将Chebyshev多项式的输入限制在[-1,1]区间。

这样，通过Chebshev多项式，每一层上的计算复杂度降低到$O(Kn)$。

# References

[1] David I Shuman, Sunil K. Narang, Pascal Frossard, Antonio Ortega, Pierre Vandergheynst. 
The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains

[2] Michael Bronstein, Joan Bruna, arthur szlam, Xavier Bresson, Yann LeCun. 
Geometric Deep Learning on Graphs and Manifolds

[3] https://www.cnblogs.com/pinard/p/6221564.html#!comments

[4] https://en.wikipedia.org/wiki/Fourier_series

[5] https://en.wikipedia.org/wiki/Circulant_matrix

[6] Bruna, Zaremba, Szlam, LeCun, Spectra Networks and Deep Locally Connected Networks on Graphs

[7] Defferrard, Bresson, Vandergheynst 2016

[8] Kipf, Welling 2016

[9] https://en.wikipedia.org/wiki/Chebyshev_polynomials

