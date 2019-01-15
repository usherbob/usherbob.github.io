---
layout:     post
title:      Graph Deep Learning
subtitle:   for beginners
date:       2018-12-23
author:     加华
header-img: img/Graph_DL.jpeg
catalog: true
mathjax: true
tags:
    - deep learning
    - graph
---

# Preface

在调研图深度的过程中，从零开始起步，将自己的论文总结记录在这里。

# The Emerging Field of Signal Processing on Graphs[1]

对于图结构信号的处理，主要是两种方式，Vertex domain和 Graph spectra domain，分别相当于欧氏空间中，我们对信号在空间域与频率域的处理方式。

The vertex domain designs of graph wavelet transforms are based on the spatial features of the graph, such as node connectivity and distances between vertices. 

The graph spectral domain designs of graph wavelets are based on the spectral features of the graph, which are encoded, e.g., in the eigenvalues and eigenvectors of one of the Laplacian matrix. The general idea of the graph spectral designs is to construct bases that are localized in both thevertex and graph spectral domains.

## 1.graph spectra与fourier transform之间的联系

### A. weighted matrix and graph signals

主要是针对一些本身不是图结构的空间结构，如何构造？ 例如一张二维图像，我们可以通过度量任意两个像素之间的街区距离定义权重，也可以度量两个像素之间的灰度值相似度定义，比较经典的是高斯核距离（式(1)）。

其次，什么是图上的信号，我是这样理解的，如同图像中每个像素点都有对应的灰度值，在图结构中，所有顶点上的信号构成了整张图的信号。

### B. Non-normalized Graph Laplacian

Laplacian算子是一个二阶差分算子，所以从它的名字，就能知道它是来干差分这个活，第一种定义是L=D-W，问什么这么定义还不清楚。但是他有比较好的性质，对称矩阵，可以作特征值分解。

### C. A Graph Fourier Transform and Notion of Frequency

类比一维时域信号的变换，可以得到Graph Fourier的表达式(式（3）)，对于图的傅里叶变换，是以Laplacian矩阵特征向量为底进行。

The graph Laplacian eigenvectors associated with low frequencies λ vary slowly across the graph;The eigenvectors associated with larger eigenvalues oscillate more rapidly and are more likely to have dissimilar values on vertices connected by an edge with high weight.图3可以很好的阐释这一点

### D. Graph Signal Representations in Two Dimensions

The graph Fourier transform (3) and its inverse (4) give
us a way to equivalently represent a signal in two different
domains: the vertex domain and the graph spectral domain.
While we often start with a signal g in the vertex domain,
it may also be useful to define a signal ĝ directly in the
graph spectral domain. We refer to such signals as kernels.


### E. Discrete Calculus and Signal Smoothness with Respect to the Intrinsic Structure of the Graph

对于图smoothness的度量比较好理解，感觉上像对于图像中边缘或者角点的度量。

Equation (8) explains why the graph
Laplacian eigenvectors associated with lower eigenvalues are
smoother, and provides another interpretation for why the
graph Laplacian spectrum carries a notion of frequency.

### F. Other Graph Matrices

## 2.Generalized Operators for Signals on Graphs

所有这些操作都可以在vertex domain或者spectra domain进行，只不过某些操作可能很难由时域或者频率域中的处理方式类比过来。其实，对于图深度来说，我最主要关心filtering和Downsampling这两部分。

### A. Filtering

1. Graph Spctra Filtering

	或者称其为频域滤波

2. Vertex Domain Filtering

To filter a signal in the
vertex domain, we simply write the output f out (i) at vertex i
as a linear combination of the components of the input signal
at vertices within a K-hop local neighborhood of vertex i.

### E. Graph Coarsening, Downsampling, and Reduction

Under Updating

# NIPS Tutorial-Geometric Deep Learning on Graphs and Manifolds [2]

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

