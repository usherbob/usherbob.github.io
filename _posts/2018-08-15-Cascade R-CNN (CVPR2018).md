---
layout:     post
title:      Cascade R-CNN
subtitle:   object detection
date:       2018-08-15
author:     加华
header-img: img/cvpr18logo.jpg
catalog: true
tags:
    - deep learning
    - object detection
---

>近期我会读大量的文献，将这些文献的收获po在博客上，第一篇论文是关于目标检测，附上原文地址：https://arxiv.org/abs/1712.00726

## Contribution

文章主要解决的问题是如何确定用于区分positive、negative proposal的IoU阈值，作者提出了通过级联多个detector，达到好的detection performance。

## 问题来源

首先要说明什么是IoU阈值，detector画出的bounding box中可能含有30%、50%、70%的被检测物体，我们需要设置一个阈值，从而能够高效的分离正负样例。

然而阈值设置的过大或过小会带来诸多问题，阈值过小时，会导致detector对于positive proposal的识别能力降低，并且引入大量负样例，导致模型训练时间增长；但

阈值设置太大，同样会导致一些问题：

- 1) overfitting during training, 正例样本会大量减少

- 2) IoU mismatch, 不同的IoU阈值适用于不同的样本集






