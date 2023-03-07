# CV-Attention
关于CV的一些经典注意力机制代码。  
目前代码格式主要用于yolov3,yolov5,yolov7,yolov8.

# Supports
| name | need_chaneel | paper |
| :----:| :----: | :----: |
| BAM | True | https://arxiv.org/pdf/1807.06514.pdf |
| CBAM | True | https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf |
| SE | True | https://arxiv.org/abs/1709.01507 |
| CoTAttention | True | https://arxiv.org/abs/2107.12292 |
| MobileViTAttention | True | https://arxiv.org/abs/2110.02178 |
| SimAM | False | http://proceedings.mlr.press/v139/yang21o/yang21o.pdf |
| SK | True | https://arxiv.org/pdf/1903.06586.pdf |
| ShuffleAttention | True | https://arxiv.org/pdf/2102.00240.pdf |
| S2Attention | True | https://arxiv.org/abs/2108.01072 |
| TripletAttention | False | https://arxiv.org/abs/2010.03045 |
| ECA | False | https://arxiv.org/pdf/1910.03151.pdf |
| ParNetAttention | True | https://arxiv.org/abs/2110.07641 |
| CoordAttention | True | https://arxiv.org/abs/2103.02907 |
| MHSA<br>Multi-Head-Self-Attention | True | https://wuch15.github.io/paper/EMNLP2019-NRMS.pdf |
| SGE | False | https://arxiv.org/pdf/1905.09646.pdf |
| A2Attention | True | https://arxiv.org/pdf/1810.11579.pdf |
| GC<br>Global Context Attention | True | https://arxiv.org/abs/1904.11492 |
| EffectiveSE<br>Effective Squeeze-Excitation | True | https://arxiv.org/abs/1911.06667 |
| GE<br>Gather-Excite Attention | True | https://arxiv.org/abs/1810.12348 |
| CrissCrossAttention | True | https://arxiv.org/abs/1811.11721 |
| Polarized Self-Attention | True | https://arxiv.org/abs/2107.00782 |
| Sequential Self-Attention | True | https://arxiv.org/abs/2107.00782 |
| GAM | True | https://arxiv.org/pdf/2112.05561v1.pdf |


# Install
部分注意力需要安装timm. 如运行提示缺少timm安装即可. 安装命令:pip install timm

# Course
1. [yolov5添加注意力哔哩哔哩视频教学链接](https://www.bilibili.com/video/BV1s84y1775U) [yolov5添加注意力-补充事项-哔哩哔哩视频教学链接](https://www.bilibili.com/video/BV1hG4y1M71X)
2. [yolov7添加注意力哔哩哔哩视频教学链接](https://www.bilibili.com/video/BV1pd4y1H7BK)
3. [yolov8添加注意力哔哩哔哩视频教学链接](https://www.bilibili.com/video/BV1dv4y167rF)

# Reference
https://github.com/xmu-xiaoma666/External-Attention-pytorch  
https://github.com/rwightman/pytorch-image-models