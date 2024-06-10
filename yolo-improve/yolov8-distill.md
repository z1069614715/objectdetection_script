# YOLOV8蒸馏项目介绍

### 首先蒸馏是什么？  
模型蒸馏（Model Distillation）是一种用于在计算机视觉中提高模型性能和效率的技术。在模型蒸馏中，通常存在两个模型，即“教师模型”和“学生模型”。

### 为什么需要蒸馏？  
1. 在不增加模型计算量和参数量的情况下提升精度，也即是可以无损提高精度。
2. 配合剪枝一起使用，可以尽量达到无损降低模型参数量、计算量，提高FPS的情况下，还能保持模型精度没有下降甚至上升，这是改进网络结构无法达到的高度。
3. 论文中的保底手段，因为剪枝和蒸馏的特殊性，其都不会增加参数量和计算量，可以在最后一个点上大幅度增加实验和工作量，因为本身蒸馏也需要做大量实验。

### 目前蒸馏方法包含：
1. Logical
    1. L1
    2. L2
    3. [BCKD](https://link.zhihu.com/?target=https%3A//arxiv.org//pdf/2308.14286)(Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection,ICCV 2023)
2. Feature
    1. Mimic
    2. [Masked Generative Distillation](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2205.01529.pdf) (ECCV 2022)
    3. [Channel-wise Distillation](https://arxiv.org/pdf/2011.13256.pdf) (ICCV 2021)
    4. [ChSimLoss Distillation](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html) (ICCV2021)
    5. [SPKDLoss Distillation](https://arxiv.org/pdf/1907.09682.pdf) (ICCV2019)

### 知识蒸馏的一些细节(具体项目会提供视频讲解)
1. Feature蒸馏可以自定义选择层进行蒸馏.
2. 蒸馏损失支持常数,线性,余弦进行动调整.
3. 支持Logical和Feature一起使用.
4. 过程中会输出Logical和Feature的损失,让用户可以及时调整对应的损失系数.
5. 支持正常训练模型时候进行蒸馏和剪枝后finetune蒸馏.
6. 支持自蒸馏.

# 实验示例结果.(以下示例实验相关命令,视频教程,实验数据都在项目里面)
#### Dataset:VisDrone(训练集只用了百分之30的数据,验证集和测试集用了全量的数据) Teacher:yolov8s Student:yolov8n (no pretrained weight)
| model | GFLOPs | mAP50(test set) | mAP50-95(test set) |
| :----: | :----: | :----: | :----: |
| yolov8n | 8.1 | 0.202 | 0.108 |
| yolov8s | 28.5 | 0.234 | 0.128 |
| yolov8n CWD Exp1 | 8.1 | 0.211(+0.009) | 0.114(+0.006) |
| yolov8n CWD Exp2 | 8.1 | 0.208(+0.006) | 0.112(+0.004) |
| yolov8n CWD Exp3 | 8.1 | 0.21(+0.008) | 0.112(+0.004) |
| yolov8n Mimic Exp1 | 8.1 | 0.203(+0.001) | 0.108(+0.0) |
| yolov8n Mimic Exp2 | 8.1 | 0.204(+0.002) | 0.107(-0.001) |
| yolov8n l2 Exp1 | 8.1 | 0.196(-0.006) | 0.106(-0.002) |
| yolov8n BCKD Exp1 | 8.1 | 0.208(+0.006) | 0.112(+0.004) |
| yolov8n BCKD Exp2 | 8.1 | 0.206(+0.004) | 0.106(-0.002) |
| yolov8n BCKD Exp3 | 8.1 | 0.209(+0.007) | 0.113(+0.005) |
| yolov8n BCKD Exp4 | 8.1 | 0.204(+0.002) | 0.11(+0.002) |
| yolov8n BCKD+CWD Exp1 | 8.1 | 0.204(+0.002) | 0.109(+0.001) |
| yolov8n BCKD+CWD Exp2 | 8.1 | 0.214(+0.012) | 0.115(+0.007) |
| yolov8n BCKD+CWD Exp3 | 8.1 | 0.21(+0.008) | 0.114(+0.006) |
| yolov8n BCKD+CWD Exp4 | 8.1 | 0.208(+0.006) | 0.113(+0.005) |

#### Dataset:VisDrone(训练集只用了百分之30的数据,验证集和测试集用了全量的数据) Teacher:yolov8s Student:yolov8n-lamp (use pretrained weight)
| model | GFLOPs | mAP50(test set) | mAP50-95(test set) |
| :----: | :----: | :----: | :----: |
| yolov8n | 8.1 | 0.225 | 0.124 |
| yolov8n-lamp | 3.2 | 0.225 | 0.123(-0.001) |
| yolov8s | 28.5 | 0.259 | 0.146 |
| yolov8n-lamp cwd exp1 | 3.2 | 0.23(+0.005) | 0.124(0.0) |

#### Dataset:VisDrone(训练集只用了百分之30的数据,验证集和测试集用了全量的数据) Teacher:yolov8s-asf-p2 Student:yolov8s-asf-p2
| model | GFLOPs | mAP50(test set) | mAP50-95(test set) |
| :----: | :----: | :----: | :----: |
| yolov8n-asf-p2 | 12.0 | 0.237 | 0.127 |
| yolov8s-asf-p2 | 35.8 | 0.282 | 0.155 |
| yolov8n-asf-p2 cwd exp1 | 12.0 | 0.24(+0.003) | 0.129(+0.002) |
| yolov8n-asf-p2 cwd exp2 | 12.0 | 0.239(+0.002) | 0.128(+0.001) |
| yolov8n-asf-p2 cwd exp3 | 12.0 | 0.236(-0.001) | 0.125(-0.002) |
| yolov8n-asf-p2 cwd exp4 | 12.0 | 0.239(+0.002) | 0.128(+0.001) |
| yolov8n-asf-p2 cwd exp5 | 12.0 | 0.234(-0.004) | 0.125(-0.002) |
| yolov8n-asf-p2 mgd exp1 | 12.0 | 0.234(-0.004) | 0.125(-0.002) |
| yolov8n-asf-p2 mgd exp2 | 12.0 | 0.238(+0.001) | 0.127(0.0) |
| yolov8n-asf-p2 BCKD exp1 | 12.0 | 0.241(+0.004) | 0.131(+0.004) |
| yolov8n-asf-p2 BCKD exp2 | 12.0 | 0.24(+0.003) | 0.13(+0.003) |
| yolov8n-asf-p2 cwd+BCKD exp1 | 12.0 | 0.241(+0.004) | 0.131(+0.004) |
| yolov8n-asf-p2 cwd+BCKD exp2 | 12.0 | 0.239(+0.002) | 0.128(+0.001) |