# Ultralytics-YOLO-DLC(模型剪枝、模型蒸馏)项目详细说明

本项目是基于 [Ultralytics-YOLO](https://github.com/z1069614715/objectdetection_script/blob/master/Ultralytics-YOLO-project.md) 项目的基础上开发**模型剪枝**和**模型蒸馏**.  
**模型剪枝**和**模型蒸馏**主要用于大论文的第四章工作，项目内有视频教程，也有群答疑。  

## 项目价格为99¥
## 支持YOLOv8、v10、v11、v12、26的目标检测、实例分割、姿态检测、旋转目标检测任务

### 首先剪枝是什么？  
模型剪枝是深度学习中的一种技术，旨在通过减少神经网络中不必要的参数和连接，来优化模型的效率和性能。模型剪枝可以分为结构剪枝和参数剪枝两种类型。  

### 为什么需要剪枝？  
剪枝可以很好地衡量模型轻量化程度与精度的关系,是替换轻量化结构完全没办法比的,比如我模型剪枝可以压缩百分之30的计算量,精度只下降了百分之1,但是你通过换模块来达到压缩百分之30的计算量,一般时间就会变长,因为大部分轻量化模块都是由时间换空间,而且精度还会下降得比较多,但是剪枝可以很好地避免这个问题.

### 剪枝的一些顾虑
大家关心最多的一个问题就是，我的结构能不能剪之类的，目前剪枝引擎是基于Torch_Pruning库的基础上再开发，具备自动跳层功能，经测试，绝大部分都可以顺利运行，但没办法保证所有的结构都能剪，有一定的风险，是否入手请自行考虑！  

### 首先蒸馏是什么？  
模型蒸馏（Model Distillation）是一种用于在计算机视觉中提高模型性能和效率的技术。在模型蒸馏中，通常存在两个模型，即“教师模型”和“学生模型”。

### 为什么需要蒸馏？  
1. 在不增加模型计算量和参数量的情况下提升精度，也即是可以无损提高精度。
2. 配合剪枝一起使用，可以尽量达到无损降低模型参数量、计算量，提高FPS的情况下，还能保持模型精度没有下降甚至上升，这是改进网络结构无法达到的高度。
3. 论文中的保底手段，因为剪枝和蒸馏的特殊性，其都不会增加参数量和计算量，可以在最后一个点上大幅度增加实验和工作量，因为本身蒸馏也需要做大量实验。

### 目前剪枝方法包含：
1. L1
2. Random
3. LAMP
4. GroupNorm

##### 注意：项目内不再附带需要稀疏训练的剪枝方法，因为稀疏训练太耗实验成本，而且还需要做多次稀疏训练的实验。

### 目前蒸馏方法包含：
1. Logical
    1. L1
    2. L2
    3. [BCKD](https://link.zhihu.com/?target=https%3A//arxiv.org//pdf/2308.14286)(Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection,ICCV 2023)
    4. Quality-Calibrated Decoupled Logical Distillation(自研)

2. Feature
    1. [Mimic](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Mimicking_Very_Efficient_CVPR_2017_paper.pdf)
    2. [Masked Generative Distillation](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2205.01529.pdf) (ECCV 2022)
    3. [Channel-wise Distillation](https://arxiv.org/pdf/2011.13256.pdf) (ICCV 2021)
    4. [ChSimLoss Distillation](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html) (ICCV2021)
    5. [SPKDLoss Distillation](https://arxiv.org/pdf/1907.09682.pdf) (ICCV2019)
    6. Detection-Guided Wavelet Feature Distillation(自研)