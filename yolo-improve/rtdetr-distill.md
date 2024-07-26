# RTDETR蒸馏项目介绍

### 首先蒸馏是什么？  
模型蒸馏（Model Distillation）是一种用于在计算机视觉中提高模型性能和效率的技术。在模型蒸馏中，通常存在两个模型，即“教师模型”和“学生模型”。

### 为什么需要蒸馏？  
1. 在不增加模型计算量和参数量的情况下提升精度，也即是可以无损提高精度。
2. 论文中的保底手段，因为蒸馏的特殊性，其都不会增加参数量和计算量，可以在最后一个点上大幅度增加实验和工作量，因为本身蒸馏也需要做大量实验。
3. 如果在模型改进过程中进行了轻量化，但是精度降低得有点多，可以尝试使用知识蒸馏来弥补轻量化带来的精度丢失问题。

### 目前蒸馏方法包含：
1. Logical
    1. RTDETRLogicLoss(根据rtdetr的特点进行开发的逻辑蒸馏)
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
#### Dataset:Visdrone(训练集只用了2500张图,验证集和测试集用了全量的数据) 为了加速实验,老师选择了yolov8s-detr,学生选择了yolov8n-detr

| model | GFLOPs | mAP50(test set) | mAP50-95(test set) |
| :----: | :----: | :----: | :----: |
| yolov8n-detr | 11.7 | 0.266 | 0.146 |
| yolov8s-detr | 27.3 | 0.286 | 0.161 |
| yolov8n-detr logloss exp1 | 11.7 | 0.272(+0.006) | 0.153(+0.007) |
| yolov8n-detr logloss exp2 | 11.7 | 0.278(+0.012) | 0.157(+0.011) |
| yolov8n-detr logloss exp3 | 11.7 | 0.271(+0.005) | 0.154(+0.008) |
| yolov8n-detr logloss exp4 | 11.7 | 0.282(+0.016) | 0.160(+0.014) |
| yolov8n-detr cwd exp1 | 11.7 | 0.255(-0.011) | 0.139(-0.007) |
| yolov8n-detr cwd exp2 | 11.7 | 0.267(+0.001) | 0.148(+0.002) |
| yolov8n-detr cwd exp3 | 11.7 | 0.268(+0.002) | 0.149(+0.003) |
| yolov8n-detr cwd exp4 | 11.7 | 0.261(-0.005) | 0.146(0.000) |
| yolov8n-detr cwd exp5 | 11.7 | 0.266(0.000) | 0.147(+0.001) |
| yolov8n-detr cwd exp6 | 11.7 | 0.264(-0.002) | 0.146(0.000) |
| yolov8n-detr cwd exp7 | 11.7 | 0.260(-0.006) | 0.144(-0.002) |
| yolov8n-detr cwd exp8 | 11.7 | 0.268(+0.002) | 0.148(+0.002) |
| yolov8n-detr cwd exp9 | 11.7 | 0.269(+0.003) | 0.149(+0.003) |
| yolov8n-detr cwd exp10 | 11.7 | 0.267(+0.001) | 0.147(+0.001) |
| yolov8n-detr cwd exp11 | 11.7 | 0.257(-0.009) | 0.141(-0.005) |
| yolov8n-detr mgd exp1 | 11.7 | 0.271(+0.005) | 0.152(+0.006) |
| yolov8n-detr mgd exp2 | 11.7 | 0.265(-0.001) | 0.148(+0.002) |
| yolov8n-detr mgd exp3 | 11.7 | 0.269(+0.003) | 0.150(+0.004) |
| yolov8n-detr mgd exp4 | 11.7 | 0.265(-0.001) | 0.147(+0.001) |
| yolov8n-detr mgd exp5 | 11.7 | 0.264(-0.002) | 0.146(0.000) |
| yolov8n-detr mgd exp6 | 11.7 | 0.270(+0.004) | 0.151(+0.005) |
| yolov8n-detr mgd exp7 | 11.7 | 0.260(-0.006) | 0.145(-0.001) |
| yolov8n-detr mgd exp8 | 11.7 | 0.271(+0.005) | 0.152(+0.006) |
| yolov8n-detr shsim exp1 | 11.7 | 0.264(-0.002) | 0.147(+0.001) |
| yolov8n-detr shsim exp2 | 11.7 | 0.266(0.000) | 0.148(+0.002) |
| yolov8n-detr shsim exp3 | 11.7 | 0.260(-0.006) | 0.143(-0.003) |
| yolov8n-detr spkd exp1 | 11.7 | 0.259(-0.007) | 0.143(-0.003) |
| yolov8n-detr spkd exp2 | 11.7 | 0.256(-0.010) | 0.142(-0.004) |
| yolov8n-detr spkd exp3 | 11.7 | 0.262(-0.004) | 0.145(-0.001) |
| yolov8n-detr logloss-mgd exp1 | 11.7 | 0.277(+0.011) | 0.157(+0.011) |
| yolov8n-detr logloss-cwd exp1 | 11.7 | 0.274(+0.008) | 0.151(+0.005) |
| yolov8n-detr logloss-cwd exp2 | 11.7 | 0.272(+0.006) | 0.153(+0.007) |