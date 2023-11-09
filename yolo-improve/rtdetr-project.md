# 目前自带的一些改进方案(持续更新)

# 为了感谢各位对RTDETR项目的支持,本项目的赠品是yolov5-PAGCP通道剪枝算法.[具体使用教程](https://www.bilibili.com/video/BV1yh4y1Z7vz/)

# 自带的一些文件说明
1. train.py
    训练模型的脚本
2. main_profile.py
    输出模型和模型每一层的参数,计算量的脚本(rtdetr-l和rtdetr-x因为thop库的问题,没办法正常输出每一层的参数和计算量和时间)
3. val.py
    使用训练好的模型计算指标的脚本
4. detect.py
    推理的脚本
5. track.py
    跟踪推理的脚本
6. heatmap.py
    生成热力图的脚本
7. get_FPS.py
    计算模型储存大小、模型推理时间、FPS的脚本
8. get_COCO_metrice.py
    计算COCO指标的脚本
9. plot_result.py
    绘制曲线对比图的脚本

# RT-DETR基准模型

1. ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml

    rtdetr-r18 summary: 421 layers, 20184464 parameters, 20184464 gradients, 58.6 GFLOPs
2. ultralytics/cfg/models/rt-detr/rtdetr-r34.yaml

    rtdetr-r34 summary: 501 layers, 30292624 parameters, 30292624 gradients, 88.9 GFLOPs
3. ultralytics/cfg/models/rt-detr/rtdetr-r50-m.yaml

    rtdetr-r50-m summary: 637 layers, 36647020 parameters, 36647020 gradients, 98.3 GFLOPs
4. ultralytics/cfg/models/rt-detr/rtdetr-r50.yaml

    rtdetr-r50 summary: 629 layers, 42944620 parameters, 42944620 gradients, 134.8 GFLOPs
5. ultralytics/cfg/models/rt-detr/rtdetr-r101.yaml

    rtdetr-r101 summary: 867 layers, 76661740 parameters, 76661740 gradients, 257.7 GFLOPs
6. ultralytics/cfg/models/rt-detr/rtdetr-l.yaml

    rtdetr-l summary: 673 layers, 32970732 parameters, 32970732 gradients, 108.3 GFLOPs
7. ultralytics/cfg/models/rt-detr/rtdetr-x.yaml

    rtdetr-x summary: 867 layers, 67468108 parameters, 67468108 gradients, 232.7 GFLOPs
# RT-DETR改进方案

### 以RT-DETR-R18为基准模型的改进方案
1. ultralytics/cfg/models/rt-detr/rt-detr-timm.yaml

    使用timm库系列的主干替换rtdetr的backbone.
2. ultralytics/cfg/models/rt-detr/rt-detr-fasternet.yaml

    使用(CVPR2023)fasternet替换rtdetr的backbone.
3. ultralytics/cfg/models/rt-detr/rt-detr-EfficientViT.yaml

    使用(CVPR2023)EfficientViT替换rtdetr的backbone.
4. ultralytics/cfg/models/rt-detr/rtdetr-convnextv2.yaml

    使用ConvNextV2替换rtdetr的backbone.
5. ultralytics/cfg/models/rt-detr/rtdetr-EfficientFormerv2.yaml

    使用EfficientFormerv2替换rtdetr的backbone.
6. ultralytics/cfg/models/rt-detr/rtdetr-repvit.yaml

    使用repvit替换rtdetr的backbone.
7. ultralytics/cfg/models/rt-detr/rtdetr-CSwomTramsformer.yaml

    使用CSwomTramsformer替换rtdetr的backbone.
8. ultralytics/cfg/models/rt-detr/rtdetr-VanillaNet.yaml

    使用VanillaNet替换rtdetr的backbone.
9. ultralytics/cfg/models/rt-detr/rtdetr-SwinTransformer.yaml

    使用SwinTransformer替换rtdetr的backbone.
10. ultralytics/cfg/models/rt-detr/rtdetr-lsknet.yaml

    使用LSKNet(2023旋转目标检测SOTA的主干)替换rtdetr的backbone.

11. ultralytics/cfg/models/rt-detr/rtdetr-CascadedGroupAttention.yaml

    使用(CVPR2023)EfficientViT中的CascadedGroupAttention改进rtdetr中的AIFI.(详细请看百度云视频-rtdetr-CascadedGroupAttention说明)

12. ultralytics/cfg/models/rt-detr/rtdetr-DWRC3.yaml

    使用[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块构建DWRC3改进rtdetr.(详细介绍请看百度云视频-20231109更新说明)

### 以RT-DETR-L为基准模型的改进方案
1. ultralytics/cfg/models/rt-detr/rtdetr-l-GhostHGNetV2.yaml

    使用GhostConv改进HGNetV2.(详细介绍请看百度云视频)
2. ultralytics/cfg/models/rt-detr/rtdetr-l-RepHGNetV2.yaml

    使用RepConv改进HGNetV2.(详细介绍请看百度云视频)

### 以Yolov8为基准模型的改进方案
1. ultralytics/cfg/models/yolo-detr/yolov8-detr.yaml

    使用RT-DETR中的TransformerDecoderHead改进yolov8.

2. ultralytics/cfg/models/yolo-detr/yolov8-detr-DWR.yaml

    使用RT-DETR中的TransformerDecoderHead和[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块改进yolov8.

3. ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml

    使用RT-DETR中的TransformerDecoderHead和(CVPR2023)FasterNet改进yolov8.(支持替换其他主干,请看百度云视频-替换主干示例教程)

### 以Yolov5为基准模型的改进方案
1. ultralytics/cfg/models/yolo-detr/yolov5-detr.yaml

    使用RT-DETR中的TransformerDecoderHead改进yolov5.

2. ultralytics/cfg/models/yolo-detr/yolov5-detr-DWR.yaml

    使用RT-DETR中的TransformerDecoderHead和[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块改进yolov5.

3. ultralytics/cfg/models/yolo-detr/yolov5-detr-fasternet.yaml

    使用RT-DETR中的TransformerDecoderHead和(CVPR2023)FasterNet改进yolov5.(支持替换其他主干,请看百度云视频-替换主干示例教程)

# 更新公告
- **20231105-rtdetr-v1.0**
    1. 初版项目发布.

- **20231109-rtdetr-v1.1**
    1. 修复断点训练不能正常使用的bug.
    2. 优化get_FPS.py中的模型导入方法.
    3. 增加以yolov5和yolov8为基准模型更换为RTDETR的Head,后续也会提供yolov5-detr,yolov8-detr相关的改进.
    4. 新增百度云视频-20231109使用说明视频和替换主干说明视频.
    5. 新增GhostHGNetV2,RepHGNetV2,详细请看使用教程中的RT-DETR改进方案.
    6. 新增使用DWRSeg中的Dilation-wise Residual(DWR)模块,加强从网络高层的可扩展感受野中提取特征,详细请看使用教程中的RT-DETR改进方案.