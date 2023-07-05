# 目前自带的一些改进方案(持续更新)

### YOLOV5 (AnchorFree+DFL+TAL) [官方预训练权重github链接](https://github.com/ultralytics/assets/releases)
#### YOLOV5的使用方式跟YOLOV8一样,就是选择配置文件选择v5的即可.
1. ultralytics/models/v5/yolov5-fasternet.yaml

    fasternet替换yolov5主干.
2. ultralytics/models/v5/yolov5-timm.yaml

    使用timm支持的主干网络替换yolov5主干.
3. ultralytics/models/v5/yolov5-dyhead.yaml

    添加基于注意力机制的目标检测头到yolov5中.
4. 增加Adaptive Training Sample Selection匹配策略.

    在ultralytics/yolo/utils/loss.py中的class v8DetectionLoss中自行选择对应的self.assigner即可.  
    此ATSS匹配策略目前占用显存比较大,因此使用的时候需要设置更小的batch,后续会进行优化这一功能.
5. Asymptotic Feature Pyramid Network[reference](https://github.com/gyyang23/AFPN/tree/master)

    a. ultralytics/models/v5/yolov5-AFPN-P345.yaml  
    b. ultralytics/models/v5/yolov5-AFPN-P345-Custom.yaml  
    c. ultralytics/models/v5/yolov5-AFPN-P2345.yaml  
    d. ultralytics/models/v5/yolov5-AFPN-P2345-Custom.yaml  
    其中Custom中的block支持:C2f, C2f_Faster, C2f_ODConv, C2f-Faster-EMA, C2f-DBB, VoVGSCSP, VoVGSCSPC, C3(default), C3Ghost

### YOLOV8
1. ultralytics/models/v8/yolov8-efficientViT.yaml

    efficientViT替换yolov8主干,训练时候如果出现nan,需要在训练的时候加上--unamp关闭AMP混合精度训练.
2. ultralytics/models/v8/yolov8-fasternet.yaml

    fasternet替换yolov8主干.
3. ultralytics/models/v8/yolov8-timm.yaml

    使用timm支持的主干网络替换yolov8主干.
4. ultralytics/models/v8/yolov8-convnextv2.yaml

    使用convnextv2网络替换yolov8主干.
5. ultralytics/models/v8/yolov8-dyhead.yaml

    添加基于注意力机制的目标检测头到yolov8中.
6. ultralytics/models/v8/yolov8-bifpn.yaml

    添加BIFPN到yolov8中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default)  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT)
    2. node_mode  
        其中目前(后续会更新喔)支持这些结构选择: C2f(default), C2f_Faster, C2f_ODConv, C2f-Faster-EMA, C2f-DBB, VoVGSCSP, VoVGSCSPC, C3, C3Ghost
    3. head_channel  
        BIFPN中的通道数,默认设置为256.
7. ultralytics/models/v8/yolov8-C2f-Faster.yaml

    使用C2f-Faster替换C2f.(使用FasterNet中的FasterBlock替换C2f中的Bottleneck)
8. ultralytics/models/v8/yolov8-C2f-ODConv.yaml

    使用C2f-ODConv替换C2f.(使用ODConv替换C2f中的Bottleneck中的Conv)
9. ultralytics/models/v8/yolov8-EfficientFormerV2.yaml

    使用EfficientFormerV2网络替换yolov8主干.  
    运行train.py中的时候需要在ultralytics/yolo/v8/detect/train.py的DetectionTrainer class中的build_dataset函数中的rect=(True if mode == 'val' else False)中的True改为False.这个是因为EfficientFormerV2固定输入640x640导致的,其他模型可以修改回去.  
    运行val.py不需要改.  
    运行detect.py中的时候需要在ultralytics/yolo/engine/predictor.py找到函数def pre_transform(self, im),在LetterBox中的auto改为False,其他模型可以修改回去.  
10. ultralytics/models/v8/yolov8-C2f-Faster-EMA.yaml

    使用C2f-Faster-EMA替换C2f.(C2f-Faster-EMA推荐可以放在主干上,Neck和head部分可以选择C2f-Faster)
11. ultralytics/models/v8/yolov8-C2f-DBB.yaml

    使用C2f-DBB替换C2f.(使用DiverseBranchBlock替换C2f中的Bottleneck中的Conv)
12. 增加Adaptive Training Sample Selection匹配策略.

    在ultralytics/yolo/utils/loss.py中的class v8DetectionLoss中自行选择对应的self.assigner即可.  
    此ATSS匹配策略目前占用显存比较大,因此使用的时候需要设置更小的batch,后续会进行优化这一功能.
13. ultralytics/models/v8/yolov8-slimneck.yaml

    使用VoVGSCSP\VoVGSCSPC和GSConv替换yolov8 neck中的C2f和Conv.
14. ultralytics/models/v8/yolov8-attention.yaml

    可以看项目视频-如何在yaml配置文件中添加注意力层  
    多种注意力机制在yolov8中的使用. [多种注意力机制github地址](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)

15. Asymptotic Feature Pyramid Network[reference](https://github.com/gyyang23/AFPN/tree/master)

    a. ultralytics/models/v8/yolov8-AFPN-P345.yaml  
    b. ultralytics/models/v8/yolov8-AFPN-P345-Custom.yaml  
    c. ultralytics/models/v8/yolov8-AFPN-P2345.yaml  
    d. ultralytics/models/v8/yolov8-AFPN-P2345-Custom.yaml  
    其中Custom中的block支持:C2f(default), C2f_Faster, C2f_ODConv, C2f-Faster-EMA, C2f-DBB, VoVGSCSP, VoVGSCSPC, C3, C3Ghost