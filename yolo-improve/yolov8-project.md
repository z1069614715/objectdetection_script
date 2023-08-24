# 目前自带的一些改进方案(持续更新)

# 为了感谢各位对V8项目的支持,本项目的赠品是yolov5-PAGCP通道剪枝算法.[具体使用教程](https://www.bilibili.com/video/BV1yh4y1Z7vz/)

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
    其中Custom中的block支持:C2f, C2f_Faster, C2f_ODConv, C2f_Faster_EMA, C2f_DBB, VoVGSCSP, VoVGSCSPC, C3(default), C3Ghost, C3_CloAtt

6. ultralytics/models/v5/yolov5-bifpn.yaml

    添加BIFPN到yolov5中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default)  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT)
    2. node_mode  
        其中目前(后续会更新喔)支持这些结构选择: C2f, C2f_Faster, C2f_ODConv, C2f_Faster_EMA, C2f_DBB, C2f_CloAtt, VoVGSCSP, VoVGSCSPC, C3(default), C3Ghost, C3_CloAtt
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

7. ultralytics/models/v5/yolov5-C3-CloAtt.yaml

    使用C3-CloAtt替换C3.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C3中的Bottleneck中)

8. ultralytics/models/v5/yolov5-RevCol.yaml

    使用(ICLR2023)Reversible Column Networks对yolov5主干进行重设计.

9. ultralytics/models/v5/yolov5-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolov5主干.

10. ultralytics/models/v5/yolov5-C3-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C3融合.

11. ultralytics/models/v5/yolov5-C3-SCcConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C3融合.  
    (取名为SCcConv的原因是在windows下命名是不区分大小写的)

12. MPDiou.[论文链接](https://arxiv.org/pdf/2307.07662v1.pdf)

    在ultralytics/yolo/utils/loss.py中的BboxLoss class中的forward函数里面进行更换对应的iou计算方式.

13. ultralytics/models/v5/yolov5-LAWDS.yaml

    Light Adaptive-weight downsampling.自研模块,具体讲解请看百度云链接中的视频.

14. ultralytics/models/v5/yolov5-C3-EMSC.yaml

    Efficient Multi-Scale Conv.自研模块,具体讲解请看百度云链接中的视频.

15. ultralytics/models/v5/yolov5-C3-EMSCP.yaml

    Efficient Multi-Scale Conv Plus.自研模块,具体讲解请看百度云链接中的视频.

16. ultralytics/models/v5/yolov5-RCSOSA.yaml

    使用[RCS-YOLO](https://github.com/mkang315/RCS-YOLO/tree/main)中的RCSOSA替换C3.

### YOLOV8
1. ultralytics/models/v8/yolov8-efficientViT.yaml

    (CVPR2023)efficientViT替换yolov8主干.
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
        其中目前(后续会更新喔)支持这些结构选择: C2f(default), C2f_Faster, C2f_ODConv, C2f_Faster_EMA, C2f_DBB, C2f_CloAtt, VoVGSCSP, VoVGSCSPC, C3, C3Ghost, C3_CloAtt
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
    其中Custom中的block支持:C2f(default), C2f_Faster, C2f_ODConv, C2f_Faster_EMA, C2f_DBB, C2f_CloAtt, VoVGSCSP, VoVGSCSPC, C3, C3Ghost, C3_CloAtt

16. ultralytics/models/v8/yolov8-vanillanet.yaml

    vanillanet替换yolov8主干.

17. ultralytics/models/v8/yolov8-C2f-CloAtt.yaml

    使用C2f-CloAtt替换C2f.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C2f中的Bottleneck中)

18. ultralytics/models/v8/yolov8-RevCol.yaml

    使用(ICLR2023)Reversible Column Networks对yolov8主干进行重设计.

19. ultralytics/models/v8/yolov8-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolov8主干.

20. ultralytics/models/v8/yolov8-C2f-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C2f融合.

21. ultralytics/models/v8/yolov8-C2f-SCcConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C2f融合.  
    (取名为SCcConv的原因是在windows下命名是不区分大小写的)

22. MPDiou.[论文链接](https://arxiv.org/pdf/2307.07662v1.pdf)

    在ultralytics/yolo/utils/loss.py中的BboxLoss class中的forward函数里面进行更换对应的iou计算方式.

23. ultralytics/models/v8/yolov8-LAWDS.yaml

    Light Adaptive-weight downsampling.自研模块,具体讲解请看百度云链接中的视频.

24. ultralytics/models/v8/yolov8-C2f-EMSC.yaml

    Efficient Multi-Scale Conv.自研模块,具体讲解请看百度云链接中的视频.

25. ultralytics/models/v8/yolov8-C2f-EMSCP.yaml

    Efficient Multi-Scale Conv Plus.自研模块,具体讲解请看百度云链接中的视频.

26. ultralytics/models/v8/yolov8-RCSOSA.yaml

    使用[RCS-YOLO](https://github.com/mkang315/RCS-YOLO/tree/main)中的RCSOSA替换C2f.

# 更新公告

- **20230620-yolov8-v1.1**
    1. 增加EMA,C2f-Faster-EMA.
    2. val.py增加batch选择.
    3. train.py增加resume断点续训.

- **20230625-yolov8-v1.2**
    1. 使用说明和视频增加断点续训教程.
    2. 增加 使用C2f-DBB替换C2f.(使用DiverseBranchBlock替换C2f中的Bottleneck中的Conv) C2f-DBB同样可以用在bifpn中的node.
    3. 使用说明中增加常见错误以及解决方案.

- **20230627-yolov8-v1.3**
    1. 增加Adaptive Training Sample Selection匹配策略.
    2. val.py增加save_txt参数.
    3. 更新使用教程.

- **20230701-yolov8-v1.4**
    1. val.py中增加imgsz参数，可以自定义val时候的图片尺寸，默认为640.
    2. 增加plot_result.py，用于绘制对比曲线图，详细请看使用说明13点.
    3. 支持计算COCO评价指标.详细请看使用说明12点.
    4. 增加yolov8-slimneck.其中VoVGSCSP\VoVGSCSPC支持在bifpn中使用,支持GSConv的替换.

- **20230703-yolov8-v1.5**
    1. 修正计算gflops.
    2. 增加YOLOV5-AnchorFree改进，详细可看使用教程.md
    3. 增加yolov8-attention.yaml，并附带视频如何在yaml中添加注意力层
    4. 更新train.py --info参数的功能，增加打印每一层的参数，增加模型融合前后的层数，参数量，计算量对比。

- **20230705-yolov8-v1.6**
    1. yolov5和yolov8 支持 Asymptotic Feature Pyramid Network.

- **20230714-yolov8-v1.7**
    1. 把添加的所有模块全部转移到ultralytics/nn/extra_modules，以便后面进行同步代码。
    2. 增加yolov5-bifpn。
    3. 修正ultralytics/models/v8/yolov8-efficientViT.yaml，经粉丝反映，EfficientViT存在同名论文，本次更新的EfficientViT更适合目标检测，之前的efficientViT的原文是在语义分割上进行提出的。
    4. 更新使用教程。
    5. 更新import逻辑，现在不需要安装mmcv也可以进行使用，但是没有安装mmcv的使用dyhead会进行报错，降低上手难度。

- **20230717-yolov8-v1.8**
    1. 修正vanillanet主干进行fuse后没法计算GFLOPs的bug.
    2. 添加yolov8-C2f-CloAtt,yolov5-C3-CloAtt.
    3. 添加yolov8-vanillanet.yaml.

- **20230723-yolov8-v1.9**
    1. 利用(ICLR2023)Reversible Column Networks对yolov5,yolov8的结构进行重设计.
    2. 支持旋转目标检测2023SOTA的LSKNet主干.
    3. 支持旋转目标检测2023SOTA的LSKNet主干中的LSKBlock注意力机制.
    4. 更新使用教程中的常见错误.
    5. 使用教程中增加常见疑问.

- **20230730-yolov8-v1.10**
    1. 增加yolov8-C2f-SCConv,yolov5-C3-SCConv.(CVPR 2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)
    2. 增加yolov8-C2f-ScConv,yolov5-C3-ScConv.(CVPR 2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)
    3. 更新使用教程.
    4. 更新视频百度云链接,增加SCConv和ScConv的使用教程.

- **20230730-yolov8-v1.11**
    1. yolov8-C2f-ScConv,yolov5-C3-ScConv分别更名为yolov8-C2f-SCcConv,yolov5-C3-SCcConv,因为在windows下命名不会区分大小写,导致解压的时候会出现覆盖请求.
    2. 支持MPDiou,具体修改方法请看使用教程.

- **20230802-yolov8-v1.11.1**
    1. 去除dataloader中的drop_last(ultralytics/yolo/data/build.py, build_dataloader func).
    2. 修正MPDiou.

- **20230806-yolov8-v1.12**
    1. 添加全新自研模块(Light Adaptive-weight downsampling),具体可看使用教程.

- **20230808-yolov8-v1.13**
    1. 添加全新自研模块(EMSC, EMSCP),具体可看使用教程.
    2. 添加RSC-YOLO中的RCSOSA到yolov5和yolov8中.
    3. 更新使用教程.

- **20230808-yolov8-v1.14**
    1. 支持SlideLoss和EMASlideLoss(利用Exponential Moving Average优化mean iou,可当自研创新模块),使用方式具体看使用教程.
    2. 支持KernelWarehouse:Towards Parameter-Efficient Dynamic Convolution(2023最新发布的动态卷积).
    3. 支持最新可变形卷积-Dynamic Snake Convolution.
    4. 支持Normalized Gaussian Wasserstein Distance(NWD).
    5. 增加CPCANet中的CPCA注意力机制.