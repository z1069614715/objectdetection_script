# 目前自带的一些改进方案(持续更新)

# 为了感谢各位对V8项目的支持,本项目的赠品是yolov5-PAGCP通道剪枝算法.[具体使用教程](https://www.bilibili.com/video/BV1yh4y1Z7vz/)

# 专栏改进汇总

### 二次创新系列
1. ultralytics/cfg/models/v8/yolov8-RevCol.yaml

    使用(ICLR2023)Reversible Column Networks对yolov8主干进行重设计,里面的支持更换不同的C2f-Block.
2. EMASlideLoss

    使用EMA思想与SlideLoss进行相结合.
3. ultralytics/cfg/models/v8/yolov8-dyhead-DCNV3.yaml

    使用[DCNV3](https://github.com/OpenGVLab/InternImage)替换DyHead中的DCNV2.
4. ultralytics/cfg/models/v8/yolov8-C2f-EMBC.yaml

    使用[Efficientnet](https://blog.csdn.net/weixin_43334693/article/details/131114618?spm=1001.2014.3001.5501)中的MBConv与EffectiveSE改进C2f.
5. ultralytics/cfg/models/v8/yolov8-GhostHGNetV2.yaml

    使用Ghost_HGNetV2作为YOLOV8的backbone.
6. ultralytics/cfg/models/v8/yolov8-RepHGNetV2.yaml

    使用Rep_HGNetV2作为YOLOV8的backbone.
7. ultralytics/cfg/models/v8/yolov8-C2f-DWR-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)的模块进行二次创新后改进C2f.
8. ultralytics/cfg/models/v8/yolov8-ASF-P2.yaml

    在ultralytics/cfg/models/v8/yolov8-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.
9. ultralytics/cfg/models/v8/yolov8-CSP-EDLAN.yaml

    使用[DualConv](https://github.com/ChipsGuardian/DualConv)打造CSP Efficient Dual Layer Aggregation Networks改进yolov8.
10. ultralytics/cfg/models/v8/yolov8-bifpn-SDI.yaml

    使用[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对BIFPN进行二次创新.
11. ultralytics/cfg/models/v8/yolov8-goldyolo-asf.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute与[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新改进yolov8的neck.
12. ultralytics/cfg/models/v8/yolov8-dyhead-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)对DyHead进行二次创新.(请关闭AMP进行训练,使用教程请看20240116版本更新说明)
13. ultralytics/cfg/models/v8/yolov8-HSPAN.yaml

    对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进yolov8的neck.
14. ultralytics/cfg/models/v8/yolov8-GDFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN与[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)进行二次创新改进Neck.
15. ultralytics/cfg/models/v8/yolov8-HSPAN-DySample.yaml

    对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN再进行创新,使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进其上采样模块.
16. ultralytics/cfg/models/v8/yolov8-ASF-DySample.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion与[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)组合得到Dynamic Sample Attentional Scale Sequence Fusion.

17. ultralytics/cfg/models/v8/yolov8-C2f-DCNV2-Dynamic.yaml

    利用自研注意力机制MPCA强化DCNV2中的offset和mask.

18. ultralytics/cfg/models/v8/yolov8-C2f-iRMB-Cascaded.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.

19. ultralytics/cfg/models/v8/yolov8-C2f-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.

20. ultralytics/cfg/models/v8/yolov8-C2f-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.

21. ultralytics/cfg/models/v8/yolov8-DBBNCSPELAN.yaml

    使用[Diverse Branch Block CVPR2021](https://arxiv.org/abs/2103.13425)对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolov8.

22. ultralytics/cfg/models/v8/yolov8-OREPANCSPELAN.yaml

    使用[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolov8.

23. ultralytics/cfg/models/v8/yolov8-DRBNCSPELAN.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolov8.

24. ultralytics/cfg/models/v8/yolov8-DynamicHGNetV2.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的DynamicConv对[CVPR2024 RTDETR](https://arxiv.org/abs/2304.08069)中的HGBlokc进行二次创新.

25. ultralytics/cfg/models/v8/yolov8-C2f-RVB-EMA.yaml

    使用[CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)中的RepViTBlock和EMA注意力机制改进C2f.

26. ultralytics/cfg/models/v8/yolov8-ELA-HSFPN.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN.

27. ultralytics/cfg/models/v8/yolov8-CA-HSFPN.yaml

    使用[Coordinate Attention CVPR2021](https://github.com/houqb/CoordAttention)改进HSFPN.

28. ultralytics/cfg/models/v8/yolov8-CAA-HSFPN.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块HSFPN.

29. ultralytics/cfg/models/v8/yolov8-CSMHSA.yaml

    对Mutil-Head Self-Attention进行创新得到Cross-Scale Mutil-Head Self-Attention.
    1. 由于高维通常包含更高级别的语义信息，而低维包含更多细节信息，因此高维信息作为query，而低维信息作为key和Value，将两者结合起来可以利用高维的特征帮助低维的特征进行精细过滤，可以实现更全面和丰富的特征表达。
    2. 通过使用高维的上采样信息进行Query操作，可以更好地捕捉到目标的全局信息，从而有助于增强模型对目标的识别和定位能力。

30. ultralytics/cfg/models/v8/yolov8-CAFMFusion.yaml

    利用具有[HCANet](https://github.com/summitgao/HCANet)中的CAFM，其具有获取全局和局部信息的注意力机制进行二次改进content-guided attention fusion.

31. ultralytics/cfg/models/v8/yolov8-C2f-Faster-CGLU.yaml

    使用[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU对CVPR2023中的FasterNet进行二次创新.

32. ultralytics/cfg/models/v8/yolov8-C2f-Star-CAA.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock和[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA改进C2f.

### 自研系列
1. ultralytics/cfg/models/v8/yolov8-LAWDS.yaml

    Light Adaptive-weight downsampling.自研模块,具体讲解请看百度云链接中的视频.

2. ultralytics/cfg/models/v8/yolov8-C2f-EMSC.yaml

    Efficient Multi-Scale Conv.自研模块,具体讲解请看百度云链接中的视频.

3. ultralytics/cfg/models/v8/yolov8-C2f-EMSCP.yaml

    Efficient Multi-Scale Conv Plus.自研模块,具体讲解请看百度云链接中的视频.

4. Lightweight Shared Convolutional Detection Head

    自研轻量化检测头.
    detect:ultralytics/cfg/models/v8/yolov8-LSCD.yaml
    seg:ultralytics/cfg/models/v8/yolov8-seg-LSCD.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LSCD.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LSCD.yaml
    1. GroupNorm在FOCS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.
    3. 在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    综合以上，我们可以让检测头做到参数量更少、计算量更少的情况下，尽可能减少精度的损失.

5. Task Align Dynamic Detection Head

    自研任务对齐动态检测头.
    detect:ultralytics/cfg/models/v8/yolov8-TADDH.yaml
    seg:ultralytics/cfg/models/v8/yolov8-seg-TADDH.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-TADDH.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-TADDH.yaml
    1. GroupNorm在FCOS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.并且在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    3. 参照TOOD的思想,除了标签分配策略上的任务对齐,我们也在检测头上进行定制任务对齐的结构,现有的目标检测器头部通常使用独立的分类和定位分支,这会导致两个任务之间缺乏交互,TADDH通过特征提取器从多个卷积层中学习任务交互特征,得到联合特征,定位分支使用DCNV2和交互特征生成DCNV2的offset和mask,分类分支使用交互特征进行动态特征选择.

6. ultralytics/cfg/models/v8/yolov8-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.

7. ultralytics/cfg/models/v8/yolov8-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.

8. ultralytics/cfg/models/v8/yolov8-RGCSPELAN.yaml

    自研RepGhostCSPELAN.
    1. 参考GhostNet中的思想(主流CNN计算的中间特征映射存在广泛的冗余)，采用廉价的操作生成一部分冗余特征图，以此来降低计算量和参数量。
    2. 舍弃yolov5与yolov8中常用的BottleNeck，为了弥补舍弃残差块所带来的性能损失，在梯度流通分支上使用RepConv，以此来增强特征提取和梯度流通的能力，并且RepConv可以在推理的时候进行融合，一举两得。
    3. 可以通过缩放因子控制RGCSPELAN的大小，使其可以兼顾小模型和大模型。

9. Lightweight Shared Convolutional Separamter BN Detection Head

    基于自研轻量化检测头上，参考NASFPN的设计思路把GN换成BN，并且BN层参数不共享.
    detect:ultralytics/cfg/models/v8/yolov8-LSCSBD.yaml
    seg:ultralytics/cfg/models/v8/yolov8-seg-LSCSBD.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LSCSBD.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LSCSBD.yaml
    1. 由于不同层级之间特征的统计量仍存在差异，Normalization layer依然是必须的，由于直接在共享参数的检测头中引入BN会导致其滑动平均值产生误差，而引入 GN 又会增加推理时的开销，因此我们参考NASFPN的做法，让检测头共享卷积层，而BN则分别独立计算。

### BackBone系列
1. ultralytics/cfg/models/v8/yolov8-efficientViT.yaml
    
    (CVPR2023)efficientViT替换yolov8主干.
2. ultralytics/cfg/models/v8/yolov8-fasternet.yaml

    (CVPR2023)fasternet替换yolov8主干.
3. ultralytics/cfg/models/v8/yolov8-timm.yaml

    使用timm支持的主干网络替换yolov8主干.

4. ultralytics/cfg/models/v8/yolov8-convnextv2.yaml

    使用convnextv2网络替换yolov8主干.
5. ultralytics/cfg/models/v8/yolov8-EfficientFormerV2.yaml

    使用EfficientFormerV2网络替换yolov8主干.(需要看[常见错误和解决方案的第五点](#a))  
6. ultralytics/cfg/models/v8/yolov8-vanillanet.yaml

    vanillanet替换yolov8主干.
7. ultralytics/cfg/models/v8/yolov8-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolov8主干.
8. ultralytics/cfg/models/v8/yolov8-swintransformer.yaml

    SwinTransformer-Tiny替换yolov8主干.
9. ultralytics/cfg/models/v8/yolov8-repvit.yaml

    [RepViT](https://github.com/THU-MIG/RepViT/tree/main)替换yolov8主干.
10. ultralytics/cfg/models/v8/yolov8-CSwinTransformer.yaml

    使用[CSWin-Transformer(CVPR2022)](https://github.com/microsoft/CSWin-Transformer/tree/main)替换yolov8主干.(需要看[常见错误和解决方案的第五点](#a))
11. ultralytics/cfg/models/v8/yolov8-HGNetV2.yaml

    使用HGNetV2作为YOLOV8的backbone.
12. ultralytics/cfg/models/v8/yolov8-unireplknet.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)替换yolov8主干.
13. ultralytics/cfg/models/v8/yolov8-TransNeXt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)改进yolov8的backbone.(需要看[常见错误和解决方案的第五点](#a))   
14. ultralytics/cfg/models/rt-detr/yolov8-rmt.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)改进rtdetr的主干.
15. ultralytics/cfg/models/v8/yolov8-pkinet.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)改进backbone.(需要安装mmcv和mmengine)
16. ultralytics/cfg/models/v8/yolov8-mobilenetv4.yaml

    使用[MobileNetV4](https://github.com/jaiwei98/MobileNetV4-pytorch/tree/main)改进yolov8-backbone.
17. ultralytics/cfg/models/v8/yolov8-starnet.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)改进yolov8-backbone.

### SPPF系列
1. ultralytics/cfg/models/v8/yolov8-FocalModulation.yaml

    使用[Focal Modulation](https://github.com/microsoft/FocalNet)替换SPPF.
2. ultralytics/cfg/models/v8/yolov8-SPPF-LSKA.yaml

    使用[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)注意力机制改进SPPF,增强多尺度特征提取能力.
3. ultralytics/cfg/models/v8/yolov8-AIFI.yaml

    使用[RT-DETR](https://arxiv.org/pdf/2304.08069.pdf)中的Attention-based Intrascale Feature Interaction(AIFI)改进yolov8.

### Neck系列
1. ultralytics/cfg/models/v8/yolov8-bifpn.yaml

    添加BIFPN到yolov8中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        支持大部分C2f-XXX结构.
    3. head_channel  
        BIFPN中的通道数,默认设置为256.
2. ultralytics/cfg/models/v8/yolov8-slimneck.yaml

    使用VoVGSCSP\VoVGSCSPC和GSConv替换yolov8 neck中的C2f和Conv.
3. Asymptotic Feature Pyramid Network[reference](https://github.com/gyyang23/AFPN/tree/master)

    a. ultralytics/cfg/models/v8/yolov8-AFPN-P345.yaml  
    b. ultralytics/cfg/models/v8/yolov8-AFPN-P345-Custom.yaml  
    c. ultralytics/cfg/models/v8/yolov8-AFPN-P2345.yaml  
    d. ultralytics/cfg/models/v8/yolov8-AFPN-P2345-Custom.yaml  
    其中Custom中的block支持大部分C2f-XXX结构.
4. ultralytics/cfg/models/v8/yolov8-RCSOSA.yaml

    使用[RCS-YOLO](https://github.com/mkang315/RCS-YOLO/tree/main)中的RCSOSA替换C2f.
5. ultralytics/cfg/models/v8/yolov8-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块
6. ultralytics/cfg/models/v8/yolov8-GFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN改进Neck.
7. ultralytics/cfg/models/v8/yolov8-EfficientRepBiPAN.yaml

    使用[YOLOV6](https://github.com/meituan/YOLOv6/tree/main)中的EfficientRepBiPAN改进Neck.
8. ultralytics/cfg/models/v8/yolov8-ASF.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolov8.
9. ultralytics/cfg/models/v8/yolov8-SDI.yaml

    使用[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对yolov8中的feature fusion部分进行重设计.
10. ultralytics/cfg/models/v8/yolov8-HSFPN.yaml

    使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进yolov8的neck.
11. ultralytics/cfg/models/v8/yolov8-CSFCN.yaml

    使用[Context and Spatial Feature Calibration for Real-Time Semantic Segmentation](https://github.com/kaigelee/CSFCN/tree/main)中的Context and Spatial Feature Calibration模块改进yolov8.
12. ultralytics/cfg/models/v8/yolov8-CGAFusion.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的content-guided attention fusion改进yolov8-neck.
13. ultralytics/cfg/models/v8/yolov8-SDFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的superficial detail fusion module改进yolov8-neck.

14. ultralytics/cfg/models/v8/yolov8-PSFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的profound semantic fusion module改进yolov8-neck.

### Head系列
1. ultralytics/cfg/models/v8/yolov8-dyhead.yaml

    添加基于注意力机制的目标检测头到yolov8中.
2. ultralytics/cfg/models/v8/yolov8-EfficientHead.yaml

    对检测头进行重设计,支持10种轻量化检测头.详细请看ultralytics/nn/extra_modules/head.py中的Detect_Efficient class.
3. ultralytics/cfg/models/v8/yolov8-aux.yaml

    参考YOLOV7-Aux对YOLOV8添加额外辅助训练头,在训练阶段参与训练,在最终推理阶段去掉.  
    其中辅助训练头的损失权重系数可在ultralytics/utils/loss.py中的class v8DetectionLoss中的__init__函数中的self.aux_loss_ratio设定,默认值参考yolov7为0.25.
4. ultralytics/cfg/models/v8/yolov8-seg-EfficientHead.yaml(实例分割)

    对检测头进行重设计,支持10种轻量化检测头.详细请看ultralytics/nn/extra_modules/head.py中的Detect_Efficient class. 
5. ultralytics/cfg/models/v8/yolov8-SEAMHead.yaml

    使用[YOLO-Face V2](https://arxiv.org/pdf/2208.02019v2.pdf)中的遮挡感知注意力改进Head,使其有效地处理遮挡场景.
6. ultralytics/cfg/models/v8/yolov8-MultiSEAMHead.yaml

    使用[YOLO-Face V2](https://arxiv.org/pdf/2208.02019v2.pdf)中的遮挡感知注意力改进Head,使其有效地处理遮挡场景.
7. ultralytics/cfg/models/v8/yolov8-PGI.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)的programmable gradient information改进YOLOV8.(PGI模块可在训练结束后去掉)
8. Lightweight Asymmetric Detection Head

    detect:ultralytics/cfg/models/v8/yolov8-LADH.yaml
    segment:ultralytics/cfg/models/v8/yolov8-seg-LADH.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LADH.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LADH.yaml
    使用[Faster and Lightweight: An Improved YOLOv5 Object Detector for Remote Sensing Images](https://www.mdpi.com/2072-4292/15/20/4974)中的Lightweight Asymmetric Detection Head改进yolov8-head.

### Label Assign系列
1. Adaptive Training Sample Selection匹配策略.

    在ultralytics/utils/loss.py中的class v8DetectionLoss中自行选择对应的self.assigner即可.

### PostProcess系列
1. soft-nms(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,ShapeIoU)

    soft-nms替换nms.(建议:仅在val.py时候使用,具体替换请看20240122版本更新说明)

### 上下采样算子
1. ultralytics/cfg/models/v8/yolov8-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample进行下采样.
2. ultralytics/cfg/models/v8/yolov8-SPDConv.yaml

    使用[SPDConv](https://github.com/LabSAINT/SPD-Conv/tree/main)进行下采样.
3. ultralytics/cfg/models/v8/yolov8-dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolov8-neck中的上采样.

4. ultralytics/cfg/models/v8/yolov8-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolov8-neck中的上采样.

5. ultralytics/cfg/models/v8/yolov8-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolov8的下采样.(请关闭AMP情况下使用)

6. ultralytics/cfg/models/v8/yolov8-v7DS.yaml

    使用[YOLOV7 CVPR2023](https://arxiv.org/abs/2207.02696)的下采样结构改进YOLOV8中的下采样.

7. ultralytics/cfg/models/v8/yolov8-ADown.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)的下采样结构改进YOLOV8中的下采样.

8. ultralytics/cfg/models/v8/yolov8-SRFD.yaml

    使用[A Robust Feature Downsampling Module for Remote Sensing Visual Tasks](https://ieeexplore.ieee.org/document/10142024)改进yolov8的下采样.

### YOLOV8-C2f系列
1. ultralytics/cfg/models/v8/yolov8-C2f-Faster.yaml

    使用C2f-Faster替换C2f.(使用FasterNet中的FasterBlock替换C2f中的Bottleneck)
2. ultralytics/cfg/models/v8/yolov8-C2f-ODConv.yaml

    使用C2f-ODConv替换C2f.(使用ODConv替换C2f中的Bottleneck中的Conv)
3. ultralytics/cfg/models/v8/yolov8-C2f-ODConv.yaml

    使用C2f-ODConv替换C2f.(使用ODConv替换C2f中的Bottleneck中的Conv)
4. ultralytics/cfg/models/v8/yolov8-C2f-Faster-EMA.yaml

    使用C2f-Faster-EMA替换C2f.(C2f-Faster-EMA推荐可以放在主干上,Neck和head部分可以选择C2f-Faster)
5. ultralytics/cfg/models/v8/yolov8-C2f-DBB.yaml

    使用C2f-DBB替换C2f.(使用DiverseBranchBlock替换C2f中的Bottleneck中的Conv)
6. ultralytics/cfg/models/v8/yolov8-C2f-CloAtt.yaml

    使用C2f-CloAtt替换C2f.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C2f中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))
7. ultralytics/cfg/models/v8/yolov8-C2f-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C2f融合.
8. ultralytics/cfg/models/v8/yolov8-C2f-SCcConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C2f融合.  
    (取名为SCcConv的原因是在windows下命名是不区分大小写的)
9. ultralytics/cfg/models/v8/yolov8-KernelWarehouse.yaml
    
    使用[Towards Parameter-Efficient Dynamic Convolution](https://github.com/OSVAI/KernelWarehouse)添加到yolov8中.  
    使用此模块需要注意,在epoch0-20的时候精度会非常低,过了20epoch会正常.
10. ultralytics/cfg/models/v8/yolov8-C2f-DySnakeConv.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C2f融合.
11. ultralytics/cfg/models/v8/yolov8-C2f-DCNV2.yaml

    使用C2f-DCNV2替换C2f.(DCNV2为可变形卷积V2)
12. ultralytics/cfg/models/v8/yolov8-C2f-DCNV3.yaml

    使用C2f-DCNV3替换C2f.([DCNV3](https://github.com/OpenGVLab/InternImage)为可变形卷积V3(CVPR2023,众多排行榜的SOTA))  
    官方中包含了一些指定版本的DCNV3 whl包,下载后直接pip install xxx即可.具体和安装DCNV3可看百度云链接中的视频.
13. ultralytics/cfg/models/v8/yolov8-C2f-OREPA.yaml

    使用C2f-OREPA替换C2f.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)
14. ultralytics/cfg/models/v8/yolov8-C2f-REPVGGOREPA.yaml

    使用C2f-REPVGGOREPA替换C2f.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)
15. ultralytics/cfg/models/v8/yolov8-C2f-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进C2f.(请关闭AMP进行训练,使用教程请看20240116版本更新说明)
16. ultralytics/cfg/models/v8/yolov8-C2f-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided改进C2f.
17. ultralytics/cfg/models/v8/yolov8-C2f-MSBlock.yaml

    使用[YOLO-MS](https://github.com/FishAndWasabi/YOLO-MS/tree/main)中的MSBlock改进C2f.
18. ultralytics/cfg/models/v8/yolov8-C2f-DLKA.yaml

    使用[deformableLKA](https://github.com/xmindflow/deformableLKA)改进C2f.
19. ultralytics/cfg/models/v8/yolov8-C2f-DAttention.yaml

    使用[Vision Transformer with Deformable Attention(CVPR2022)](https://github.com/LeapLabTHU/DAT)改进C2f.(需要看[常见错误和解决方案的第五点](#a))  
    使用注意点请看百度云视频.(DAttention(Vision Transformer with Deformable Attention CVPR2022)使用注意说明.)
20. 使用[ParC-Net](https://github.com/hkzhang-git/ParC-Net/tree/main)中的ParC_Operator改进C2f.(需要看[常见错误和解决方案的第五点](#a))  
    使用注意点请看百度云视频.(20231031更新说明)    
21. ultralytics/cfg/models/v8/yolov8-C2f-DWR.yaml

    使用[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块,加强从网络高层的可扩展感受野中提取特征.
22. ultralytics/cfg/models/v8/yolov8-C2f-RFAConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFAConv改进yolov8.

23. ultralytics/cfg/models/v8/yolov8-C2f-RFCBAMConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFCBAMConv改进yolov8.

24. ultralytics/cfg/models/v8/yolov8-C2f-RFCAConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFCAConv改进yolov8.
25. ultralytics/cfg/models/v8/yolov8-C2f-FocusedLinearAttention.yaml

    使用[FLatten Transformer(ICCV2023)](https://github.com/LeapLabTHU/FLatten-Transformer)中的FocusedLinearAttention改进C2f.(需要看[常见错误和解决方案的第五点](#a))    
    使用注意点请看百度云视频.(20231114版本更新说明.)
26. ultralytics/cfg/models/v8/yolov8-C2f-MLCA.yaml

    使用[Mixed Local Channel Attention 2023](https://github.com/wandahangFY/MLCA/tree/master)改进C2f.(用法请看百度云视频-20231129版本更新说明)

27. ultralytics/cfg/models/v8/yolov8-C2f-AKConv.yaml

    使用[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进C2f.(用法请看百度云视频-20231129版本更新说明)
28. ultralytics/cfg/models/v8/yolov8-C2f-UniRepLKNetBlock.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的UniRepLKNetBlock改进C2f.
29. ultralytics/cfg/models/v8/yolov8-C2f-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进C2f.
30. ultralytics/cfg/models/v8/yolov8-C2f-AggregatedAtt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进C2f.(需要看[常见错误和解决方案的第五点](#a))   

31. ultralytics/cfg/models/v8/yolov8-C2f-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进yolov8中的C2f.

32. ultralytics/cfg/models/v8/yolov8-C2f-iRMB.yaml

    使用[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB改进C2f.

33. ultralytics/cfg/models/v8/yolov8-C2f-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)对C2f中的BottleNeck进行改进,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

34. ultralytics/cfg/models/v8/yolov8-C2f-LVMB.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)与Cross Stage Partial进行结合,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

35. ultralytics/cfg/models/v8/yolov8-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行改进yolov8.

36. ultralytics/cfg/models/v8/yolov8-C2f-DynamicConv.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的DynamicConv改进C2f.

37. ultralytics/cfg/models/v8/yolov8-C2f-GhostDynamicConv.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的GhostModule改进C2f.

38. ultralytics/cfg/models/v8/yolov8-C2f-RVB.yaml

    使用[CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)中的RepViTBlock改进C2f.

39. ultralytics/cfg/models/v8/yolov8-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进yolov8.

40. ultralytics/cfg/models/v8/yolov8-C2f-RetBlock.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进C2f.

41. ultralytics/cfg/models/v8/yolov8-C2f-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块改进C2f.

42. ultralytics/cfg/models/v8/yolov8-RepNCSPELAN_CAA.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块改进RepNCSPELAN.

43. ultralytics/cfg/models/v8/yolov8-C2f-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进C2f.

44. ultralytics/cfg/models/v8/yolov8-C2f-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C2f.

45. ultralytics/cfg/models/v8/yolov8-C2f-Star.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock改进C2f.

### 组合系列
1. ultralytics/cfg/models/v8/yolov8-fasternet-bifpn.yaml

    fasternet与bifpn的结合.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

2. ultralytics/cfg/models/v8/yolov8-ELA-HSFPN-TADDH.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN,使用自研动态动态对齐检测头改进Head.

3. ultralytics/cfg/models/v8/yolov8-FDPN-TADDH.yaml

    自研结构的融合.
    1. 自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    2. 自研任务对齐动态检测头(Task Align Dynamic Detection Head)

4. ultralytics/cfg/models/v8/yolov8-starnet-C2f-Star-LSCD.yaml

    轻量化模型组合.
    1. CVPR2024-StarNet Backbone.
    2. C2f-Star.
    3. Lightweight Shared Convolutional Detection Head.

### 注意力系列
1. EMA
2. SimAM
3. SpatialGroupEnhance
4. BiLevelRoutingAttention, BiLevelRoutingAttention_nchw
5. TripletAttention
6. CoordAtt
7. CBAM
8. BAMBlock
9. EfficientAttention(CloFormer中的注意力)
10. LSKBlock
11. SEAttention
12. CPCA
13. deformable_LKA
14. EffectiveSEModule
15. LSKA
16. SegNext_Attention
17. DAttention(Vision Transformer with Deformable Attention CVPR2022)
18. FocusedLinearAttention(ICCV2023)
19. MLCA
20. TransNeXt_AggregatedAttention
21. LocalWindowAttention(EfficientViT中的CascadedGroupAttention注意力)
22. Efficient Local Attention
23. CAA(CVPR2024 PKINet中的注意力)
24. CAFM

### Loss系列
1. SlideLoss,EMASlideLoss.(可动态调节正负样本的系数,让模型更加注重难分类,错误分类的样本上)
2. IoU,GIoU,DIoU,CIoU,EIoU,SIoU,MPDIoU,ShapeIoU.
3. Inner-IoU,Inner-GIoU,Inner-DIoU,Inner-CIoU,Inner-EIoU,Inner-SIoU,Inner-ShapeIoU.
4. Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
5. Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
6. FocalLoss,VarifocalLoss,QualityfocalLoss
7. Focaler-IoU系列(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,WIoU,MPDIoU,ShapeIoU)
8. Powerful-IoU,Powerful-IoUV2,Inner-Powerful-IoU,Inner-Powerful-IoUV2,Focaler-Powerful-IoU,Focaler-Powerful-IoUV2,Wise-Powerful-IoU(v1,v2,v3),Wise-Powerful-IoUV2(v1,v2,v3)[论文链接](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640)
9. Normalized Gaussian Wasserstein Distance.

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

- **20230824-yolov8-v1.14**
    1. 支持SlideLoss和EMASlideLoss(利用Exponential Moving Average优化mean iou,可当自研创新模块),使用方式具体看使用教程.
    2. 支持KernelWarehouse:Towards Parameter-Efficient Dynamic Convolution(2023最新发布的动态卷积).
    3. 支持最新可变形卷积-Dynamic Snake Convolution.
    4. 支持Normalized Gaussian Wasserstein Distance(NWD).
    5. 增加CPCANet中的CPCA注意力机制.
    6. 更新使用教程.

- **20230830-yolov8-v1.15**
    1. 对检测头进行重设计,支持10种(参数量和计算量更低的)检测头,详细请看使用教程.

- **20230904-yolov8-v1.16**
    1. 支持DCNV2,DCNV3.详细请看项目百度云视频.
    2. 使用DCNV3改进DyHead.(ultralytics/models/v5/yolov5-dyhead-DCNV3.yaml,ultralytics/models/v8/yolov8-dyhead-DCNV3.yaml)
    3. 根据YOLOV7-AUX辅助训练头思想,改进YOLOV8,增加辅助训练头,训练时候参与训练,检测时候去掉.(ultralytics/models/v5/yolov5-AuxHead.yaml, ultralytics/models/v8/yolov8-AuxHead.yaml)
    4. 增加C3-Faster(ultralytics/models/v5/yolov5-C3-Faster.yaml).
    5. 增加C3-ODConv(ultralytics/models/v5/yolov5-C3-ODConv.yaml).
    6. 增加C3-Faster-EMA(ultralytics/models/v5/yolov5-C3-Faster-EMA.yaml).
    7. 更新使用教程.

- **20230909-yolov8-v1.17**
    1. 优化辅助训练头部分代码.
    2. 修复多卡训练中的一些bug.
    3. 更新使用教程.(百度云视频中增加关于C3-XXX和C2f-XXX移植到官方yolov5上的讲解)
    4. 支持TAL标签分配策略中使用NWD(具体可看使用教程).

- **20230915-yolov8-v1.18**
    1. 新增Online Convolutional Re-parameterization (CVPR2022).(超越DBB和RepVGG) (C3-OREPA,C3-REPVGGOREPA,C2f-OREPA,C2f-REPVGGOREPA)
    2. 新增FocalModulation.
    3. 支持RepViT和SwinTransformer-Tiny主干.
    4. 利用OREPA优化自研模块(EMSC,EMSCP).
    5. 更新使用教程和百度云视频.

- **20230916-yolov8-v1.19**
    1. 去除OREPA_1x1,该结构会让模型无法收敛或者NAN.
    2. 新增yolov8-fasternet-bifpn和yolov5-fasternet-bifpn.
    3. 更新使用教程和百度云视频.(更新OREPA的视频和增加如何看懂代码结构-以C2f-Faster-EMA为例).

- **20230919-yolov8-v1.19.1**
    1. 修复C2f-ODConv在20epochs后精度异常问题.
    2. 修复BAM注意力机制中的padding问题.
    3. 修复EfficientAttention(CloFormer中的注意力)注意力机制不能在配置文件添加的问题.
    4. 去除C2f-EMSP-OREPA,C2f-EMSCP-OREPA,C3-EMSP-OREPA,C3-EMSCP-OREPA,这部分不稳定,容易出现NAN.
    5. 群公告中增加使用前必看的百度云视频链接.

- **20230924-yolov8-v1.20**
    1. 增加自研注意力机制MPCA(基于CVPR2021 CA注意力机制).详细可看百度云视频.
    2. 使用自研注意力机制MPCA强化DCNV2中的offset和mask生成.详细可看百度云视频和使用教程.
    3. 把timm配置文件的预训练权重参数改为False,也即是默认不下载和使用预训练权重.
    4. 利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.

- **20230927-yolov8-v1.21**
    1. 使用YOLO-MS中的MSBlock改进C2f和C3模块,具体请看使用教程.
    2. 使用GCNet中的Light-weight Context Guided改进C2f和C3模块,具体请看使用教程.
    3. 使用GCNet中的Light-weight Context Guided Down替换YOLO中的下采样模块,具体请看使用教程.

- **20231010-yolov8-v1.22**
    1. RepViT同步官方源码.
    2. 经实验发现网络全使用C2f-MSBlock和C3-MSBlock不稳定,因此在Neck部分还是使用C2f或C3,具体可参看对应的配置文件.
    3. 支持deformableLKA注意力机制,并进行改进C2f和C3,提出C2f_DLKA,C3_DLKA.
    4. 使用DAMO-YOLO中的RepGFPN改进yolov8中的Neck.
    5. 使用YOLOV6中的EfficientRepBiPAN改进yolov8中的Neck.
    6. 新增支持SPDConv进行下采样.
    7. 使用Efficientnet中的MBConv与EffectiveSE改进C2f和C3.

- **20231020-yolov8-v1.23**
    1. 更新使用教程和百度云视频.(更新DAttention使用说明视频).
    2. 增加LSKA, SegNext_Attention, DAttention(Vision Transformer with Deformable Attention CVPR2022).
    3. 使用LSKA改进SPPF,增强多尺度特征提取能力.
    4. 使用[Vision Transformer with Deformable Attention(CVPR2022)]改进C2f,C3.

- **20231107-yolov8-v1.24**
    1. 新增CVPR2022-CSwinTransformer主干.
    2. 新增yolov5-AIFI.yaml,yolov8-AIFI.yaml.
    3. 新增使用ParC-Net中的位置感知循环卷积改进C3,C2f.
    4. 新增使用DWRSeg中的Dilation-wise Residual(DWR)模块,加强从网络高层的可扩展感受野中提取特征.(yolov5-C3-DWR.yaml,yolov8-C2f-DWR.yaml)
    5. 把当前所有的改进同步到ultralytics-8.0.202版本上.
    6. 更新新版百度云链接视频.
    7. 新增热力图、FPS脚本.

- **20231114-yolov8-v1.25**
    1. 新增EIou,SIou.
    2. 新增Inner-IoU,Inner-GIoU,Inner-DIoU,Inner-CIoU,Inner-EIoU,Inner-SIoU.
    3. 使用今年最新的MPDIoU与Inner-IoU相结合得到Inner-MPDIoU.
    4. 新增[FLatten Transformer(ICCV2023)](https://github.com/LeapLabTHU/FLatten-Transformer)中的FocusedLinearAttention改进C3,C2f.
    5. 更新get_FPS脚本中的模型导入方式,避免一些device报错.
    6. 更新百度云链接视频-20231114版本更新说明.

- **20231114-yolov8-v1.26**
    1. 修正MPDIOU中的mpdiou_hw参数.
    2. 更新使用教程.

- **20231129-yolov8-v1.27**
    1. 新增Mixed Local Channel Attention改进C2f和C3.
    2. 新增AKConv改进C2f和C3.
    3. 更新使用教程.
    4. 更新百度云链接视频-20231129版本更新说明.

- **20231207-yolov8-v1.28**
    1. 新增支持2023最新大卷积核CNN架构RepLKNet升级版-UniRepLKNet.
    2. 新增UniRepLKNet中的[UniRepLKNetBlock, DilatedReparamBlock]改进C3和C2f.
    3. 使用UniRepLKNet中的DilatedReparamBlock对DWRSeg中的Dilation-wise Residual(DWR)模块进行二次创新后改进C3和C2f.
    4. 修复get_FPS.py测速前没有进行fuse的问题.
    5. 更新使用教程.
    6. 更新百度云链接视频-20231207版本更新说明.

- **20231217-yolov8-v1.29**
    1. 新增ASF-YOLO中的Attentional Scale Sequence Fusion,并在其基础上增加P2检测层并进行优化网络结构.
    2. 新增使用DualConv打造CSP Efficient Dual Layer Aggregation Networks.
    3. 更新使用教程.
    4. 更新百度云链接视频-20231217版本更新说明.

- **20231227-yolov8-v1.30**
    1. 新增支持TransNeXt主干和TransNeXt中的聚焦感知注意力机制.
    2. 新增U-NetV2中的Semantics and Detail Infusion Module,分别对BIFPN和PAFPN中的feature fusion部分进行二次创新.
    3. 更新使用教程.
    4. 更新百度云链接视频-20231227版本更新说明.

- **20240104-yolov8-v1.31**
    1. 新增Shape-IoU,Inner-Shape-IoU.
    2. 更新使用教程.
    3. 更新百度云链接视频-20230104版本更新说明.

- **20240111-yolov8-v1.32**
    1. 支持FocalLoss,VarifocalLoss,QualityfocalLoss.
    2. 支持Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
    3. 支持Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
    4. 更新使用教程.
    5. 更新百度云链接视频-20230111版本更新说明.

- **20240116-yolov8-v1.33**
    1. 使用ASF-YOLO中Attentional Scale Sequence Fusion与GOLD-YOLO中的Gatherand-Distribute进行二次创新结合.
    2. 支持最新的DCNV4,C2f-DCNV4,C3-DCNV4,并使用DCNV4对DyHead进行二次创新(DyHead_DCNV4).
    3. 修复不使用wise的情况下断点续训的bug.
    4. 更新使用教程.
    5. 更新百度云链接视频-20230116版本更新说明.

- **20240122-yolov8-v1.34**
    1. 使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进YOLOV5、YOLOV8中的Neck.
    2. 对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进YOLOV5、YOLOV8中的Neck.
    3. 增加CARAFE轻量化上采样算子.
    4. 增加DySample(ICCV2023)动态上采样算子.
    5. 增加Haar wavelet downsampling下采样算子.
    6. 支持soft-nms.(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,ShapeIoU)
    7. 更新使用教程.
    8. 更新百度云链接视频-20230122版本更新说明.

- **20240203-yolov8-v1.35**
    1. 增加Focaler-IoU(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,WIoU,MPDIoU,ShapeIoU).
    2. 增加RepGFPN与DySample的二次创新组合.
    3. 增加ASF-YOLO中的ASSF与DySample的二次创新组合.
    4. 增加HS-PAN与DySample的二次创新组合.
    5. 使用遮挡感知注意力SEAM,MultiSEAM改进Head,得到具有遮挡感知识别的SEAMHead,MultiSEAMHead.
    6. 优化plot_result.py,使用线性插值来填充inf或者nan的数据,降低出现乱码问题的概率.
    7. 更新使用教程.
    8. 更新百度云链接视频-20230203版本更新说明.

- **20240208-yolov8-v1.36**
    1. 将所有改进代码同步到8.1.9上.

- **20240216-yolov8-v1.37**
    1. 增加EMO模型中的iRMB模块,并使用(EfficientViT-CVPR2023)中的CascadedAttention对其二次创新得到iRMB_Cascaded.
    2. 新增Shift-ConvNets相关改进内容.(rtdetr-SWC.yaml,rtdetr-R50-SWC.yaml,yolov8-detr-C2f-SWC.yaml,yolov5-detr-C3-SWC.yaml)
    3. 使用UniRepLKNet中的DilatedReparamBlock对EMO中的iRMB进行二次创新.
    4. 使用Shift-ConvNets中的具有移位操作的卷积对EMO中的iRMB进行二次创新.
    5. 修复一些已知问题.
    6. 更新使用教程.
    8. 百度云视频增加20240216更新说明.

- **20240219-yolov8-v1.38**
    1. 使用最新的Mamba架构(号称超越Transformer的新架构)改进C2f(提供两种改进方式).
    2. 新增Powerful-IoU,Powerful-IoUV2,Inner-Powerful-IoU,Inner-Powerful-IoUV2,Focaler-Powerful-IoU,Focaler-Powerful-IoUV2,Wise-Powerful-IoU(v1,v2,v3),Wise-Powerful-IoUV2(v1,v2,v3)系列.
    3. 修复一些已知问题.
    4. 更新使用教程.
    5. 百度云视频增加20240219更新说明.

- **20240222-yolov8-v1.39**
    1. 新增YOLOV9中的RepNCSPELAN模块.
    2. 使用DBB,OREPA,DilatedReparamBlock对YOLOV9中的RepNCSPELAN模块进行二次创新.
    3. 更新使用教程.
    4. 百度云视频增加20240222更新说明.

- **20240229-yolov8-v1.40**
    1. 新增YOLOV9中的ADown下采样模块.
    2. 新增YOLOV7中的下采样模块.
    3. 新增YOLOV9中的programmable gradient information,并且PGI模块可以在训练后去除.
    4. 更新使用教程.
    5. 百度云视频增加20240229更新说明.

- **20240303-yolov8-v1.41**
    1. 新增CVPR2024-parameternet中的GhostModule与DynamicConv.
    2. 使用CVPR2024-parameternet中的DynamicConv对CVPR2024-RTDETR中的HGBlokc进行二次创新.
    3. 更新使用教程.
    4. 百度云视频增加20240303更新说明.

- **20240309-yolov8-v1.42**
    1. 新增拆分CVPR2024 RepVIT里面的block,提出C2f-RVB、C2f-RVB-EMA.
    2. 新增Lightweight Object Detection论文中的Dynamic Group Convolution Shuffle Transformer.
    3. 新增自研Lightweight Shared Convolutional Detection Head,支持Detect、Seg、Pose、Obb.
    4. 更新使用教程.
    5. 百度云视频增加20240309更新说明.

- **20240314-yolov8-v1.43**
    1. 新增自研Task Align Dynamic Detection Head,支持Detect、Seg、Pose、Obb.
    2. 更新使用教程，新增几个常见疑问回答.
    3. 修复shapeiou调用不生效的问题.
    4. 百度云视频增加20240314更新说明.

- **20240323-yolov8-v1.44**
    1. 新增CVPR2024-RMT主干,并支持RetBlock改进C3、C2f.
    2. 新增2024年新出的Efficient Local Attention,并用其对HSFPN进行二次创新，并加入自研检测头TADDH.
    3. 使用CVPR2021-CoordAttention对HSFPN进行二次创新.
    4. 更新使用教程,增加多个常见疑问解答.
    5. 百度云视频增加20240323更新说明.

- **20240330-yolov8-v1.45**
    1. 新增CVPR2024 PKINet主干.
    2. 新增CVPR2024 PKINet中的PKIModule和CAA模块,提出C2f-PKI.
    3. 使用CVPR2024 PKINet中的Context Anchor Attention改进RepNCSPELAN、HSFPN.
    4. 更新使用教程.
    5. 百度云视频增加20240330更新说明.

- **20240406-yolov8-v1.46**
    1. 新增CVPR2024 Frequency-Adaptive Dilated Convolution.
    2. 新增自研Focusing Diffusion Pyramid Network.
    3. 更新使用教程.
    4. 百度云视频增加20240406更新说明.

- **20240408-yolov8-v1.47**
    1. 修复自研Focusing Diffusion Pyramid Network的一个小bug.
    2. 新增使用自研特征聚焦扩散金字塔网络和自研任务对齐动态检测头相结合的配置文件yolov8-FDPN-TADDH.yaml
    3. 新增HCFNet针对小目标分割的Parallelized Patch-Aware Attention Module改进C2f.
    4. 新增HCFNet针对小目标分割的Dimension-Aware Selective Integration Module对自研Focusing Diffusion Pyramid Network再次进行创新.
    5. 更新使用教程.
    6. 百度云视频增加20240408更新说明.

- **20240414-yolov8-v1.48**
    1. 新增Cross-Scale Mutil-Head Self-Attention,对Mutil-Head Self-Attention进行二次创新.
    2. 更新使用教程.
    3. 百度云视频增加20240414更新说明.

- **20240420-yolov8-v1.49**
    1. 新增A Robust Feature Downsampling Module for Remote Sensing Visual Tasks中的下采样.
    2. 新增Context and Spatial Feature Calibration for Real-Time Semantic Segmentation中的Context and Spatial Feature Calibration.
    3. 更新使用教程.
    4. 百度云视频增加20240420更新说明.

- **20240428-yolov8-v1.50**
    1. 修复20240420更新中的Context and Spatial Feature Calibration序号错误问题.
    2. 新增支持mobilenetv4-backbone.
    3. 新增支持content-guided attention fusion改进yolov8-neck.
    4. 新增支持使用CAFM对CGAFusion进行二次改进,得到CAFMFusion改进yolov8-neck.
    5. 更新使用教程.
    6. 百度云视频增加20240428更新说明.

- **20240501-yolov8-v1.51**
    1. get_FPS.py脚本新增可以通过yaml测试推理速度.
    2. 新增自研RGCSPELAN,其比C3、ELAN、C2f、RepNCSPELAN更低参数量和计算量更快推理速度.
    3. 更新使用教程.
    4. 百度云视频增加20240501更新说明.

- **20240505-yolov8-v1.52**
    1. 新增LADH.(Lightweight Asymmetric Detection Head).
    2. 使用CVPR2024-TransNext中的Convolutional GLU对CVPR2023-FasterBlock进行二次创新.
    3. 更新使用教程.
    4. 百度云视频增加20240505更新说明.

- **20240512-yolov8-v1.53**
    1. 基于LSCD自研轻量化检测头再次进行改进得到LSCSBD.
    2. 新增PSFusion中的superficial detail fusion module、profound semantic fusion module改进yolov8-neck.
    3. 更新使用教程.
    4. 百度云视频增加20240512更新说明.

- **20240513-yolov8-v1.54**
    1. 支持CVPR2024-StarNet,新一代SOTA轻量化模型.
    2. 使用CVPR2024-StarNet对C2f进行创新得到C2f-Star.
    3. 使用CVPR2024-StarNet与CVPR2024-PKINet进行组合创新得到C2f-Star-CAA.
    4. 增加轻量化模型组合配置文件,融合StarNet、C2f-Star、LSCD.
    5. 更新使用教程.
    6. 百度云视频增加20240513更新说明.