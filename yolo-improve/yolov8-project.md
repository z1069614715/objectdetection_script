# 目前自带的一些改进方案(持续更新)

# 为了感谢各位对V8项目的支持,本项目的赠品是yolov5-PAGCP通道剪枝算法.[具体使用教程](https://www.bilibili.com/video/BV1yh4y1Z7vz/)

<a id="b"></a>

#### 目前支持的一些block (yolov5默认C3,yolov8默认C2f) (部分block可能会与主结构有冲突,具体以是否能运行为主)

##### C2f系列
C2f, C2f_Faster, C2f_ODConv, C2f_Faster_EMA, C2f_DBB, C2f_CloAtt, C2f_SCConv, C2f_ScConv, C2f_EMSC, C2f_EMSCP, C2f_KW, C2f_DCNv2, C2f_DCNv3, C2f_OREPA, C2f_REPVGGOREPA, C2f_DCNv2_Dynamic, C2f_MSBlock, C2f_ContextGuided, C2f_DLKA, C2f_EMBC
##### C3系列  
C3, C3Ghost, C3_CloAtt, C3_SCConv, C3_ScConv, C3_EMSC, C3_EMSCP, C3_KW, C3_ODConv, C3_Faster, C3_Faster_EMA, C3_DCNv2, C3_DCNv3, C3_DBB, C3_OREPA, C3_REPVGGOREPA, C3_DCNv2_Dynamic, C3_MSBlock, C3_ContextGuided, C3_DLKA, C3_EMBC
##### 其他系列
VoVGSCSP, VoVGSCSPC, RCSOSA  

<a id="c"></a>

#### 目前整合的一些注意力机制 还需要别的注意力机制可从[github](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)拉取对应的代码到ultralytics/nn/extra_modules/attention.py即可. 视频教程可看项目视频中的(如何在yaml配置文件中添加注意力层)
EMA, SimAM, SpatialGroupEnhance, BiLevelRoutingAttention, BiLevelRoutingAttention_nchw, TripletAttention, CoordAtt, CBAM, BAMBlock, EfficientAttention(CloFormer中的注意力), LSKBlock, SEAttention, CPCA, deformable_LKA, EffectiveSEModule, LSKA, SegNext_Attention, DAttention(Vision Transformer with Deformable Attention CVPR2022)

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
    其中Custom中的block具体支持[链接](#b)

6. ultralytics/models/v5/yolov5-bifpn.yaml

    添加BIFPN到yolov5中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default)  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT)
    2. node_mode  
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

7. ultralytics/models/v5/yolov5-C3-CloAtt.yaml

    使用C3-CloAtt替换C3.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C3中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))  

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

17. ultralytics/models/v5/yolov5-KernelWarehouse.yaml
    
    使用[Towards Parameter-Efficient Dynamic Convolution](https://github.com/OSVAI/KernelWarehouse)添加到yolov5中.  
    使用此模块需要注意,在epoch0-20的时候精度会非常低,过了20epoch会正常.

18. Normalized Gaussian Wasserstein Distance.[论文链接](https://arxiv.org/abs/2110.13389)

    在Loss中使用:
        在ultralytics/yolo/utils/loss.py中的BboxLoss class中的__init__函数里面设置self.nwd_loss为True.  
        比例系数调整self.iou_ratio, self.iou_ratio代表iou的占比,(1-self.iou_ratio)为代表nwd的占比.  
    在TAL标签分配中使用:
        在ultralytics/yolo/utils/tal.py中的def get_box_metrics函数中进行更换即可.
    以上这两可以配合使用,也可以单独使用.

19. SlideLoss and EMASlideLoss.[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)

    在ultralytics/yolo/utils/loss.py中的class v8DetectionLoss进行设定.

20. ultralytics/models/v5/yolov5-C3-DySnakeConv.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C3融合.

21. ultralytics/models/v5/yolov5-EfficientHead.yaml

    对检测头进行重设计,支持10种轻量化检测头.详细请看ultralytics/nn/extra_modules/head.py中的Detect_Efficient class.

22. ultralytics/models/v5/yolov5-AuxHead.yaml

    参考YOLOV7-Aux对YOLOV5添加额外辅助训练头,在训练阶段参与训练,在最终推理阶段去掉.  
    其中辅助训练头的损失权重系数可在ultralytics/yolo/utils/loss.py中的class v8DetectionLoss中的__init__函数中的self.aux_loss_ratio设定,默认值参考yolov7为0.25.

23. ultralytics/models/v5/yolov5-C3-DCNV2.yaml

    使用C3-DCNV2替换C3.(DCNV2为可变形卷积V2)

24. ultralytics/models/v5/yolov5-C3-DCNV3.yaml

    使用C3-DCNV3替换C3.([DCNV3](https://github.com/OpenGVLab/InternImage)为可变形卷积V3(CVPR2023,众多排行榜的SOTA))  
    官方中包含了一些指定版本的DCNV3 whl包,下载后直接pip install xxx即可.具体和安装DCNV3可看百度云链接中的视频.

25. ultralytics/models/v5/yolov5-C3-Faster.yaml

    使用C3-Faster替换C3.(使用FasterNet中的FasterBlock替换C3中的Bottleneck)

26. ultralytics/models/v5/yolov5-C3-ODConv.yaml

    使用C3-ODConv替换C3.(使用ODConv替换C3中的Bottleneck中的Conv)

27. ultralytics/models/v5/yolov5-C3-Faster-EMA.yaml

    使用C3-Faster-EMA替换C3.(C3-Faster-EMA推荐可以放在主干上,Neck和head部分可以选择C3-Faster)

28. ultralytics/models/v5/yolov5-dyhead-DCNV3.yaml

    使用[DCNV3](https://github.com/OpenGVLab/InternImage)替换DyHead中的DCNV2.

29. ultralytics/models/v5/yolov5-FocalModulation.yaml

    使用[Focal Modulation](https://github.com/microsoft/FocalNet)替换SPPF.

30. ultralytics/models/v5/yolov5-C3-DBB.yaml

    使用C3-DBB替换C3.(使用DiverseBranchBlock替换C3中的Bottleneck中的Conv)

31. ultralytics/models/v5/yolov5-C3-OREPA.yaml

    使用C3-OREPA替换C2f.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)

32. ultralytics/models/v5/yolov5-C3-REPVGGOREPA.yaml

    使用C3-REPVGGOREPA替换C3.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)

33. ultralytics/models/v5/yolov5-swintransformer.yaml

    SwinTransformer-Tiny替换yolov5主干.

34. ultralytics/models/v5/yolov5-repvit.yaml

    [RepViT](https://github.com/THU-MIG/RepViT/tree/main)替换yolov5主干.

35. ultralytics/models/v5/yolov5-fasternet-bifpn.yaml

    fasternet与bifpn的结合.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default)  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT)
    2. node_mode  
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

36. ultralytics/models/v5/yolov5-C3-DCNV2-Dynamic.yaml

    利用自研注意力机制MPCA强化DCNV2中的offset和mask.

37. ultralytics/models/v5/yolov5-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块

38. ultralytics/models/v5/yolov5-C3-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided改进C3.

39. ultralytics/models/v5/yolov5-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample进行下采样.

40. ultralytics/models/v5/yolov5-C3-MSBlock.yaml

    使用[YOLO-MS](https://github.com/FishAndWasabi/YOLO-MS/tree/main)中的MSBlock改进C3.

41. ultralytics/models/v5/yolov5-C3-DLKA.yaml

    使用[deformableLKA](https://github.com/xmindflow/deformableLKA)改进C3.

42. ultralytics/models/v5/yolov5-GFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN改进Neck.

43. ultralytics/models/v5/yolov5-SPDConv.yaml

    使用[SPDConv](https://github.com/LabSAINT/SPD-Conv/tree/main)进行下采样.

44. ultralytics/models/v5/yolov5-EfficientRepBiPAN.yaml

    使用[YOLOV6](https://github.com/meituan/YOLOv6/tree/main)中的EfficientRepBiPAN改进Neck.

45. ultralytics/models/v5/yolov5-C3-EMBC.yaml

    使用[Efficientnet](https://blog.csdn.net/weixin_43334693/article/details/131114618?spm=1001.2014.3001.5501)中的MBConv与EffectiveSE改进C3.

46. ultralytics/models/v5/yolov5-SPPF-LSKA.yaml

    使用[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)注意力机制改进SPPF,增强多尺度特征提取能力.

47. ultralytics/models/v5/yolov5-C3-DAttention.yaml

    使用[Vision Transformer with Deformable Attention(CVPR2022)](https://github.com/LeapLabTHU/DAT)改进C2f.  
    使用注意点请看百度云视频.

### YOLOV8
1. ultralytics/models/v8/yolov8-efficientViT.yaml

    (CVPR2023)efficientViT替换yolov8主干.

2. ultralytics/models/v8/yolov8-fasternet.yaml

    (CVPR2023)fasternet替换yolov8主干.

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
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

7. ultralytics/models/v8/yolov8-C2f-Faster.yaml

    使用C2f-Faster替换C2f.(使用FasterNet中的FasterBlock替换C2f中的Bottleneck)

8. ultralytics/models/v8/yolov8-C2f-ODConv.yaml

    使用C2f-ODConv替换C2f.(使用ODConv替换C2f中的Bottleneck中的Conv)

9. ultralytics/models/v8/yolov8-EfficientFormerV2.yaml

    使用EfficientFormerV2网络替换yolov8主干.(需要看[常见错误和解决方案的第五点](#a))  
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
    目前内部整合的注意力可看[链接](#c)

15. Asymptotic Feature Pyramid Network[reference](https://github.com/gyyang23/AFPN/tree/master)

    a. ultralytics/models/v8/yolov8-AFPN-P345.yaml  
    b. ultralytics/models/v8/yolov8-AFPN-P345-Custom.yaml  
    c. ultralytics/models/v8/yolov8-AFPN-P2345.yaml  
    d. ultralytics/models/v8/yolov8-AFPN-P2345-Custom.yaml  
    其中Custom中的block支持这些[结构](#b)

16. ultralytics/models/v8/yolov8-vanillanet.yaml

    vanillanet替换yolov8主干.

17. ultralytics/models/v8/yolov8-C2f-CloAtt.yaml

    使用C2f-CloAtt替换C2f.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C2f中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))  

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

27. ultralytics/models/v8/yolov8-KernelWarehouse.yaml
    
    使用[Towards Parameter-Efficient Dynamic Convolution](https://github.com/OSVAI/KernelWarehouse)添加到yolov8中.  
    使用此模块需要注意,在epoch0-20的时候精度会非常低,过了20epoch会正常.

28. Normalized Gaussian Wasserstein Distance.[论文链接](https://arxiv.org/abs/2110.13389)

    在Loss中使用:
        在ultralytics/yolo/utils/loss.py中的BboxLoss class中的__init__函数里面设置self.nwd_loss为True.  
        比例系数调整self.iou_ratio, self.iou_ratio代表iou的占比,(1-self.iou_ratio)为代表nwd的占比.  
    在TAL标签分配中使用:
        在ultralytics/yolo/utils/tal.py中的def get_box_metrics函数中进行更换即可.
    以上这两可以配合使用,也可以单独使用.

29. SlideLoss and EMASlideLoss.[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)

    在ultralytics/yolo/utils/loss.py中的class v8DetectionLoss进行设定.

30. ultralytics/models/v8/yolov8-C2f-DySnakeConv.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C2f融合.

31. ultralytics/models/v8/yolov8-EfficientHead.yaml

    对检测头进行重设计,支持10种轻量化检测头.详细请看ultralytics/nn/extra_modules/head.py中的Detect_Efficient class.

32. ultralytics/models/v8/yolov8-AuxHead.yaml

    参考YOLOV7-Aux对YOLOV8添加额外辅助训练头,在训练阶段参与训练,在最终推理阶段去掉.  
    其中辅助训练头的损失权重系数可在ultralytics/yolo/utils/loss.py中的class v8DetectionLoss中的__init__函数中的self.aux_loss_ratio设定,默认值参考yolov7为0.25.

33. ultralytics/models/v8/yolov8-C2f-DCNV2.yaml

    使用C2f-DCNV2替换C2f.(DCNV2为可变形卷积V2)

34. ultralytics/models/v8/yolov8-C2f-DCNV3.yaml

    使用C2f-DCNV3替换C2f.([DCNV3](https://github.com/OpenGVLab/InternImage)为可变形卷积V3(CVPR2023,众多排行榜的SOTA))  
    官方中包含了一些指定版本的DCNV3 whl包,下载后直接pip install xxx即可.具体和安装DCNV3可看百度云链接中的视频.

35. ultralytics/models/v8/yolov8-dyhead-DCNV3.yaml

    使用[DCNV3](https://github.com/OpenGVLab/InternImage)替换DyHead中的DCNV2.

36. ultralytics/models/v8/yolov8-FocalModulation.yaml

    使用[Focal Modulation](https://github.com/microsoft/FocalNet)替换SPPF.

37. ultralytics/models/v8/yolov8-C2f-OREPA.yaml

    使用C2f-OREPA替换C2f.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)

38. ultralytics/models/v8/yolov8-C2f-REPVGGOREPA.yaml

    使用C2f-REPVGGOREPA替换C2f.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)

39. ultralytics/models/v8/yolov8-swintransformer.yaml

    SwinTransformer-Tiny替换yolov8主干.

40. ultralytics/models/v8/yolov8-repvit.yaml

    [RepViT](https://github.com/THU-MIG/RepViT/tree/main)替换yolov8主干.

41. ultralytics/models/v8/yolov8-fasternet-bifpn.yaml

    fasternet与bifpn的结合.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default)  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT)
    2. node_mode  
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

42. ultralytics/models/v8/yolov8-C2f-DCNV2-Dynamic.yaml

    利用自研注意力机制MPCA强化DCNV2中的offset和mask.

43. ultralytics/models/v8/yolov8-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块

44. ultralytics/models/v8/yolov8-C2f-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided改进C2f.

45. ultralytics/models/v8/yolov8-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample进行下采样.

46. ultralytics/models/v8/yolov8-C2f-MSBlock.yaml

    使用[YOLO-MS](https://github.com/FishAndWasabi/YOLO-MS/tree/main)中的MSBlock改进C2f.

47. ultralytics/models/v8/yolov8-C2f-DLKA.yaml

    使用[deformableLKA](https://github.com/xmindflow/deformableLKA)改进C2f.

48. ultralytics/models/v8/yolov8-GFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN改进Neck.

49. ultralytics/models/v8/yolov8-SPDConv.yaml

    使用[SPDConv](https://github.com/LabSAINT/SPD-Conv/tree/main)进行下采样.

50. ultralytics/models/v8/yolov8-EfficientRepBiPAN.yaml

    使用[YOLOV6](https://github.com/meituan/YOLOv6/tree/main)中的EfficientRepBiPAN改进Neck.

51. ultralytics/models/v8/yolov8-C2f-EMBC.yaml

    使用[Efficientnet](https://blog.csdn.net/weixin_43334693/article/details/131114618?spm=1001.2014.3001.5501)中的MBConv与EffectiveSE改进C2f.

52. ultralytics/models/v8/yolov8-SPPF-LSKA.yaml

    使用[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)注意力机制改进SPPF,增强多尺度特征提取能力.

53. ultralytics/models/v8/yolov8-C2f-DAttention.yaml

    使用[Vision Transformer with Deformable Attention(CVPR2022)](https://github.com/LeapLabTHU/DAT)改进C2f.  
    使用注意点请看百度云视频.

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

- **20231010-yolov8-v1.23**
    1. 更新使用教程和百度云视频.(更新DAttention使用说明视频).
    2. 增加LSKA, SegNext_Attention, DAttention(Vision Transformer with Deformable Attention CVPR2022).
    3. 使用LSKA改进SPPF,增强多尺度特征提取能力.
    4. 使用[Vision Transformer with Deformable Attention(CVPR2022)]改进C2f,C3.