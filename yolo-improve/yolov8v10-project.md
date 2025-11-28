# [基于Ultralytics的YOLOV8V10改进项目.(69.9¥)](https://github.com/z1069614715/objectdetection_script)

# 目前自带的一些改进方案(目前拥有合计300+个改进点！持续更新！)

# 为了感谢各位对本项目的支持,本项目的赠品是yolov5-PAGCP通道剪枝算法.[具体使用教程](https://www.bilibili.com/video/BV1yh4y1Z7vz/)

# 专栏改进汇总

## YOLOV8系列
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

33. ultralytics/cfg/models/v8/yolov8-bifpn-GLSA.yaml

    使用[GLSA](https://github.com/Barrett-python/DuAT)模块对bifpn进行二次创新.

34. ultralytics/cfg/models/v8/yolov8-BIMAFPN.yaml

    利用BIFPN的思想对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次改进得到BIMAFPN.

35. ultralytics/cfg/models/v8/yolov8-C2f-AdditiveBlock-CGLU.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.

36. ultralytics/cfg/models/v8/yolov8-C2f-MSMHSA-CGLU.yaml

    使用[CMTFNet](https://github.com/DrWuHonglin/CMTFNet/tree/main)中的M2SA和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.

37. ultralytics/cfg/models/v8/yolov8-C2f-IdentityFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的IdentityFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

38. ultralytics/cfg/models/v8/yolov8-C2f-RandomMixing-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的RandomMixing和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

39. ultralytics/cfg/models/v8/yolov8-C2f-PoolingFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的PoolingFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

40. ultralytics/cfg/models/v8/yolov8-C2f-ConvFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的ConvFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

41. ultralytics/cfg/models/v8/yolov8-C2f-CaFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的CaFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

42. ultralytics/cfg/models/v8/yolov8-MAN-Faster.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新改进yolov8.

43. ultralytics/cfg/models/v8/yolov8-MAN-FasterCGLU.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU进行二次创新改进yolov8.

44. ultralytics/cfg/models/v8/yolov8-MAN-Star.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock进行二次创新改进yolov8.

45. ultralytics/cfg/models/v8/yolov8-MutilBackbone-MSGA.yaml

    使用[MSA^2 Net](https://github.com/xmindflow/MSA-2Net)中的Multi-Scale Adaptive Spatial Attention Gate对自研系列MutilBackbone再次创新.

46. ultralytics/cfg/models/v8/yolov8-slimneck-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade对slimneck二次创新.

47. ultralytics/cfg/models/v8/yolov8-MAN-FasterCGLU-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade和[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU进行二次创新改进yolov8.

48. ultralytics/cfg/models/v8/yolov8-CDFA.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的WaveletConv与[AAAI2025 ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)的ContrastDrivenFeatureAggregation结合改进yolov8.

49. ultralytics/cfg/models/v8/yolov8-C2f-StripCGLU.yaml

    使用[Strip R-CNN](https://arxiv.org/pdf/2501.03775)中的StripBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C2f.

50. ultralytics/cfg/models/v8/yolov8-C2f-Faster-KAN.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN对(CVPR2023)fasternet中的FastetBlock进行二次创新.

51. ultralytics/cfg/models/v8/yolov8-C2f-DIMB-KAN.yaml

    在yolov8-C2f-DIMB.yaml的基础上把mlp模块换成[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN.

52. Localization Quality Estimation - Lightweight Shared Convolutional Detection Head

    Localization Quality Estimation模块出自[GFocalV2](https://arxiv.org/abs/2011.12885).
    detect:ultralytics/cfg/models/v8/yolov8-LSCD-LQE.yaml
    seg:ultralytics/cfg/models/v8/yolov8-seg-LSCD-LQE.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LSCD-LQE.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LSCD-LQE.yaml

53. ultralytics/cfg/models/v8/yolov8-C2f-EfficientVIM-CGLU.yaml

    使用[CVPR2025 EfficientViM](https://github.com/mlvlab/EfficientViM)中的EfficientViMBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C2f.

54. ultralytics/cfg/models/v8/yolov8-EUCB-SC.yaml

    使用[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)中的EUCB和[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix改进yolov8的上采样.

55. ultralytics/cfg/models/v8/yolov8-EMBSFPN-SC.yaml

    在ultralytics/cfg/models/v8/yolov8-EMBSFPN.yaml方案上引入[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix.

56. ultralytics/cfg/models/v8/yolov8-MFMMAFPN.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次创新.

57. ultralytics/cfg/models/v8/yolov8-MBSMFFPN.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对yolov8-EMBSFPN.yaml再次创新 Multi-Branch&Scale Modulation-Fusion FPN.

58. ultralytics/cfg/models/v8/yolov8-C2f-mambaout-LSConv.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)的LSConv与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进C2f.

59. ultralytics/cfg/models/v8/yolov8-SOEP-RFPN-MFM.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE和[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对原创改进SOEP再次创新.

60. ultralytics/cfg/models/v8/yolov8-SOEP-PST.yaml

    使用[Pyramid Sparse Transformer](https://arxiv.org/abs/2505.12772)中的Pyramid Sparse Transformer对SOEP进行二次创新.

61. ultralytics/cfg/models/v8/yolov8-MAN-GCConv.yaml

    使用[CVPR2025 Golden Cudgel Network](https://github.com/gyyang23/GCNet)中的GCConv改进[Hyper-YOLO TPAMI2025](https://www.arxiv.org/pdf/2408.04804)中的Mixed Aggregation Network.

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

10. ultralytics/cfg/models/v8/yolov8-EIEStem.yaml

    1. 通过SobelConv分支，可以提取图像的边缘信息。由于Sobel滤波器可以检测图像中强度的突然变化，因此可以很好地捕捉图像的边缘特征。这些边缘特征在许多计算机视觉任务中都非常重要，例如图像分割和物体检测。
    2. EIEStem模块还结合空间信息，除了边缘信息，EIEStem还通过池化分支提取空间信息，保留重要的空间信息。结合边缘信息和空间信息，可以帮助模型更好地理解图像内容。
    3. 通过3D组卷积高效实现Sobel算子。

11. ultralytics/cfg/models/v8/yolov8-C2f-EIEM.yaml

    提出了一种新的EIEStem模块，旨在作为图像识别任务中的高效前端模块。该模块结合了提取边缘信息的SobelConv分支和提取空间信息的卷积分支，能够学习到更加丰富的图像特征表示。
    1. 边缘信息学习: 卷积神经网络 (CNN)通常擅长学习空间信息，但是对于提取图像中的边缘信息可能稍显不足。EIEStem 模块通过SobelConv分支，显式地提取图像的边缘特征。Sobel滤波器是一种经典的边缘检测滤波器，可以有效地捕捉图像中强度的突然变化，从而获得重要的边缘信息。
    2. 空间信息保留: 除了边缘信息，图像中的空间信息也同样重要。EIEStem模块通过一个额外的卷积分支 (conv_branch) 来提取空间信息。与SobelCon 分支不同，conv_branch提取的是原始图像的特征，可以保留丰富的空间细节。
    3. 特征融合: EIEStem模块将来自SobelConv分支和conv_branch提取的特征进行融合 (concatenate)。 这种融合操作使得学习到的特征表示既包含了丰富的边缘信息，又包含了空间信息，能够更加全面地刻画图像内容。

12. ultralytics/cfg/models/v8/yolov8-ContextGuideFPN.yaml

    Context Guide Fusion Module（CGFM）是一个创新的特征融合模块，旨在改进YOLOv8中的特征金字塔网络（FPN）。该模块的设计考虑了多尺度特征融合过程中上下文信息的引导和自适应调整。
    1. 上下文信息的有效融合：通过SE注意力机制，模块能够在特征融合过程中捕捉并利用重要的上下文信息，从而增强特征表示的有效性，并有效引导模型学习检测目标的信息，从而提高模型的检测精度。
    2. 特征增强：通过权重化的特征重组操作，模块能够增强重要特征，同时抑制不重要特征，提升特征图的判别能力。
    3. 简单高效：模块结构相对简单，不会引入过多的计算开销，适合在实时目标检测任务中应用。
    这期视频讲解在B站:https://www.bilibili.com/video/BV1Vx4y1n7hZ/

13. ultralytics/cfg/models/v8/yolov8-LSDECD.yaml

    基于自研轻量化检测头上(LSCD)，使用detail-enhanced convolution进一步改进，提高检测头的细节捕获能力，进一步改善检测精度.
    detect:ultralytics/cfg/models/v8/yolov8-LSDECD.yaml
    segment:ultralytics/cfg/models/v8/yolov8-seg-LSDECD.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LSDECD.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LSDECD.yaml
    1. DEA-Net中设计了一个细节增强卷积（DEConv），具体来说DEConv将先验信息整合到普通卷积层，以增强表征和泛化能力。然后，通过使用重参数化技术，DEConv等效地转换为普通卷积，不需要额外的参数和计算成本。

14. ultralytics/cfg/models/v8/yolov8-C2f-SMPCGLU.yaml

    Self-moving Point Convolutional GLU模型改进C2f.
    SMP来源于[CVPR2023-SMPConv](https://github.com/sangnekim/SMPConv),Convolutional GLU来源于[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt).
    1. 普通的卷积在面对数据中的多样性和复杂性时，可能无法捕捉到有效的特征，因此我们采用了SMPConv，其具备最新的自适应点移动机制，从而更好地捕捉局部特征，提高特征提取的灵活性和准确性。
    2. 在SMPConv后添加CGLU，Convolutional GLU 结合了卷积和门控机制，能够选择性地通过信息通道，提高了特征提取的有效性和灵活性。

15. Re-CalibrationFPN

    为了加强浅层和深层特征的相互交互能力，推出重校准特征金字塔网络(Re-CalibrationFPN).
    P2345：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P2345.yaml(带有小目标检测头的ReCalibrationFPN)
    P345：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P345.yaml
    P3456：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P3456.yaml(带有大目标检测头的ReCalibrationFPN)
    1. 浅层语义较少，但细节丰富，有更明显的边界和减少失真。此外，深层蕴藏着丰富的物质语义信息。因此，直接融合低级具有高级特性的特性可能导致冗余和不一致。为了解决这个问题，我们提出了SBA模块，它有选择地聚合边界信息和语义信息来描绘更细粒度的物体轮廓和重新校准物体的位置。
    2. 相比传统的FPN结构，SBA模块引入了高分辨率和低分辨率特征之间的双向融合机制，使得特征之间的信息传递更加充分，进一步提升了多尺度特征融合的效果。
    3. SBA模块通过自适应的注意力机制，根据特征图的不同分辨率和内容，自适应地调整特征的权重，从而更好地捕捉目标的多尺度特征。

16. ultralytics/cfg/models/v8/yolov8-CSP-PTB.yaml

    Cross Stage Partial - Partially Transformer Block
    在计算机视觉任务中，Transformer结构因其强大的全局特征提取能力而受到广泛关注。然而，由于Transformer结构的计算复杂度较高，直接将其应用于所有通道会导致显著的计算开销。为了在保证高效特征提取的同时降低计算成本，我们设计了一种混合结构，将输入特征图分为两部分，分别由CNN和Transformer处理，结合了卷积神经网络(CNN)和Transformer机制的模块，旨在增强特征提取的能力。
    我们提出了一种名为CSP_PTB(Cross Stage Partial - Partially Transformer Block)的模块，旨在结合CNN和Transformer的优势，通过对输入通道进行部分分配来优化计算效率和特征提取能力。
    1. 融合局部和全局特征：多项研究表明，CNN的感受野大小较少，导致其只能提取局部特征，但Transformer的MHSA能够提取全局特征，能够同时利用两者的优势。
    2. 保证高效特征提取的同时降低计算成本：为了能引入Transformer结构来提取全局特征又不想大幅度增加计算复杂度，因此提出Partially Transformer Block，只对部分通道使用TransformerBlock。
    3. MHSA_CGLU包含Mutil-Head-Self-Attention和[ConvolutionalGLU(TransNext CVPR2024)](https://github.com/DaiShiResearch/TransNeXt)，其中Mutil-Head-Self-Attention负责提取全局特征，ConvolutionalGLU用于增强非线性特征表达能力，ConvolutionalGLU相比于传统的FFN，具有更强的性能。
    4. 可以根据不同的模型大小和具体的运行情况调节用于Transformer的通道数。

17. ultralytics/cfg/models/v8/yolov8-SOEP.yaml  
    
    小目标在正常的P3、P4、P5检测层上略显吃力，比较传统的做法是加上P2检测层来提升小目标的检测能力，但是同时也会带来一系列的问题，例如加上P2检测层后计算量过大、后处理更加耗时等问题，日益激发需要开发新的针对小目标有效的特征金字塔，我们基于原本的PAFPN上进行改进，提出SmallObjectEnhancePyramid，相对于传统的添加P2检测层，我们使用P2特征层经过SPDConv得到富含小目标信息的特征给到P3进行融合，然后使用CSP思想和基于[AAAI2024的OmniKernel](https://ojs.aaai.org/index.php/AAAI/article/view/27907)进行改进得到CSP-OmniKernel进行特征整合，OmniKernel模块由三个分支组成，即三个分支，即全局分支、大分支和局部分支、以有效地学习从全局到局部的特征表征，最终从而提高小目标的检测性能。(该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！)
    出现这个报错的:RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR,如果你是40系显卡,需要更新torch大于2.0，并且cuda大于12.0.

18. ultralytics/cfg/models/v8/yolov8-CGRFPN.yaml

    Context-Guided Spatial Feature Reconstruction Feature Pyramid Network.
    1. 借鉴[ECCV2024-CGRSeg](https://github.com/nizhenliang/CGRSeg)中的Rectangular Self-Calibration Module经过精心设计,用于空间特征重建和金字塔上下文提取,它在水平和垂直方向上捕获全局上下文，并获得轴向全局上下文来显式地建模矩形关键区域.
    2. PyramidContextExtraction Module使用金字塔上下文提取模块（PyramidContextExtraction），有效整合不同层级的特征信息，提升模型的上下文感知能力。
    3. FuseBlockMulti 和 DynamicInterpolationFusion 这些模块用于多尺度特征的融合，通过动态插值和多特征融合，进一步提高了模型的多尺度特征表示能力和提升模型对复杂背景下目标的识别能力。

19. ultralytics/cfg/models/v8/yolov8-FeaturePyramidSharedConv.yaml

    1. 多尺度特征提取
        通过使用不同膨胀率的卷积层，模块能够提取不同尺度的特征。这对捕捉图像中不同大小和不同上下文的信息非常有利。
        低膨胀率捕捉局部细节，高膨胀率捕捉全局上下文。
    2. 参数共享
        使用共享的卷积层 self.share_conv，大大减少了需要训练的参数数量。相比于每个膨胀率使用独立的卷积层，共享卷积层能够减少冗余，提升模型效率。
        减少了模型的存储和计算开销，提升了计算效率。
    3. 高效的通道变换
        通过1x1卷积层 self.cv1 和 self.cv2，模块能够高效地调整通道数，并进行特征融合。1x1卷积层在减少参数量的同时还能保留重要的特征信息。
    4. 更细粒度的特征提取
        FeaturePyramidSharedConv 使用卷积操作进行特征提取，能够捕捉更加细粒度的特征。相比之下，SPPF 的池化操作可能会丢失一些细节信息。
        卷积操作在特征提取时具有更高的灵活性和表达能力，可以更好地捕捉图像中的细节和复杂模式。

20. APT(Adaptive Power Transformation)-TAL.

    为了使不同gt预测对的匹配质量和损失权重更具鉴别性，我们通过自定义的PowerTransformer显著增强高质量预测框的权重，抑制低质量预测框的影响，并使模型在学习的过程可以更关注质量高的预测框。

21. ultralytics/cfg/models/v8/yolov8-EMBSFPN.yaml

    基于BIFPN、[MAF-YOLO](https://arxiv.org/pdf/2407.04381)、[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)提出全新的Efficient Multi-Branch&Scale FPN.
    Efficient Multi-Branch&Scale FPN拥有<轻量化>、<多尺度特征加权融合>、<多尺度高效卷积模块>、<高效上采样模块>、<全局异构核选择机制>。
    1. 具有多尺度高效卷积模块和全局异构核选择机制，Trident网络的研究表明，具有较大感受野的网络更适合检测较大的物体，反之，较小尺度的目标则从较小的感受野中受益，因此我们在FPN阶段，对于不同尺度的特征层选择不同的多尺度卷积核以适应并逐步获得多尺度感知场信息。
    2. 借鉴BIFPN中的多尺度特征加权融合，能把Concat换成Add来减少参数量和计算量的情况下，还能通过不同尺度特征的重要性进行自适用选择加权融合。
    3. 高效上采样模块来源于CVPR2024-EMCAD中的EUCB，能够在保证一定效果的同时保持高效性。

22. ultralytics/cfg/models/v8/yolov8-CSP-PMSFA.yaml

    自研模块:CSP-Partial Multi-Scale Feature Aggregation.
    1. 部分多尺度特征提取：参考CVPR2020-GhostNet、CVPR2024-FasterNet的思想，采用高效的PartialConv，该模块能够从输入中提取多种尺度的特征信息，但它并不是在所有通道上进行这种操作，而是部分（Partial）地进行，从而提高了计算效率。
    2. 增强的特征融合: 最后的 1x1 卷积层通过将不同尺度的特征融合在一起，同时使用残差连接将输入特征与处理后的特征相加，有效保留了原始信息并引入了新的多尺度信息，从而提高模型的表达能力。

23. ultralytics/cfg/models/v8/yolov8-MutilBackbone-DAF.yaml

    自研MutilBackbone-DynamicAlignFusion.
    1. 为了避免在浅层特征图上消耗过多计算资源，设计的MutilBackbone共享一个stem的信息，这个设计有利于避免计算量过大，推理时间过大的问题。
    2. 为了避免不同Backbone信息融合出现不同来源特征之间的空间差异，我们为此设计了DynamicAlignFusion，其先通过融合来自两个不同模块学习到的特征，然后生成一个名为DynamicAlignWeight去调整各自的特征，最后使用一个可学习的通道权重，其可以根据输入特征动态调整两条路径的权重，从而增强模型对不同特征的适应能力。

24. Rep Shared Convolutional Detection Head

    自研重参数轻量化检测头.
    detect:ultralytics/cfg/models/v8/yolov8-RSCD.yaml
    seg:ultralytics/cfg/models/v8/yolov8-seg-RSCD.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-RSCD.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-RSCD.yaml
    1. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.但由于共享参数可能限制模型的表达能力，因为不同特征可能需要不同的卷积核来捕捉复杂的模式。共享参数可能无法充分捕捉这些差异。为了尽量弥补实现轻量化所采取的共享卷积带来的负面影响，我们使用可重参数化卷积，通过引入更多的可学习参数，网络可以更有效地从数据中提取特征，进而弥补轻量化模型后可能带来的精度丢失问题，并且重参数化卷积可以大大提升参数利用率，并且在推理阶段与普通卷积无差，为模型带来无损的优化方案。
    2. 在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.

25. ultralytics/cfg/models/v8/yolov8-CSP-FreqSpatial.yaml

    FreqSpatial 是一个融合时域和频域特征的卷积神经网络（CNN）模块。该模块通过在时域和频域中提取特征，旨在捕捉不同层次的空间和频率信息，以增强模型在处理图像数据时的鲁棒性和表示能力。模块的主要特点是将 Scharr 算子（用于边缘检测）与 时域卷积 和 频域卷积 结合，通过多种视角捕获图像的结构特征。
    1. 时域特征提取：从原始图像中提取出基于空间结构的特征，主要捕捉图像的细节、边缘信息等。
    2. 频域特征提取：从频率域中提取出频率相关的模式，捕捉到图像的低频和高频成分，能够帮助模型在全局和局部的尺度上提取信息。
    3. 特征融合：将时域和频域的特征进行加权相加，得到最终的输出特征图。这种加权融合允许模型同时考虑空间结构信息和频率信息，从而增强模型在多种场景下的表现能力。

26. ultralytics/cfg/models/v8/yolov8-C2f-MutilScaleEdgeInformationSelect.yaml

    基于自研CSP-MutilScaleEdgeInformationEnhance再次创新.
    我们提出了一个 多尺度边缘信息选择模块（MutilScaleEdgeInformationSelect），其目的是从多尺度边缘信息中高效选择与目标任务高度相关的关键特征。为了实现这一目标，我们引入了一个具有通过聚焦更重要的区域能力的注意力机制[ICCV2023 DualDomainSelectionMechanism, DSM](https://github.com/c-yn/FocalNet)。该机制通过聚焦图像中更重要的区域（如复杂边缘和高频信号区域），在多尺度特征中自适应地筛选具有更高任务相关性的特征，从而显著提升了特征选择的精准度和整体模型性能。

27. GlobalEdgeInformationTransfer

    实现版本1：ultralytics/cfg/models/v8/yolov8-GlobalEdgeInformationTransfer1.yaml
    实现版本2：ultralytics/cfg/models/v8/yolov8-GlobalEdgeInformationTransfer2.yaml
    实现版本3：ultralytics/cfg/models/v8/yolov8-GlobalEdgeInformationTransfer3.yaml
    总所周知，物体框的定位非常之依赖物体的边缘信息，但是对于常规的目标检测网络来说，没有任何组件能提高网络对物体边缘信息的关注度，我们需要开发一个能让边缘信息融合到各个尺度所提取的特征中，因此我们提出一个名为GlobalEdgeInformationTransfer(GEIT)的模块，其可以帮助我们把浅层特征中提取到的边缘信息传递到整个backbone上，并与不同尺度的特征进行融合。
    1. 由于原始图像中含有大量背景信息，因此从原始图像上直接提取边缘信息传递到整个backbone上会给网络的学习带来噪声，而且浅层的卷积层会帮助我们过滤不必要的背景信息，因此我们选择在网络的浅层开发一个名为MutilScaleEdgeInfoGenetator的模块，其会利用网络的浅层特征层去生成多个尺度的边缘信息特征图并投放到主干的各个尺度中进行融合。
    2. 对于下采样方面的选择，我们需要较为谨慎，我们的目标是保留并增强边缘信息，同时进行下采样，选择MaxPool 会更合适。它能够保留局部区域的最强特征，更好地体现边缘信息。因为 AvgPool 更适用于需要平滑或均匀化特征的场景，但在保留细节和边缘信息方面的表现不如 MaxPool。
    3. 对于融合部分，ConvEdgeFusion巧妙地结合边缘信息和普通卷积特征，提出了一种新的跨通道特征融合方式。首先，使用conv_channel_fusion进行边缘信息与普通卷积特征的跨通道融合，帮助模型更好地整合不同来源的特征。然后采用conv_3x3_feature_extract进一步提取融合后的特征，以增强模型对局部细节的捕捉能力。最后通过conv_1x1调整输出特征维度。

28. ultralytics/cfg/models/v8/yolov8-C2f-DIMB.yaml

    自研模块DynamicInceptionDWConv2d.(详细请看项目内配置文件.md)

29. ultralytics/cfg/models/v8/yolov8-HAFB-1.yaml
    
    自研Hierarchical Attention Fusion Block.(详细请看项目内配置文件.md)

30. ultralytics/cfg/models/v8/yolov8-HAFB-2.yaml

    HAFB另外一种使用方法.

31. ultralytics/cfg/models/v8/yolov8-MutilBackbone-HAFB.yaml
    
    yolov8-MutilBackbone-DAF.yaml基础上用上HAFB.

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
18. ultralytics/cfg/models/v8/yolov8-mambaout.yaml
     
    使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOut替换BackBone.
19. ultralytics/cfg/models/v8/yolov8-lsnet.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)中的lsnet替换yolov8的backbone.
20. ultralytics/cfg/models/v8/yolov8-overlock.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的overlock-backbone替换backbone.

### SPPF系列
1. ultralytics/cfg/models/v8/yolov8-FocalModulation.yaml

    使用[Focal Modulation](https://github.com/microsoft/FocalNet)替换SPPF.
2. ultralytics/cfg/models/v8/yolov8-SPPF-LSKA.yaml

    使用[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)注意力机制改进SPPF,增强多尺度特征提取能力.
3. ultralytics/cfg/models/v8/yolov8-AIFI.yaml

    使用[RT-DETR](https://arxiv.org/pdf/2304.08069.pdf)中的Attention-based Intrascale Feature Interaction(AIFI)改进yolov8.
4. ultralytics/cfg/models/v8/yolov8-AIFIRepBN.yaml

    使用[ICML-2024 SLAB](https://github.com/xinghaochen/SLAB)中的RepBN改进AIFI.
5. ultralytics/cfg/models/v8/yolov8-ASSR.yaml
     
    使用[CVPR2025 MambaIR](https://github.com/csguoh/MambaIR)中的Attentive State Space Group改进yolov8.

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

15. ultralytics/cfg/models/v8/yolov8-GLSA.yaml

    使用[GLSA](https://github.com/Barrett-python/DuAT)模块改进yolov8的neck.

16. ultralytics/cfg/models/v8/yolov8-CTrans.yaml

    使用[[AAAI2022] UCTransNet](https://github.com/McGregorWwww/UCTransNet/tree/main)中的ChannelTransformer改进yolov8-neck.(需要看[常见错误和解决方案的第五点](#a))  

17. ultralytics/cfg/models/v8/yolov8-p6-CTrans.yaml

    使用[[AAAI2022] UCTransNet](https://github.com/McGregorWwww/UCTransNet/tree/main)中的ChannelTransformer改进yolov8-neck.(带有p6版本)(需要看[常见错误和解决方案的第五点](#a))  

18. ultralytics/cfg/models/v8/yolov8-MAFPN.yaml

    使用[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN改进Neck.

19. Cross-Layer Feature Pyramid Transformer.   

    P345:ultralytics/cfg/models/v8/yolov8-CFPT.yaml
    P2345:ultralytics/cfg/models/v8/yolov8-CFPT-P2345.yaml
    P3456:ultralytics/cfg/models/v8/yolov8-CFPT-P3456.yaml
    P23456:ultralytics/cfg/models/v8/yolov8-CFPT-P23456.yaml

    使用[CFPT](https://github.com/duzw9311/CFPT/tree/main)改进neck.

20. ultralytics/cfg/models/v8/yolov8-hyper.yaml

    使用[Hyper-YOLO TPAMI2025](https://www.arxiv.org/pdf/2408.04804)中的Hypergraph Computation in Semantic Space改进yolov8.

21. ultralytics/cfg/models/v8/yolov8-msga.yaml

    使用[MSA^2 Net](https://github.com/xmindflow/MSA-2Net)中的Multi-Scale Adaptive Spatial Attention Gate改进yolov8-neck.

22. ultralytics/cfg/models/v8/yolov8-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade改进yolov8-neck.

23. ultralytics/cfg/models/v8/yolov8-mscafsa.yaml

    使用[BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation](https://github.com/nkicsl/SF-UNet)的Frequency-Spatial Attention和Multi-scale Progressive Channel Attention改进yolov8-neck.

24. ultralytics/cfg/models/v8/yolov8-fsa.yaml

    使用[BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation](https://github.com/nkicsl/SF-UNet)的Frequency-Spatial Attention改进yolov8.

25. ultralytics/cfg/models/v8/yolov8-MFM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM改进neck.

26. ultralytics/cfg/models/v8/yolov8-GDSAFusion.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的GDSAFusion改进neck.

27. ultralytics/cfg/models/v8/yolov8-RFPN.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE改进YOLOV8-neck.

28. ultralytics/cfg/models/v8/yolov8-PST.yaml

    使用[Pyramid Sparse Transformer](https://arxiv.org/abs/2505.12772)中的Pyramid Sparse Transformer改进neck.

29. ultralytics/cfg/models/v8/yolov8-HS-FPN.yaml

    使用[AAAI2025 HS-FPN](https://github.com/ShiZican/HS-FPN/tree/main)中的HFP和SDP改进yolo-neck.

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
9. Localization Quality Estimation Head

    此模块出自[GFocalV2](https://arxiv.org/abs/2011.12885).
    detect:ultralytics/cfg/models/v8/yolov8-LQEHead.yaml
    segmet:ultralytics/cfg/models/v8/yolov8-seg-LQE.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LQE.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LQE.yaml

### Label Assign系列
1. Adaptive Training Sample Selection匹配策略.

    在ultralytics/utils/loss.py中的class v8DetectionLoss中自行选择对应的self.assigner即可.

### PostProcess系列
1. soft-nms(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,ShapeIoU)

    soft-nms替换nms.(建议:仅在val.py时候使用,具体替换请看20240122版本更新说明)

2. ultralytics/cfg/models/v8/yolov8-nmsfree.yaml

    仿照yolov10的思想采用双重标签分配和一致匹配度量进行训练,后处理不需要NMS!

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

9. ultralytics/cfg/models/v8/yolov8-WaveletPool.yaml

    使用[Wavelet Pooling](https://openreview.net/forum?id=rkhlb8lCZ)改进YOLOV8的上采样和下采样。

10. ultralytics/cfg/models/v8/yolov8-LDConv.yaml

    使用[LDConv](https://github.com/CV-ZhangXin/LDConv/tree/main)改进下采样.

11. ultralytics/cfg/models/v8/yolov8-PSConv.yaml

    使用[AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://github.com/JN-Yang/PConv-SDloss-Data)中的Pinwheel-shaped Convolution改进yolov8.

12. ultralytics/cfg/models/v8/yolov8-EUCB.yaml

    使用[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)中的EUCB改进yolov8的上采样.

13. ultralytics/cfg/models/v8/yolov8-LoGStem.yaml

    使用[LEGNet](https://github.com/lwCVer/LEGNet)中的LoGStem改进Stem(第一第二层卷积).

14. ultralytics/cfg/models/v8/yolov8-FourierConv.yaml

    使用[MIA2025 Fourier Convolution Block with global receptive field for MRI reconstruction](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002743)中的FourierConv改进Conv.

15. ultralytics/cfg/models/v8/yolov8-GCConv.yaml

    使用[CVPR2025 Golden Cudgel Network](https://github.com/gyyang23/GCNet)中的GCConv改进下采样.

16. ultralytics/cfg/models/v8/yolov8-RepStem.yaml

    使用[ICCV2023 FastVit](https://arxiv.org/pdf/2303.14189)中的RepStem改进yolov8下采样.

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

46. ultralytics/cfg/models/v8/yolov8-C2f-KAN.yaml

    KAN In! Mamba Out! Kolmogorov-Arnold Networks.
    目前支持:
    1. FastKANConv2DLayer
    2. KANConv2DLayer
    3. KALNConv2DLayer
    4. KACNConv2DLayer
    5. KAGNConv2DLayer

47. ultralytics/cfg/models/v8/yolov8-C2f-DEConv.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的detail-enhanced convolution改进C2f.

48. ultralytics/cfg/models/v8/yolov8-C2f-Heat.yaml

    使用[vHeat](https://github.com/MzeroMiko/vHeat/tree/main)中的HeatBlock改进C2f.

49. ultralytics/cfg/models/v8/yolov8-C2f-WTConv.yaml

    使用[ECCV2024 Wavelet Convolutions for Large Receptive Fields](https://github.com/BGU-CS-VIL/WTConv)中的WTConv改进C2f-BottleNeck.

50. ultralytics/cfg/models/v8/yolov8-C2f-FMB.yaml

    使用[ECCV2024 SMFANet](https://github.com/Zheng-MJ/SMFANet/tree/main)的Feature Modulation block改进C2f.

51. ultralytics/cfg/models/v8/yolov8-C2f-gConv.yaml

    使用[Rethinking Performance Gains in Image Dehazing Networks](https://arxiv.org/abs/2209.11448)的gConvblock改进C2f.

52. ultralytics/cfg/models/v8/yolov8-C2f-WDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的WDBB改进c2f.

53. ultralytics/cfg/models/v8/yolov8-C2f-DeepDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的DeepDBB改进c2f.

54. ultralytics/cfg/models/v8/yolov8-C2f-AdditiveBlock.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock改进c2f.

55. ultralytics/cfg/models/v8/yolov8-C2f-MogaBlock.yaml

    使用[MogaNet ICLR2024](https://github.com/Westlake-AI/MogaNet)中的MogaBlock改进C2f.

56. ultralytics/cfg/models/v8/yolov8-C2f-IdentityFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的IdentityFormer改进c2f.

57. ultralytics/cfg/models/v8/yolov8-C2f-RandomMixing.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的RandomMixingFormer改进c2f.(需要看[常见错误和解决方案的第五点](#a))

58. ultralytics/cfg/models/v8/yolov8-C2f-PoolingFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的PoolingFormer改进c2f.

59. ultralytics/cfg/models/v8/yolov8-C2f-ConvFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的ConvFormer改进c2f.

60. ultralytics/cfg/models/v8/yolov8-C2f-CaFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的CaFormer改进c2f.

61. ultralytics/cfg/models/v8/yolov8-C2f-SFHF.yaml

    使用[SFHformer ECCV2024](https://github.com/deng-ai-lab/SFHformer)中的block改进C2f.

62. ultralytics/cfg/models/v8/yolov8-C2f-MSM.yaml

    使用[Revitalizing Convolutional Network for Image Restoration TPAMI2024](https://zhuanlan.zhihu.com/p/720777160)中的MSM改进C2f.

63. ultralytics/cfg/models/v8/yolov8-C2f-RAB.yaml

    使用[Pattern Recognition 2024|DRANet](https://github.com/WenCongWu/DRANet)中的HDRAB(hybrid dilated residual attention block)改进C2f.

64. ultralytics/cfg/models/v8/yolov8-C2f-HDRAB.yaml

    使用[Pattern Recognition 2024|DRANet](https://github.com/WenCongWu/DRANet)中的RAB( residual attention block)改进C2f.

65. ultralytics/cfg/models/v8/yolov8n-C2f-LFE.yaml

    使用[Efficient Long-Range Attention Network for Image Super-resolution ECCV2022](https://github.com/xindongzhang/ELAN)中的Local feature extraction改进C2f.

66. ultralytics/cfg/models/v8/yolov8-C2f-SFA.yaml

    使用[FreqFormer](https://github.com/JPWang-CS/FreqFormer)的Frequency-aware Cascade Attention-SFA改进C2f.

67. ultralytics/cfg/models/v8/yolov8-C2f-CTA.yaml

    使用[FreqFormer](https://github.com/JPWang-CS/FreqFormer)的Frequency-aware Cascade Attention-CTA改进C2f.

68. ultralytics/cfg/models/v8/yolov8-C2f-CAMixer.yaml

    使用[CAMixerSR CVPR2024](https://github.com/icandle/CAMixerSR)中的CAMixer改进C2f.

69. ultralytics/cfg/models/v8/yolov8-MAN.yaml

    使用[Hyper-YOLO TPAMI2025](https://www.arxiv.org/pdf/2408.04804)中的Mixed Aggregation Network改进yolov8.

70. ultralytics/cfg/models/v8/yolov8-C2f-HFERB.yaml

    使用[ICCV2023 CRAFT-SR](https://github.com/AVC2-UESTC/CRAFT-SR)中的high-frequency enhancement residual block改进C2f.

71. ultralytics/cfg/models/v8/yolov8-C2f-DTAB.yaml

    使用[AAAI2025 TBSN](https://github.com/nagejacob/TBSN)中的DTAB改进C2f.

72. ultralytics/cfg/models/v8/yolov8-C2f-JDPM.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的joint domain perception module改进C2f.

73. ultralytics/cfg/models/v8/yolov8-C2f-ETB.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的entanglement transformer block改进C2f.

74. ultralytics/cfg/models/v8/yolov8-C2f-AP.yaml

    使用[AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://github.com/JN-Yang/PConv-SDloss-Data)中的Asymmetric Padding bottleneck改进C2f.

75. ultralytics/cfg/models/v8/yolov8-C2f-Strip.yaml

    使用[Strip R-CNN](https://arxiv.org/pdf/2501.03775)中的StripBlock改进C2f.

76. ultralytics/cfg/models/v8/yolov8-C2f-Kat.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAT改进C2f.

77. ultralytics/cfg/models/v8/yolov8-C2f-GlobalFilter.yaml

    使用[T-PAMI Global Filter Networks for Image Classification](https://github.com/raoyongming/GFNet)中的GlobalFilterBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C2f.

78. ultralytics/cfg/models/v8/yolov8-C2f-DynamicFilter.yaml

    使用[AAAI2024 FFT-Based Dynamic Token Mixer for Vision](https://github.com/okojoalg/dfformer)中的DynamicFilter改进C2f.

79. ultralytics/cfg/models/v8/yolov8-RepHMS.yaml
    
    使用[MHAF-YOLO](https://github.com/yang-0201/MHAF-YOLO)中的RepHMS改进yolov8.

80. ultralytics/cfg/models/v8/yolov8-C2f-SAVSS.yaml

    使用[CVPR2025 SCSegamba](https://github.com/Karl1109/SCSegamba)中的Structure-Aware Scanning Strategy改进C2f.

81. ultralytics/cfg/models/v8/yolov8-C2f-mambaout.yaml
     
     使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock改进C2f.

82. ultralytics/cfg/models/v8/yolov8-C2f-EfficientVIM.yaml

    使用[CVPR2025 EfficientViM](https://github.com/mlvlab/EfficientViM)中的EfficientViMBlock改进C2f.

83. ultralytics/cfg/models/v8/yolov8-C2f-LEGM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的LEGM改进C2f.

84. ultralytics/cfg/models/v8/yolov8-C2f-LSBlock.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)中的LSBlock改进C2f.

85. ultralytics/cfg/models/v8/yolov8-C2f-LFEM.yaml

    使用[LEGNet](https://github.com/lwCVer/LEGNet)中的LFEModule改进C2f.

86. ultralytics/cfg/models/v8/yolov8-C2f-RCB.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的RepConvBlock改进C2f.

87. ultralytics/cfg/models/v8/yolov8-C2f-TransMamba.yaml

    使用[TransMamba](https://github.com/sunshangquan/TransMamba)的TransMamba改进C2f

88. ultralytics/cfg/models/v8/yolov8-C2f-EVS.yaml

    使用[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EVS改进C2f

89. ultralytics/cfg/models/v8/yolov8-C2f-EBlock.yaml

    使用[CVPR2025 DarkIR](https://github.com/cidautai/DarkIR)中的EBlock改进C2f.

90. ultralytics/cfg/models/v8/yolov8-C2f-DBlock.yaml

    使用[CVPR2025 DarkIR](https://github.com/cidautai/DarkIR)中的DBlock改进C2f.

91. ultralytics/cfg/models/v8/yolov8-C2f-SFSConv.yaml

    使用[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv改进C2f.

92. ultralytics/cfg/models/v8/yolov8-FCM.yaml

    使用[AAAI2025 FBRT-YOLO](https://github.com/galaxy-oss/FCM)的模块改进yolov8.

93. ultralytics/cfg/models/v8/yolov8-C2f-GroupMamba.yaml

    使用[CVPR2025 GroupMamba](https://github.com/Amshaker/GroupMamba)中的GroupMambaBlock改进C2f.

94. ultralytics/cfg/models/v8/yolov8-C2f-MambaVision.yaml

    使用[CVPR2025 MambaVision](https://github.com/NVlabs/MambaVision)中的MambaVision改进C2f.

95. ultralytics/cfg/models/v8/yolov8-C2f-FourierConv.yaml

    使用[MIA2025 Fourier Convolution Block with global receptive field for MRI reconstruction](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002743)中的FourierConv改进C2f.

96. ultralytics/cfg/models/v8/yolov8-C2f-GLVSS.yaml

    使用[TGRS2025 UMFormer](https://github.com/takeyoutime/UMFormer)中的GLVSS改进C2f.

97. ultralytics/cfg/models/v8/yolov8-C2f-ESC.yaml

    使用[ICCV2025 ESC: Emulating Self-attention with Convolution for Efficient Image Super-Resolution](https://github.com/dslisleedh/ESC)中的ESC改进C2f.

98. ultralytics/cfg/models/v8/yolov8-C2f-ConvAttn.yaml

    使用[ICCV2025 ESC: Emulating Self-attention with Convolution for Efficient Image Super-Resolution](https://github.com/dslisleedh/ESC)中的ConvAttn改进C2f.

99. ultralytics/cfg/models/v8/yolov8-C2f-UniConv.yaml

    使用[ICCV2025 UniConvBlock](https://github.com/ai-paperwithcode/UniConvNet)中的UniConvBlock改进C2f.

100. ultralytics/cfg/models/v8/yolov8-C2f-GCConv.yaml

    使用[CVPR2025 Golden Cudgel Network](https://github.com/gyyang23/GCNet)中的GCConv改进C2f.

101. ultralytics/cfg/models/v8/yolov8-C2f-CFBlock.yaml

    使用[AAAI2024 SCTNet](https://arxiv.org/pdf/2312.17071)中的CFBlock改进C2f.

102. ultralytics/cfg/models/v8/yolov8-C2f-CSSC.yaml

    使用[TGRS2025 ASCNet](https://ieeexplore.ieee.org/document/10855453)中的CSSC改进C2f.

103. ultralytics/cfg/models/v8/yolov8-C2f-CNCM.yaml

    使用[TGRS2025 ASCNet](https://ieeexplore.ieee.org/document/10855453)中的CNCM改进C2f.

104. ultralytics/cfg/models/v8/yolov8-C2f-HFRB.yaml

    使用[ICCV2025 HFRB](https://arxiv.org/pdf/2507.10689)中的HFRB改进C2f.

105. ultralytics/cfg/models/v8/yolov8-C2f-EVA.yaml

    使用[ICIP2025 BEVANET](https://arxiv.org/pdf/2508.07300)中的EVA改进C2f.

106. ultralytics/cfg/models/v8/yolov8-C2f-RMBC.yaml

    使用[PlainUSR](https://arxiv.org/pdf/2409.13435)中的RepMBConv改进C2f.

107. ultralytics/cfg/models/v8/yolov8-C2f-RMBC-LA.yaml

    使用[PlainUSR](https://arxiv.org/pdf/2409.13435)中的RepMBConv和Local Importance-based Attention改进C2f.

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

## YOLOV10系列
#### 以下配置文件都基于v10n，如果需要使用其他大小的模型(s,m,b,l,x)可以看项目视频百度云链接-YOLOV10模型大小切换教程.

### 二次创新系列
1. SlideLoss and EMASlideLoss.[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)

    在ultralytics/utils/loss.py中的class v8DetectionLoss进行设定.

2. ultralytics/cfg/models/v10/yolov10n-RevCol.yaml

    使用[(ICLR2023)Reversible Column Networks](https://github.com/megvii-research/RevCol)对yolov10主干进行重设计,里面的支持更换不同的C2f-Block.

3. ultralytics/cfg/models/v10/yolov10n-BIMAFPN.yaml

    利用BIFPN的思想对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次改进得到BIMAFPN.

4. ultralytics/cfg/models/v10/yolov10n-C2f-AdditiveBlock-CGLU.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.

5. ultralytics/cfg/models/v10/yolov10n-ASF-P2.yaml

    在ultralytics/cfg/models/v8/yolov8-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.

6. ultralytics/cfg/models/v10/yolov10n-ASF-DySample.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion与[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)组合得到Dynamic Sample Attentional Scale Sequence Fusion.

7. ultralytics/cfg/models/v10/yolov10n-goldyolo-asf.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute与[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新改进yolov10的neck.

8. ultralytics/cfg/models/v10/yolov10n-C2f-MSMHSA-CGLU.yaml

    使用[CMTFNet](https://github.com/DrWuHonglin/CMTFNet/tree/main)中的M2SA和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.

9. ultralytics/cfg/models/v10/yolov10n-C2f-IdentityFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的IdentityFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

10. ultralytics/cfg/models/v10/yolov10n-C2f-RandomMixing-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的RandomMixing和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

11. ultralytics/cfg/models/v10/yolov10n-C2f-PoolingFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的PoolingFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

12. ultralytics/cfg/models/v10/yolov10n-C2f-ConvFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的ConvFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

13. ultralytics/cfg/models/v10/yolov10n-C2f-CaFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的CaFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进c2f.

14. ultralytics/cfg/models/v10/yolov10n-dyhead-DCNV3.yaml

    使用[DCNV3](https://github.com/OpenGVLab/InternImage)替换DyHead中的DCNV2.

15. ultralytics/cfg/models/v10/yolov10n-dyhead-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)对DyHead进行二次创新.

16. ultralytics/cfg/models/v10/yolov10n-C2f-iRMB-Cascaded.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.

17. ultralytics/cfg/models/v10/yolov10n-C2f-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.

18. ultralytics/cfg/models/v10/yolov10n-C2f-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.

19. ultralytics/cfg/models/v10/yolov10n-ELA-HSFPN.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN.

20. ultralytics/cfg/models/v10/yolov10n-CA-HSFPN.yaml

    使用[Coordinate Attention CVPR2021](https://github.com/houqb/CoordAttention)改进HSFPN.

21. ultralytics/cfg/models/v10/yolov10n-CAA-HSFPN.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块HSFPN.

22. ultralytics/cfg/models/v10/yolov10n-MAN-Faster.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新改进yolov10.

23. ultralytics/cfg/models/v10/yolov10n-MAN-FasterCGLU.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU进行二次创新改进yolov10.

24. ultralytics/cfg/models/v10/yolov10n-MAN-Star.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock进行二次创新改进yolov10.

25. ultralytics/cfg/models/v10/yolov10n-MutilBackbone-MSGA.yaml

    使用[MSA^2 Net](https://github.com/xmindflow/MSA-2Net)中的Multi-Scale Adaptive Spatial Attention Gate对自研系列MutilBackbone再次创新.

26. ultralytics/cfg/models/v10/yolov10n-slimneck-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade对slimneck二次创新.

27. ultralytics/cfg/models/v10/yolov10n-MAN-FasterCGLU-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade和[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU进行二次创新改进yolov10.

28. ultralytics/cfg/models/v10/yolov10n-CDFA.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的WaveletConv与[AAAI2025 ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)的ContrastDrivenFeatureAggregation结合改进yolov10.

29. ultralytics/cfg/models/v10/yolov10n-C2f-StripCGLU.yaml

    使用[Strip R-CNN](https://arxiv.org/pdf/2501.03775)中的StripBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C2f.

30. ultralytics/cfg/models/v10/yolov10n-C2f-Faster-KAN.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN对(CVPR2023)fasternet中的FastetBlock进行二次创新.

31. ultralytics/cfg/models/v10/yolov10n-C2f-DIMB-KAN.yaml

    在yolov10n-C2f-DIMB.yaml的基础上把mlp模块换成[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN.

32. ultralytics/cfg/models/v10/yolov10n-C2f-EfficientVIM-CGLU.yaml

    使用[CVPR2025 EfficientViM](https://github.com/mlvlab/EfficientViM)中的EfficientViMBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C2f.

33. ultralytics/cfg/models/v10/yolov10n-LSCD-LQE.yaml

    Localization Quality Estimation Head-LSCD-NMSFree,Localization Quality Estimation此模块出自[GFocalV2](https://arxiv.org/abs/2011.12885).

34. ultralytics/cfg/models/v10/yolov10n-EUCB-SC.yaml

    使用[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)中的EUCB和[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix改进yolov10的上采样.

35. ultralytics/cfg/models/v10/yolov10n-EMBSFPN-SC.yaml

    在ultralytics/cfg/models/v10/yolov10n-EMBSFPN.yaml方案上引入[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix.

36. ultralytics/cfg/models/v10/yolov10n-MFMMAFPN.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次创新.

37. ultralytics/cfg/models/v10/yolov10n-MBSMFFPN.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对yolov10n-EMBSFPN.yaml再次创新 Multi-Branch&Scale Modulation-Fusion FPN.

38. ultralytics/cfg/models/v10/yolov10n-C2f-mambaout-LSConv.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)的LSConv与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进C2f.

39. ultralytics/cfg/models/v10/yolov10n-SOEP-RFPN-MFM.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE和[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对原创改进SOEP再次创新.

40. ultralytics/cfg/models/v10/yolov10n-SOEP-PST.yaml

    使用[Pyramid Sparse Transformer](https://arxiv.org/abs/2505.12772)中的Pyramid Sparse Transformer改进SOEP.

41. ultralytics/cfg/models/v10/yolov10n-MAN-GCConv.yaml

    使用[CVPR2025 Golden Cudgel Network](https://github.com/gyyang23/GCNet)中的GCConv改进[Hyper-YOLO TPAMI2025](https://www.arxiv.org/pdf/2408.04804)中的Mixed Aggregation Network.

### 自研系列

1. ultralytics/cfg/models/v10/yolov10n-C2f-EMSC.yaml

    Efficient Multi-Scale Conv.自研模块,具体讲解请看百度云链接中的视频.

2. ultralytics/cfg/models/v10/yolov10n-C2f-EMSCP.yaml

    Efficient Multi-Scale Conv Plus.自研模块,具体讲解请看百度云链接中的视频.

3. ultralytics/cfg/models/v10/yolov10n-LAWDS.yaml

    Light Adaptive-weight downsampling.自研模块,具体讲解请看百度云链接中的视频.

4. ultralytics/cfg/models/v10/yolov10n-LSCD.yaml

    自研轻量化检测头.(Lightweight Shared Convolutional Detection Head)
    1. GroupNorm在FCOS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.
    3. 在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    综合以上，我们可以让检测头做到参数量更少、计算量更少的情况下，尽可能减少精度的损失.

5. ultralytics/cfg/models/v10/yolov10n-CGRFPN.yaml

    Context-Guided Spatial Feature Reconstruction Feature Pyramid Network.
    1. 借鉴[ECCV2024-CGRSeg](https://github.com/nizhenliang/CGRSeg)中的Rectangular Self-Calibration Module经过精心设计,用于空间特征重建和金字塔上下文提取,它在水平和垂直方向上捕获全局上下文，并获得轴向全局上下文来显式地建模矩形关键区域.
    2. PyramidContextExtraction Module使用金字塔上下文提取模块（PyramidContextExtraction），有效整合不同层级的特征信息，提升模型的上下文感知能力。
    3. FuseBlockMulti 和 DynamicInterpolationFusion 这些模块用于多尺度特征的融合，通过动态插值和多特征融合，进一步提高了模型的多尺度特征表示能力和提升模型对复杂背景下目标的识别能力。

6. ultralytics/cfg/models/v10/yolov10n-FeaturePyramidSharedConv.yaml

    1. 多尺度特征提取
        通过使用不同膨胀率的卷积层，模块能够提取不同尺度的特征。这对捕捉图像中不同大小和不同上下文的信息非常有利。
        低膨胀率捕捉局部细节，高膨胀率捕捉全局上下文。
    2. 参数共享
        使用共享的卷积层 self.share_conv，大大减少了需要训练的参数数量。相比于每个膨胀率使用独立的卷积层，共享卷积层能够减少冗余，提升模型效率。
        减少了模型的存储和计算开销，提升了计算效率。
    3. 高效的通道变换
        通过1x1卷积层 self.cv1 和 self.cv2，模块能够高效地调整通道数，并进行特征融合。1x1卷积层在减少参数量的同时还能保留重要的特征信息。
    4. 更细粒度的特征提取
        FeaturePyramidSharedConv 使用卷积操作进行特征提取，能够捕捉更加细粒度的特征。相比之下，SPPF 的池化操作可能会丢失一些细节信息。
        卷积操作在特征提取时具有更高的灵活性和表达能力，可以更好地捕捉图像中的细节和复杂模式。

7. APT(Adaptive Power Transformation)-TAL.

    为了使不同gt预测对的匹配质量和损失权重更具鉴别性，我们通过自定义的PowerTransformer显著增强高质量预测框的权重，抑制低质量预测框的影响，并使模型在学习的过程可以更关注质量高的预测框。

8. ultralytics/cfg/models/v10/yolov10n-SOEP.yaml 

    小目标在正常的P3、P4、P5检测层上略显吃力，比较传统的做法是加上P2检测层来提升小目标的检测能力，但是同时也会带来一系列的问题，例如加上P2检测层后计算量过大、后处理更加耗时等问题，日益激发需要开发新的针对小目标有效的特征金字塔，我们基于原本的PAFPN上进行改进，提出SmallObjectEnhancePyramid，相对于传统的添加P2检测层，我们使用P2特征层经过SPDConv得到富含小目标信息的特征给到P3进行融合，然后使用CSP思想和基于[AAAI2024的OmniKernel](https://ojs.aaai.org/index.php/AAAI/article/view/27907)进行改进得到CSP-OmniKernel进行特征整合，OmniKernel模块由三个分支组成，即三个分支，即全局分支、大分支和局部分支、以有效地学习从全局到局部的特征表征，最终从而提高小目标的检测性能。

9. ultralytics/cfg/models/v10/yolov10n-EMBSFPN.yaml

    基于BIFPN、[MAF-YOLO](https://arxiv.org/pdf/2407.04381)、[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)提出全新的Efficient Multi-Branch&Scale FPN.
    Efficient Multi-Branch&Scale FPN拥有<轻量化>、<多尺度特征加权融合>、<多尺度高效卷积模块>、<高效上采样模块>、<全局异构核选择机制>。
    1. 具有多尺度高效卷积模块和全局异构核选择机制，Trident网络的研究表明，具有较大感受野的网络更适合检测较大的物体，反之，较小尺度的目标则从较小的感受野中受益，因此我们在FPN阶段，对于不同尺度的特征层选择不同的多尺度卷积核以适应并逐步获得多尺度感知场信息。
    2. 借鉴BIFPN中的多尺度特征加权融合，能把Concat换成Add来减少参数量和计算量的情况下，还能通过不同尺度特征的重要性进行自适用选择加权融合。
    3. 高效上采样模块来源于CVPR2024-EMCAD中的EUCB，能够在保证一定效果的同时保持高效性。

10. ultralytics/cfg/models/v10/yolov10n-CSP-PMSFA.yaml

    自研模块:CSP-Partial Multi-Scale Feature Aggregation.
    1. 部分多尺度特征提取：参考CVPR2020-GhostNet、CVPR2024-FasterNet的思想，采用高效的PartialConv，该模块能够从输入中提取多种尺度的特征信息，但它并不是在所有通道上进行这种操作，而是部分（Partial）地进行，从而提高了计算效率。
    2. 增强的特征融合: 最后的 1x1 卷积层通过将不同尺度的特征融合在一起，同时使用残差连接将输入特征与处理后的特征相加，有效保留了原始信息并引入了新的多尺度信息，从而提高模型的表达能力。

11. ultralytics/cfg/models/v10/yolov10n-MutilBackbone-DAF.yaml

    自研MutilBackbone-DynamicAlignFusion.
    1. 为了避免在浅层特征图上消耗过多计算资源，设计的MutilBackbone共享一个stem的信息，这个设计有利于避免计算量过大，推理时间过大的问题。
    2. 为了避免不同Backbone信息融合出现不同来源特征之间的空间差异，我们为此设计了DynamicAlignFusion，其先通过融合来自两个不同模块学习到的特征，然后生成一个名为DynamicAlignWeight去调整各自的特征，最后使用一个可学习的通道权重，其可以根据输入特征动态调整两条路径的权重，从而增强模型对不同特征的适应能力。

12. ultralytics/cfg/models/v10/yolov10n-TADDH.yaml

    自研任务对齐动态检测头
    1. GroupNorm在FCOS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.并且在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    3. 参照TOOD的思想,除了标签分配策略上的任务对齐,我们也在检测头上进行定制任务对齐的结构,现有的目标检测器头部通常使用独立的分类和定位分支,这会导致两个任务之间缺乏交互,TADDH通过特征提取器从多个卷积层中学习任务交互特征,得到联合特征,定位分支使用DCNV2和交互特征生成DCNV2的offset和mask,分类分支使用交互特征进行动态特征选择.

13. ultralytics/cfg/models/v10/yolov10n-C2f-MutilScaleEdgeInformationEnhance.yaml

    自研CSP-MutilScaleEdgeInformationEnhance.
    MutilScaleEdgeInformationEnhance模块结合了多尺度特征提取、边缘信息增强和卷积操作。它的主要目的是从不同尺度上提取特征，突出边缘信息，并将这些多尺度特征整合到一起，最后通过卷积层输出增强的特征。这个模块在特征提取和边缘增强的基础上有很好的表征能力.
    1. 多尺度特征提取：通过 nn.AdaptiveAvgPool2d 进行多尺度的池化，提取不同大小的局部信息，有助于捕捉图像的多层次特征。
    2. 边缘增强：EdgeEnhancer 模块专门用于提取边缘信息，使得网络对边缘的敏感度增强，这对许多视觉任务（如目标检测、语义分割等）有重要作用。
    3. 特征融合：将不同尺度下提取的特征通过插值操作对齐到同一尺度，然后将它们拼接在一起，最后经过卷积层融合成统一的特征表示，能够提高模型对多尺度特征的感知。

14. ultralytics/cfg/models/v10/yolov10n-RSCD.yaml

    自研重参数轻量化检测头.(Rep Shared Convolutional Detection Head)
    1. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.但由于共享参数可能限制模型的表达能力，因为不同特征可能需要不同的卷积核来捕捉复杂的模式。共享参数可能无法充分捕捉这些差异。为了尽量弥补实现轻量化所采取的共享卷积带来的负面影响，我们使用可重参数化卷积，通过引入更多的可学习参数，网络可以更有效地从数据中提取特征，进而弥补轻量化模型后可能带来的精度丢失问题，并且重参数化卷积可以大大提升参数利用率，并且在推理阶段与普通卷积无差，为模型带来无损的优化方案。
    2. 在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.

15. ultralytics/cfg/models/v10/yolov10n-CSP-FreqSpatial.yaml

    FreqSpatial 是一个融合时域和频域特征的卷积神经网络（CNN）模块。该模块通过在时域和频域中提取特征，旨在捕捉不同层次的空间和频率信息，以增强模型在处理图像数据时的鲁棒性和表示能力。模块的主要特点是将 Scharr 算子（用于边缘检测）与 时域卷积 和 频域卷积 结合，通过多种视角捕获图像的结构特征。
    1. 时域特征提取：从原始图像中提取出基于空间结构的特征，主要捕捉图像的细节、边缘信息等。
    2. 频域特征提取：从频率域中提取出频率相关的模式，捕捉到图像的低频和高频成分，能够帮助模型在全局和局部的尺度上提取信息。
    3. 特征融合：将时域和频域的特征进行加权相加，得到最终的输出特征图。这种加权融合允许模型同时考虑空间结构信息和频率信息，从而增强模型在多种场景下的表现能力。

16. ultralytics/cfg/models/v10/yolov10n-C2f-MutilScaleEdgeInformationSelect.yaml

    基于自研CSP-MutilScaleEdgeInformationEnhance再次创新.
    我们提出了一个 多尺度边缘信息选择模块（MutilScaleEdgeInformationSelect），其目的是从多尺度边缘信息中高效选择与目标任务高度相关的关键特征。为了实现这一目标，我们引入了一个具有通过聚焦更重要的区域能力的注意力机制[ICCV2023 DualDomainSelectionMechanism, DSM](https://github.com/c-yn/FocalNet)。该机制通过聚焦图像中更重要的区域（如复杂边缘和高频信号区域），在多尺度特征中自适应地筛选具有更高任务相关性的特征，从而显著提升了特征选择的精准度和整体模型性能。

17. ultralytics/cfg/models/v10/yolov10n-LSDECD.yaml

    基于自研轻量化检测头上(LSCD)，使用detail-enhanced convolution进一步改进，提高检测头的细节捕获能力，进一步改善检测精度.
    关于DEConv在运行的时候重参数化后比重参数化前的计算量还要大的问题:是因为重参数化前thop库其计算不准的问题,看重参数化后的参数即可.
    1. DEA-Net中设计了一个细节增强卷积（DEConv），具体来说DEConv将先验信息整合到普通卷积层，以增强表征和泛化能力。然后，通过使用重参数化技术，DEConv等效地转换为普通卷积，不需要额外的参数和计算成本。

18. ultralytics/cfg/models/v10/yolov10n-ContextGuideFPN.yaml

    Context Guide Fusion Module（CGFM）是一个创新的特征融合模块，旨在改进YOLOv8中的特征金字塔网络（FPN）。该模块的设计考虑了多尺度特征融合过程中上下文信息的引导和自适应调整。
    1. 上下文信息的有效融合：通过SE注意力机制，模块能够在特征融合过程中捕捉并利用重要的上下文信息，从而增强特征表示的有效性，并有效引导模型学习检测目标的信息，从而提高模型的检测精度。
    2. 特征增强：通过权重化的特征重组操作，模块能够增强重要特征，同时抑制不重要特征，提升特征图的判别能力。
    3. 简单高效：模块结构相对简单，不会引入过多的计算开销，适合在实时目标检测任务中应用。

19. Re-CalibrationFPN

    为了加强浅层和深层特征的相互交互能力，推出重校准特征金字塔网络(Re-CalibrationFPN).
    P2345：ultralytics/cfg/models/v10/yolov10n-ReCalibrationFPN-P2345.yaml(带有小目标检测头的ReCalibrationFPN)
    P345：ultralytics/cfg/models/v10/yolov10n-ReCalibrationFPN-P345.yaml
    P3456：ultralytics/cfg/models/v10/yolov10n-ReCalibrationFPN-P3456.yaml(带有大目标检测头的ReCalibrationFPN)
    1. 浅层语义较少，但细节丰富，有更明显的边界和减少失真。此外，深层蕴藏着丰富的物质语义信息。因此，直接融合低级具有高级特性的特性可能导致冗余和不一致。为了解决这个问题，我们提出了[SBA](https://github.com/Barrett-python/DuAT)模块，它有选择地聚合边界信息和语义信息来描绘更细粒度的物体轮廓和重新校准物体的位置。
    2. 相比传统的FPN结构，[SBA](https://github.com/Barrett-python/DuAT)模块引入了高分辨率和低分辨率特征之间的双向融合机制，使得特征之间的信息传递更加充分，进一步提升了多尺度特征融合的效果。
    3. [SBA](https://github.com/Barrett-python/DuAT)模块通过自适应的注意力机制，根据特征图的不同分辨率和内容，自适应地调整特征的权重，从而更好地捕捉目标的多尺度特征。

20. ultralytics/cfg/models/v10/yolov10n-CSP-PTB.yaml

    Cross Stage Partial - Partially Transformer Block
    在计算机视觉任务中，Transformer结构因其强大的全局特征提取能力而受到广泛关注。然而，由于Transformer结构的计算复杂度较高，直接将其应用于所有通道会导致显著的计算开销。为了在保证高效特征提取的同时降低计算成本，我们设计了一种混合结构，将输入特征图分为两部分，分别由CNN和Transformer处理，结合了卷积神经网络(CNN)和Transformer机制的模块，旨在增强特征提取的能力。
    我们提出了一种名为CSP_PTB(Cross Stage Partial - Partially Transformer Block)的模块，旨在结合CNN和Transformer的优势，通过对输入通道进行部分分配来优化计算效率和特征提取能力。
    1. 融合局部和全局特征：多项研究表明，CNN的感受野大小较少，导致其只能提取局部特征，但Transformer的MHSA能够提取全局特征，能够同时利用两者的优势。
    2. 保证高效特征提取的同时降低计算成本：为了能引入Transformer结构来提取全局特征又不想大幅度增加计算复杂度，因此提出Partially Transformer Block，只对部分通道使用TransformerBlock。
    3. MHSA_CGLU包含Mutil-Head-Self-Attention和[ConvolutionalGLU(TransNext CVPR2024)](https://github.com/DaiShiResearch/TransNeXt)，其中Mutil-Head-Self-Attention负责提取全局特征，ConvolutionalGLU用于增强非线性特征表达能力，ConvolutionalGLU相比于传统的FFN，具有更强的性能。
    4. 可以根据不同的模型大小和具体的运行情况调节用于Transformer的通道数。

21. GlobalEdgeInformationTransfer

    实现版本1：ultralytics/cfg/models/v10/yolov10n-GlobalEdgeInformationTransfer1.yaml
    实现版本3：ultralytics/cfg/models/v10/yolov10n-GlobalEdgeInformationTransfer3.yaml
    实现版本2：ultralytics/cfg/models/v10/yolov10n-GlobalEdgeInformationTransfer2.yaml
    总所周知，物体框的定位非常之依赖物体的边缘信息，但是对于常规的目标检测网络来说，没有任何组件能提高网络对物体边缘信息的关注度，我们需要开发一个能让边缘信息融合到各个尺度所提取的特征中，因此我们提出一个名为GlobalEdgeInformationTransfer(GEIT)的模块，其可以帮助我们把浅层特征中提取到的边缘信息传递到整个backbone上，并与不同尺度的特征进行融合。
    1. 由于原始图像中含有大量背景信息，因此从原始图像上直接提取边缘信息传递到整个backbone上会给网络的学习带来噪声，而且浅层的卷积层会帮助我们过滤不必要的背景信息，因此我们选择在网络的浅层开发一个名为MutilScaleEdgeInfoGenetator的模块，其会利用网络的浅层特征层去生成多个尺度的边缘信息特征图并投放到主干的各个尺度中进行融合。
    2. 对于下采样方面的选择，我们需要较为谨慎，我们的目标是保留并增强边缘信息，同时进行下采样，选择MaxPool 会更合适。它能够保留局部区域的最强特征，更好地体现边缘信息。因为 AvgPool 更适用于需要平滑或均匀化特征的场景，但在保留细节和边缘信息方面的表现不如 MaxPool。
    3. 对于融合部分，ConvEdgeFusion巧妙地结合边缘信息和普通卷积特征，提出了一种新的跨通道特征融合方式。首先，使用conv_channel_fusion进行边缘信息与普通卷积特征的跨通道融合，帮助模型更好地整合不同来源的特征。然后采用conv_3x3_feature_extract进一步提取融合后的特征，以增强模型对局部细节的捕捉能力。最后通过conv_1x1调整输出特征维度。

22. ultralytics/cfg/models/v10/yolov10n-C2f-DIMB.yaml

    自研模块DynamicInceptionDWConv2d.(详细请看项目内配置文件.md)

23. ultralytics/cfg/models/v10/yolov10n-HAFB-1.yaml
    
    自研Hierarchical Attention Fusion Block.(详细请看项目内配置文件.md)

24. ultralytics/cfg/models/v10/yolov10n-HAFB-2.yaml

    HAFB另外一种使用方法.

25. ultralytics/cfg/models/v10/yolov10n-MutilBackbone-HAFB.yaml
    
    yolov10n-MutilBackbone-DAF.yaml基础上用上HAFB.

### BackBone系列

1. ultralytics/cfg/models/v10/yolov10n-efficientViT.yaml

    (CVPR2023)efficientViT替换yolov10主干.

2. ultralytics/cfg/models/v10/yolov10n-fasternet.yaml

    (CVPR2023)fasternet替换yolov10主干.

3. ultralytics/cfg/models/v10/yolov10n-timm.yaml

    使用timm支持的主干网络替换yolov10主干.

4. ultralytics/cfg/models/v10/yolov10n-convnextv2.yaml

    使用convnextv2网络替换yolov10主干.

5. ultralytics/cfg/models/v10/yolov10n-EfficientFormerV2.yaml

    使用EfficientFormerV2网络替换yolov10主干.(需要看[常见错误和解决方案的第五点](#a))  

6. ultralytics/cfg/models/v10/yolov10n-vanillanet.yaml

    vanillanet替换yolov10主干.

7. ultralytics/cfg/models/v10/yolov10n-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolov10主干.

8. ultralytics/cfg/models/v10/yolov10n-swintransformer.yaml

    SwinTransformer-Tiny替换yolov10主干.

9. ultralytics/cfg/models/v10/yolov10n-repvit.yaml

    [CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)替换yolov10主干.

10. ultralytics/cfg/models/v10/yolov10n-CSwinTransformer.yaml

    使用[CSWin-Transformer(CVPR2022)](https://github.com/microsoft/CSWin-Transformer/tree/main)替换yolov10主干.(需要看[常见错误和解决方案的第五点](#a))

11. ultralytics/cfg/models/v10/yolov10n-HGNetV2.yaml

    使用HGNetV2作为YOLOV10的backbone.

12. ultralytics/cfg/models/v10/yolov10n-unireplknet.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)替换yolov10主干.

13. ultralytics/cfg/models/v10/yolov10n-TransNeXt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)改进yolov10的backbone.(需要看[常见错误和解决方案的第五点](#a))   

14. ultralytics/cfg/models/v10/yolov10n-rmt.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)改进yolov10的主干.

15. ultralytics/cfg/models/v10/yolov10n-pkinet.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)改进backbone.(需要安装mmcv和mmengine)

16. ultralytics/cfg/models/v10/yolov10n-mobilenetv4.yaml

    使用[MobileNetV4](https://github.com/jaiwei98/MobileNetV4-pytorch/tree/main)改进yolov10的backbone.

17. ultralytics/cfg/models/v10/yolov10n-starnet.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)改进yolov10-backbone.

18. ultralytics/cfg/models/v10/yolov10n-mambaout.yaml
     
    使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOut替换BackBone.

19. ultralytics/cfg/models/v10/yolov10n-lsnet.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)中的lsnet替换yolov10的backbone.

20. ultralytics/cfg/models/v10/yolov10n-overlock.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的overlock-backbone替换backbone.

### SPPF系列

1. ultralytics/cfg/models/v10/yolov10n-FocalModulation.yaml

    使用[Focal Modulation](https://github.com/microsoft/FocalNet)替换SPPF.

2. ultralytics/cfg/models/v10/yolov10n-SPPF-LSKA.yaml

    使用[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)注意力机制改进SPPF,增强多尺度特征提取能力.

3. ultralytics/cfg/models/v10/yolov10n-AIFIRep.yaml

    使用[ICML-2024 SLAB](https://github.com/xinghaochen/SLAB)与AIFI改进yolov10.

### Neck系列

1. ultralytics/cfg/models/v10/yolov10n-bifpn.yaml

    添加BIFPN到yolov10中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        其中支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

2. ultralytics/cfg/models/v10/yolov10n-slimneck.yaml

    使用[VoVGSCSP\VoVGSCSPC和GSConv](https://github.com/AlanLi1997/slim-neck-by-gsconv)替换yolov10 neck中的C2f和Conv.

3. ultralytics/cfg/models/v10/yolov10n-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.

4. ultralytics/cfg/models/v10/yolov10n-MAFPN.yaml

    使用[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN改进Neck.

5. ultralytics/cfg/models/v10/yolov10n-ASF.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolov10.

6. Cross-Layer Feature Pyramid Transformer.   

    P345:ultralytics/cfg/models/v10/yolov10n-CFPT.yaml
    P2345:ultralytics/cfg/models/v10/yolov10n-CFPT-P2345.yaml
    P3456:ultralytics/cfg/models/v10/yolov10n-CFPT-P3456.yaml
    P23456:ultralytics/cfg/models/v10/yolov10n-CFPT-P23456.yaml

    使用[CFPT](https://github.com/duzw9311/CFPT/tree/main)改进neck.
7. ultralytics/cfg/models/v10/yolov10n-RCSOSA.yaml

    使用[RCS-YOLO](https://github.com/mkang315/RCS-YOLO/tree/main)中的RCSOSA替换C2f.

8. ultralytics/cfg/models/v10/yolov10n-GFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN改进Neck.

9. ultralytics/cfg/models/v10/yolov10n-EfficientRepBiPAN.yaml

    使用[YOLOV6](https://github.com/meituan/YOLOv6/tree/main)中的EfficientRepBiPAN改进Neck.

10. ultralytics/cfg/models/v10/yolov10n-HSFPN.yaml

    使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进yolov10的neck.

11. ultralytics/cfg/models/v10/yolov10n-hyper.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的Hypergraph Computation in Semantic Space改进yolov10.

12. ultralytics/cfg/models/v10/yolov10n-msga.yaml

    使用[MSA^2 Net](https://github.com/xmindflow/MSA-2Net)中的Multi-Scale Adaptive Spatial Attention Gate改进yolov10-neck.

13. ultralytics/cfg/models/v10/yolov10n-CGAFusion.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的content-guided attention fusion改进yolov10-neck.

14. ultralytics/cfg/models/v10/yolov10n-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade改进yolov10-neck.

15. ultralytics/cfg/models/v10/yolov10n-fsa.yaml

    使用[BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation](https://github.com/nkicsl/SF-UNet)的Frequency-Spatial Attention改进yolov10.

16. ultralytics/cfg/models/v10/yolov10n-mscafsa.yaml

    使用[BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation](https://github.com/nkicsl/SF-UNet)的Frequency-Spatial Attention和Multi-scale Progressive Channel Attention改进yolov10-neck.

17. ultralytics/cfg/models/v10/yolov10n-MFM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM改进neck.

18. ultralytics/cfg/models/v10/yolov10n-GDSAFusion.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的GDSAFusion改进neck.

19. ultralytics/cfg/models/v10/yolov10n-RFPN.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE改进YOLOV10n-neck.

20. ultralytics/cfg/models/v10/yolov10n-PST.yaml

    使用[Pyramid Sparse Transformer](https://arxiv.org/abs/2505.12772)中的Pyramid Sparse Transformer改进neck.

21. ultralytics/cfg/models/v10/yolov10n-HS-FPN.yaml

    使用[AAAI2025 HS-FPN](https://github.com/ShiZican/HS-FPN/tree/main)中的HFP和SDP改进yolo-neck.

### Head系列

1. ultralytics/cfg/models/v10/yolov10n-dyhead.yaml

    添加基于注意力机制的目标检测头到yolov10中.

2. ultralytics/cfg/models/v10/yolov10n-LQE.yaml

    Localization Quality Estimation Head-NMSFree,Localization Quality Estimation此模块出自[GFocalV2](https://arxiv.org/abs/2011.12885).

### Label Assign系列
### PostProcess系列

### 上下采样算子

1. ultralytics/cfg/models/v10/yolov10n-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample进行下采样.

2. ultralytics/cfg/models/v10/yolov10n-SPDConv.yaml

    使用[SPDConv](https://github.com/LabSAINT/SPD-Conv/tree/main)进行下采样.

3. ultralytics/cfg/models/v10/yolov10n-dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolov10-neck中的上采样.

4. ultralytics/cfg/models/v10/yolov10n-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolov10-neck中的上采样.

5. ultralytics/cfg/models/v10/yolov10n-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolov8的下采样.(请关闭AMP情况下使用)

6. ultralytics/cfg/models/v8=10/yolov10n-v7DS.yaml

    使用[YOLOV7 CVPR2023](https://arxiv.org/abs/2207.02696)的下采样结构改进YOLOV10中的下采样.

7. ultralytics/cfg/models/v10/yolov10n-ADown.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)的下采样结构改进YOLOV10中的下采样.

8. ultralytics/cfg/models/v10/yolov10n-SRFD.yaml

    使用[A Robust Feature Downsampling Module for Remote Sensing Visual Tasks](https://ieeexplore.ieee.org/document/10142024)改进yolov10的下采样.

9. ultralytics/cfg/models/v10/yolov10n-WaveletPool.yaml

    使用[Wavelet Pooling](https://openreview.net/forum?id=rkhlb8lCZ)改进YOLOV10的上采样和下采样。

10. ultralytics/cfg/models/v10/yolov10n-LDConv.yaml

    使用[LDConv](https://github.com/CV-ZhangXin/LDConv/tree/main)改进下采样.

11. ultralytics/cfg/models/v10/yolov10n-PSConv.yaml

    使用[AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://github.com/JN-Yang/PConv-SDloss-Data)中的Pinwheel-shaped Convolution改进yolov10.

12. ultralytics/cfg/models/v10/yolov10n-EUCB.yaml

    使用[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)中的EUCB改进yolov10的上采样.

13. ultralytics/cfg/models/v10/yolov10n-LoGStem.yaml

    使用[LEGNet](https://github.com/lwCVer/LEGNet)中的LoGStem改进Stem(第一第二层卷积).

14. ultralytics/cfg/models/v10/yolov10n-FourierConv.yaml

    使用[MIA2025 Fourier Convolution Block with global receptive field for MRI reconstruction](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002743)中的FourierConv改进Conv.

15. ultralytics/cfg/models/v10/yolov10n-RepStem.yaml

    使用[ICCV2023 FastVit](https://arxiv.org/pdf/2303.14189)中的RepStem改进yolov10下采样.

16. ultralytics/cfg/models/v10/yolov10n-C2f-GCConv.yaml

    使用[CVPR2025 Golden Cudgel Network](https://github.com/gyyang23/GCNet)中的GCConv改进C2f.

### C2f系列

1. ultralytics/cfg/models/v10/yolov10n-C2f-WTConv.yaml

    使用[ECCV2024 Wavelet Convolutions for Large Receptive Fields](https://github.com/BGU-CS-VIL/WTConv)中的WTConv改进C2f-BottleNeck.

2. ultralytics/cfg/models/v10/yolov10n-attention.yaml

    可以看项目视频-如何在yaml配置文件中添加注意力层  
    多种注意力机制在yolov10中的使用. [多种注意力机制github地址](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)  
    目前内部整合的注意力可看[链接](#c)

3. ultralytics/cfg/models/v10/yolov10n-C2f-FMB.yaml

    使用[ECCV2024 SMFANet](https://github.com/Zheng-MJ/SMFANet/tree/main)的Feature Modulation block改进C2f.

4. ultralytics/cfg/models/v10/yolov10n-C2f-Faster.yaml

    使用C2f-Faster替换C2f.(使用FasterNet中的FasterBlock替换C2f中的Bottleneck)

5. ultralytics/cfg/models/v10/yolov10n-C2f-ODConv.yaml

    使用C2f-ODConv替换C2f.(使用ODConv替换C2f中的Bottleneck中的Conv)

6. ultralytics/cfg/models/v10/yolov10n-C2f-Faster-EMA.yaml

    使用C2f-Faster-EMA替换C2f.(C2f-Faster-EMA推荐可以放在主干上,Neck和head部分可以选择C2f-Faster)

7. ultralytics/cfg/models/v10/yolov10n-C2f-DBB.yaml

    使用C2f-DBB替换C2f.(使用DiverseBranchBlock替换C2f中的Bottleneck中的Conv)

8. ultralytics/cfg/models/v10/yolov10n-C2f-CloAtt.yaml

    使用C2f-CloAtt替换C2f.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C2f中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))

9. ultralytics/cfg/models/v10/yolov10n-C2f-gConv.yaml

    使用[Rethinking Performance Gains in Image Dehazing Networks](https://arxiv.org/abs/2209.11448)的gConvblock改进C2f.

10. ultralytics/cfg/models/v10/yolov10n-C2f-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C2f融合.

11. ultralytics/cfg/models/v10/yolov10n-C2f-SCcConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C2f融合.  
    (取名为SCcConv的原因是在windows下命名是不区分大小写的)

12. ultralytics/cfg/models/v10/yolov10n-KernelWarehouse.yaml

    使用[Towards Parameter-Efficient Dynamic Convolution](https://github.com/OSVAI/KernelWarehouse)添加到yolov10中.  
    使用此模块需要注意,在epoch0-20的时候精度会非常低,过了20epoch会正常.

13. ultralytics/cfg/models/v10/yolov10n-C2f-DySnakeConv.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C2f融合.

14. ultralytics/cfg/models/v10/yolov10n-C2f-WDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的WDBB改进c2f.

15. ultralytics/cfg/models/v10/yolov10n-C2f-DeepDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的DeepDBB改进c2f.

16. ultralytics/cfg/models/v10/yolov10n-C2f-AdditiveBlock.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock改进c2f.

17. ultralytics/cfg/models/v10/yolov10n-C2f-MogaBlock.yaml

    使用[MogaNet ICLR2024](https://github.com/Westlake-AI/MogaNet)中的MogaBlock改进C2f.

18. ultralytics/cfg/models/v10/yolov10n-C2f-IdentityFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的IdentityFormer改进c2f.

19. ultralytics/cfg/models/v10/yolov10n-C2f-RandomMixing.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的RandomMixingFormer改进c2f.(需要看[常见错误和解决方案的第五点](#a))

20. ultralytics/cfg/models/v10/yolov10n-C2f-PoolingFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的PoolingFormer改进c2f.

21. ultralytics/cfg/models/v10/yolov10n-C2f-ConvFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的ConvFormer改进c2f.

22. ultralytics/cfg/models/v10/yolov10n-C2f-CaFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的CaFormer改进c2f.

23. ultralytics/cfg/models/v10/yolov10n-C2f-FFCM.yaml

    使用[Efficient Frequency-Domain Image Deraining with Contrastive Regularization ECCV2024](https://github.com/deng-ai-lab/FADformer)中的Fused_Fourier_Conv_Mixer改进C2f.

25. ultralytics/cfg/models/v10/yolov10n-C2f-SFHF.yaml

    使用[SFHformer ECCV2024](https://github.com/deng-ai-lab/SFHformer)中的block改进C2f.

26. ultralytics/cfg/models/v10/yolov10n-C2f-MSM.yaml

    使用[Revitalizing Convolutional Network for Image Restoration TPAMI2024](https://zhuanlan.zhihu.com/p/720777160)中的MSM改进C2f.

27. ultralytics/cfg/models/v10/yolov10n-C2f-iRMB.yaml

    使用[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB改进C2f.

30. ultralytics/cfg/models/v10/yolov10n-C2f-RAB.yaml

    使用[Pattern Recognition 2024|DRANet](https://github.com/WenCongWu/DRANet)中的HDRAB(hybrid dilated residual attention block)改进C2f.

31. ultralytics/cfg/models/v10/yolov10n-C2f-HDRAB.yaml

    使用[Pattern Recognition 2024|DRANet](https://github.com/WenCongWu/DRANet)中的RAB( residual attention block)改进C2f.

32. ultralytics/cfg/models/v10/yolov10n-C2f-LFE.yaml

    使用[Efficient Long-Range Attention Network for Image Super-resolution ECCV2022](https://github.com/xindongzhang/ELAN)中的Local feature extraction改进C2f.

32. ultralytics/cfg/models/v10/yolov10n-C2f-SFA.yaml

    使用[FreqFormer](https://github.com/JPWang-CS/FreqFormer)的Frequency-aware Cascade Attention-SFA改进C2f.

33. ultralytics/cfg/models/v10/yolov10n-C2f-CTA.yaml

    使用[FreqFormer](https://github.com/JPWang-CS/FreqFormer)的Frequency-aware Cascade Attention-CTA改进C2f.

34. ultralytics/cfg/models/v10/yolov10n-C2f-CAMixer.yaml

    使用[CAMixerSR CVPR2024](https://github.com/icandle/CAMixerSR)中的CAMixer改进C2f.

35. ultralytics/cfg/models/v10/yolov10n-MAN.yaml

    使用[Hyper-YOLO TPAMI2025](https://www.arxiv.org/pdf/2408.04804)中的Mixed Aggregation Network改进yolov10.

36. ultralytics/cfg/models/v10/yolov10n-C2f-HFERB.yaml

    使用[ICCV2023 CRAFT-SR](https://github.com/AVC2-UESTC/CRAFT-SR)中的high-frequency enhancement residual block改进C2f.

37. ultralytics/cfg/models/v10/yolov10n-C2f-DTAB.yaml

    使用[AAAI2025 TBSN](https://github.com/nagejacob/TBSN)中的DTAB改进C2f.

38. ultralytics/cfg/models/v10/yolov10n-C2f-JDPM.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的joint domain perception module改进C2f.

39. ultralytics/cfg/models/v10/yolov10n-C2f-ETB.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的entanglement transformer block改进C2f.

40. ultralytics/cfg/models/v10/yolov10n-C2f-AP.yaml

    使用[AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://github.com/JN-Yang/PConv-SDloss-Data)中的Asymmetric Padding bottleneck改进C2f.

41. ultralytics/cfg/models/v10/yolov10n-C2f-Kat.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAT改进C2f.

42. ultralytics/cfg/models/v10/yolov10n-C2f-GlobalFilter.yaml

    使用[T-PAMI Global Filter Networks for Image Classification](https://github.com/raoyongming/GFNet)中的GlobalFilterBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C2f.

43. ultralytics/cfg/models/v10/yolov10n-C2f-DynamicFilter.yaml

    使用[AAAI2024 FFT-Based Dynamic Token Mixer for Vision](https://github.com/okojoalg/dfformer)中的DynamicFilter改进C2f.

44. ultralytics/cfg/models/v10/yolov10n-RepHMS.yaml

    使用[MHAF-YOLO](https://github.com/yang-0201/MHAF-YOLO)中的RepHMS改进yolov10.

45. ultralytics/cfg/models/v10/yolov10n-C2f-SAVSS.yaml

    使用[CVPR2025 SCSegamba](https://github.com/Karl1109/SCSegamba)中的Structure-Aware Scanning Strategy改进C2f.

46. ultralytics/cfg/models/v10/yolov10n-C2f-mambaout.yaml
     
     使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock改进C2f.

47. ultralytics/cfg/models/v10/yolov10n-C2f-EfficientVIM.yaml

    使用[CVPR2025 EfficientViM](https://github.com/mlvlab/EfficientViM)中的EfficientViMBlock改进C2f.

48. ultralytics/cfg/models/v10/yolov10n-C2f-LEGM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的LEGM改进C2f.

49. ultralytics/cfg/models/v10/yolov10n-C2f-RCB.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的RepConvBlock改进C2f.

50. ultralytics/cfg/models/v10/yolov10n-C2f-LFEM.yaml

    使用[LEGNet](https://github.com/lwCVer/LEGNet)中的LFEModule改进C2f.

51. ultralytics/cfg/models/v10/yolov10n-C2f-LSBlock.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)中的LSBlock改进C2f.

52. ultralytics/cfg/models/v10/yolov10n-C2f-TransMamba.yaml

    使用[TransMamba](https://github.com/sunshangquan/TransMamba)的TransMamba改进C2f

53. ultralytics/cfg/models/v10/yolov10n-C2f-EVS.yaml

    使用[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EVS改进C2f.(编译教程请看:20240219版本更新说明)

54. ultralytics/cfg/models/v10/yolov10n-C2f-EBlock.yaml

    使用[CVPR2025 DarkIR](https://github.com/cidautai/DarkIR)中的EBlock改进C2f.

55. ultralytics/cfg/models/v10/yolov10n-C2f-DBlock.yaml

    使用[CVPR2025 DarkIR](https://github.com/cidautai/DarkIR)中的DBlock改进C2f.

56. ultralytics/cfg/models/v10/yolov10n-C2f-SFSConv.yaml

    使用[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv改进C2f.

57. ultralytics/cfg/models/v10/yolov10n-FCM.yaml

    使用[AAAI2025 FBRT-YOLO](https://github.com/galaxy-oss/FCM)的模块改进yolov10.

58. ultralytics/cfg/models/v10/yolov10n-C2f-GroupMamba.yaml

    使用[CVPR2025 GroupMamba](https://github.com/Amshaker/GroupMamba)中的GroupMambaBlock改进C2f.

59. ultralytics/cfg/models/v10/yolov10n-C2f-MambaVision.yaml

    使用[CVPR2025 MambaVision](https://github.com/NVlabs/MambaVision)中的MambaVision改进C2f.

60. ultralytics/cfg/models/v10/yolov10n-C2f-FourierConv.yaml

    使用[MIA2025 Fourier Convolution Block with global receptive field for MRI reconstruction](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002743)中的FourierConv改进C2f.

61. ultralytics/cfg/models/v10/yolov10n-C2f-GLVSS.yaml

    使用[TGRS2025 UMFormer](https://github.com/takeyoutime/UMFormer)中的GLVSS改进C2f.

62. ultralytics/cfg/models/v10/yolov10n-C2f-ESC.yaml

    使用[ICCV2025 ESC: Emulating Self-attention with Convolution for Efficient Image Super-Resolution](https://github.com/dslisleedh/ESC)中的ESC改进C2f.

63. ultralytics/cfg/models/v10/yolov10n-C2f-ConvAttn.yaml

    使用[ICCV2025 ESC: Emulating Self-attention with Convolution for Efficient Image Super-Resolution](https://github.com/dslisleedh/ESC)中的ConvAttn改进C2f.

64. ultralytics/cfg/models/v10/yolov10n-C2f-UniConv.yaml

    使用[ICCV2025 UniConvBlock](https://github.com/ai-paperwithcode/UniConvNet)中的UniConvBlock改进C2f.

65. ultralytics/cfg/models/v10/yolov10n-C2f-GCConv.yaml

    使用[CVPR2025 Golden Cudgel Network](https://github.com/gyyang23/GCNet)中的GCConv改进C2f.

66. ultralytics/cfg/models/v10/yolov10n-C2f-CFBlock.yaml

    使用[AAAI2024 SCTNet](https://arxiv.org/pdf/2312.17071)中的CFBlock改进C2f.

67. ultralytics/cfg/models/v10/yolov10n-C2f-CSSC.yaml

    使用[TGRS2025 ASCNet](https://ieeexplore.ieee.org/document/10855453)中的CSSC改进C2f.

68. ultralytics/cfg/models/v10/yolov10n-C2f-CNCM.yaml

    使用[TGRS2025 ASCNet](https://ieeexplore.ieee.org/document/10855453)中的CNCM改进C2f.

69. ultralytics/cfg/models/v10/yolov10n-C2f-HFRB.yaml

    使用[ICCV2025 HFRB](https://arxiv.org/pdf/2507.10689)中的HFRB改进C2f.

70. ultralytics/cfg/models/v10/yolov10n-C2f-EVA.yaml

    使用[ICIP2025 BEVANET](https://arxiv.org/pdf/2508.07300)中的EVA改进C2f.

71. ultralytics/cfg/models/v10/yolov10n-C2f-RMBC.yaml

    使用[PlainUSR](https://arxiv.org/pdf/2409.13435)中的RepMBConv改进C2f.

72. ultralytics/cfg/models/v10/yolov10n-C2f-RMBC-LA.yaml

    使用[PlainUSR](https://arxiv.org/pdf/2409.13435)中的RepMBConv和Local Importance-based Attention改进C2f.

### PSA系列

1. ultralytics/cfg/models/v10/yolov10n-PTSSA.yaml
    
    使用[Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention改进PSA.

2. ultralytics/cfg/models/v10/yolov10n-ASSR.yaml
     
    使用[CVPR2025 MambaIR](https://github.com/csguoh/MambaIR)中的Attentive State Space Group改进yolov10.

### 组合系列

1. ultralytics/cfg/models/v10/yolov10n-starnet-bifpn.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)和bifpn改进yolov10.

2. ultralytics/cfg/models/v10/yolov10n-ELA-HSFPN-TADDH.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN,使用自研动态动态对齐检测头改进Head.

# Mamba-YOLO
1. [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)

    集成Mamba-YOLO.(需要编译请看百度云视频-20240619版本更新说明)
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L.yaml
    ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml

# Hyper-YOLO
1. ultralytics/cfg/models/hyper-yolo/hyper-yolo.yaml
2. ultralytics/cfg/models/hyper-yolo/hyper-yolot.yaml
3. ultralytics/cfg/models/hyper-yolo/hyper-yolo-seg.yaml


# 注意力系列
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
22. Efficient Local Attention[Efficient Local Attention](https://arxiv.org/abs/2403.01123)
23. CAA(CVPR2024 PKINet中的注意力)
24. CAFM
25. AFGCAttention[Neural Networks ECCV2024](https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387)

# Loss系列
1. SlideLoss,EMASlideLoss.(可动态调节正负样本的系数,让模型更加注重难分类,错误分类的样本上)
2. IoU,GIoU,DIoU,CIoU,EIoU,SIoU,MPDIoU,ShapeIoU.
3. Inner-IoU,Inner-GIoU,Inner-DIoU,Inner-CIoU,Inner-EIoU,Inner-SIoU,Inner-ShapeIoU.
4. Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
5. Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
6. FocalLoss,VarifocalLoss,QualityfocalLoss
7. Focaler-IoU系列(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,WIoU,MPDIoU,ShapeIoU)
8. Powerful-IoU,Powerful-IoUV2,Inner-Powerful-IoU,Inner-Powerful-IoUV2,Focaler-Powerful-IoU,Focaler-Powerful-IoUV2,Wise-Powerful-IoU(v1,v2,v3),Wise-Powerful-IoUV2(v1,v2,v3)[论文链接](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640)
9. Normalized Gaussian Wasserstein Distance.
10. Gaussian Combined Distance.

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

- **20240523-yolov8-v1.55**
    1. KAN In! Mamba Out!,集成pytorch-kan-conv，支持多种KAN变种！
    2. 同步DCNV4-CVPR2024最新代码.
    3. 修复AIFI在某些组合会报错的问题.
    4. 更新使用教程.
    5. 百度云视频增加20240523更新说明.

- **20240526-yolov8-v1.56**
    1. 支持YOLOV8-NMSFree，仿照yolov10的思想采用双重标签分配和一致匹配度量进行训练,后处理不需要NMS!
    2. 新增边缘信息增强模块自研模块，EIEStem、EIEM。
    3. 更新使用教程.
    4. 百度云视频增加20240526更新说明.

- **20240601-yolov8-v1.57**
    1. 新增自研ContextGuideFPN.
    2. 新增detail-enhanced convolution改进c2f.
    3. 新增自研LSDECD，在LSCD的基础上引入可重参数化的detail-enhanced convolution.
    4. 新增自研SMPCGLU，里面的模块分别来自CVPR2023和CVPR2024.
    5. 更新使用教程.
    6. 百度云视频增加20240601更新说明.

- **20240609-yolov8-v1.58**
    1. 新增支持物理传热启发的视觉表征模型vHeat中的vHeatBlock.
    2. 新增自研重校准特征金字塔网络(Re-CalibrationFPN),推出多个版本(P2345,P345,P3456).
    3. 更新使用教程.
    4. 百度云视频增加20240609更新说明.

- **20240613-yolov8-v1.59**
    1. 新增WaveletPool改进上采样和下采样.
    2. 新增自研Cross Stage Partial - Partially Transformer Block模块.
    3. 更新使用教程.
    4. 百度云视频增加20240613更新说明.

- **20240619-yolov8-v1.60**
    1. 集成mamba-yolo.
    2. 新增GLSA改进yolov8-neck.
    3. 新增GLSA对BIFPN进行二次创新.
    4. 更新使用教程.
    5. 百度云视频增加20240619更新说明.

- **20240627-yolov8-v1.61**
    1. 新增UCTransNet中的ChannelTransformer改进yolov8-neck.
    2. 新增自研SmallObjectEnhancePyramid.
    3. 更新使用教程.
    4. 百度云视频增加20240627更新说明.

- **20240707-yolov8-v1.62**
    1. 更新使用教程,增加常见疑问.  

- **20240713-ultralytics-v1.63**
    1. ultralytics版本已更新至8.2.50，后续会更新YOLOv8、YOLOv10的改进方案.
    2. 新增YOLOV10改进、后续会一步一步更新V10的配置文件.（目前更新了backbone系列,一些自研系列的改进到v10中）
    3. 更新使用教程.
    4. 百度云视频增加20240713更新说明.
    5. 百度云视频更新(断点续训教程、计算COCO指标教程、plot_result.py使用教程、项目使用教程必看系列、YOLOV10版本切换教程一)
    6. 补充了EMSC和EMSCP的结构图.

- **20240720-ultralytics-v1.64**
    1. 修复一些已知问题.
    2. 新增自研Context-Guided Spatial Feature Reconstruction Feature Pyramid Network.
    3. 新增Wavelet Convolutions for Large Receptive Fields中的WTConv改进C2f.
    4. 新增UBRFC-Net中的Adaptive Fine-Grained Channel Attention.
    5. 更新使用教程.
    6. 百度云视频增加20240720更新说明.
    7. 增加v10多个改进、主要是上下采样系列.

- **20240729-ultralytics-v1.65**
    1. 新增自研FeaturePyramidSharedConv.
    2. 新增ECCV2024-SMFANet中的Feature Modulation block.
    3. 增加v10多个改进.
    4. 更新使用教程.
    5. 百度云视频增加20240729更新说明.

- **20240803-ultralytics-v1.66**
    1. 新增LDConv.
    2. 新增Rethinking Performance Gains in Image Dehazing Networks中的gConv.
    3. 新增MAF-YOLO中的MAFPN，并利用BIFPN的思想对MAFPN进行二次创新得到BIMAFPN.
    4. 更新使用教程.
    5. 百度云视频增加20240803更新说明.

- **20240813-ultralytics-v1.67**
    1. 新增APT-TAL标签分配策略.
    2. 新增YOLO-MIF中的WDBB、DeepDBB的重参数化模块.
    3. 新增SLAB中的RepBN改进AIFI.
    4. 更新使用教程.
    5. 百度云视频增加20240813更新说明.

- **20240822-ultralytics-v1.68**
    1. 新增CAS-ViT的AdditiveBlock.
    2. 新增TransNeXt的Convolutional GLU对CAS-ViT的AdditiveBlock进行二次创新.
    3. 新增自研Efficient Multi-Branch&Scale FPN.
    4. 新增v10多个改进.
    5. 更新使用教程.
    6. 百度云视频增加20240822更新说明.

- **20240831-ultralytics-v1.69**
    1. 新增CMTFUnet和TransNext的二次创新模块.
    2. 新增自研CSP-Partial Multi-Scale Feature Aggregation.
    3. 更新使用教程.
    4. 百度云视频增加20240831更新说明.

- **20240908-ultralytics-v1.70**
    1. 新增Cross-Layer Feature Pyramid Transformer for Small Object Detection in Aerial Images中的CFPT.
    2. 新增ICLR2024中的MogaBlock.
    3. 新增v10多个改进.
    4. 更新使用教程.
    5. 百度云视频增加20240908更新说明.

- **20240920-ultralytics-v1.71**
    1. 新增CVPR2024-SHViT中的SHSABlock和其的二次创新.
    2. 新增BIBM2024-SMAFormer中的SMAFormerBlock和其的二次创新.
    3. 新增TPAMI2024-FreqFusion中的FreqFusion改进Neck.
    4. 新增v10多个改进.
    5. 更新使用教程.
    6. 百度云视频增加20240920更新说明.

- **20241007-ultralytics-v1.72**
    1. 新增自研MutilBackBone-DynamicAlignFusion.
    2. 新增Metaformer TPAMI2024的IdentityFormer、RandomMixingFormer、PoolingFormer、ConvFormer、CaFormer改进C2f.
    3. 新增Metaformer TPAMI2024的IdentityFormer、RandomMixingFormer、PoolingFormer、ConvFormer、CaFormer与CVPR2024-TranXNet的二次创新模块改进C2f.
    4. 更新使用教程.
    5. 百度云视频增加20241007更新说明.

- **20241024-ultralytics-v1.73**
    1. 增加v10多个改进.
    2. 新增自研CSP-MutilScaleEdgeInformationEnhance.
    3. 新增Efficient Frequency-Domain Image Deraining with Contrastive Regularization中的Fused_Fourier_Conv_Mixer.
    4. 更新使用教程.
    5. 百度云视频增加20241024更新说明.

- **20241031-ultralytics-v1.74**
    1. 新增v8、v10自研Rep Shared Convolutional Detection Head.
    2. 更新使用教程.
    3. 百度云视频增加20241031更新说明.

- **20241109-ultralytics-v1.75**
    1. 新增自研CSP-FreqSpatial.
    2. 新增SFHformer ECCV2024中的block改进C2f.
    3. 新增Revitalizing Convolutional Network for Image Restoration TPAMI2024中的MSM改进C2f.
    4. 增加v10多个改进.
    5. 更新使用教程.
    6. 百度云视频增加20241109更新说明.

- **20241122-ultralytics-v1.76**
    1. 基于自研CSP-MutilScaleEdgeInformationEnhance再次创新得到CSP-MutilScaleEdgeInformationSelect.
    2. 新增Pattern Recognition 2024|DRANet中的HDRAB和RAB模块改进C2f.
    3. 新增ECCV2022-ELAN中的Local feature extraction改进C2f.
    4. 增加v10多个改进.
    5. 更新使用教程.
    6. 百度云视频增加20241122更新说明.

- **20241204-ultralytics-v1.77**
    1. 新增自研GlobalEdgeInformationTransfer.
    2. 新增FreqFormer的Frequency-aware Cascade Attention改进C2f.
    3. 更新使用教程.
    4. 百度云视频增加20241204更新说明.

- **20241219-ultralytics-v1.78**
    1. 新增CAMixerSR中的CAMixer改进C2f.
    2. 新增支持Hyper-YOLO，并可以利用项目自带的改进改进Hyper-YOLO.
    3. 新增Hyper-YOLO中的Hypergraph Computation in Semantic Space和Mixed Aggregation Network的改进.
    4. 更新使用教程.
    5. 百度云视频增加20241219更新说明.

- **20250101-ultralytics-v1.79**
    1. 新增基于Hyper-YOLO中的Mixed Aggregation Network三个二次改进系列.
    2. 新增使用MSA^2 Net中的Multi-Scale Adaptive Spatial Attention Gate改进yolo11-neck.
    3. 新增使用MSA^2 Net中的Multi-Scale Adaptive Spatial Attention Gate改进自研系列的MutilBackbone.
    4. 更新使用教程.
    5. 百度云视频增加20250101更新说明.

- **20250119-ultralytics-v1.80**
    1. 新增CRAFT-SR中的high-frequency enhancement residual block.
    2. 新增AAAI2025-TBSN中的DTAB.
    3. 新增ECCV2024-FSEL中的多个模块.
    4. 新增ACMMM2024-WFEN中的小波变换特征融合.
    5. 新增AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection中的Pinwheel-shaped Convolution类型改进.
    6. 新增AAAI2025 ConDSeg中的ContrastDrivenFeatureAggregation与ACMMM2024 WFEN中的小波变换进行创新.
    7. 更新使用教程.
    8. 百度云视频增加20250119更新说明.

- **20250207-ultralytics-v1.81**
    1. 新增遥感目标检测Strip R-CNN中的StripBlock及其二次创新.
    2. 新增BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation中的Frequency-Spatial Attention和Multi-scale Progressive Channel Attention.
    3. 新增ICLR2025 Kolmogorov-Arnold Transformer中的KAT及其配合FasterBlock的二次创新.<此模块需要编译>
    4. 更新使用教程.
    5. 百度云视频增加20250207更新说明.

- **20250220-ultralytics-v1.82**
    1. 新增自研模块DynamicInceptionDWConv2d.
    2. 新增GlobalFilter和DynamicFilter.
    3. 更新使用教程.
    4. 百度云视频增加20250220更新说明.

- **20250308-ultralytics-v1.83**
    1. 新增自研模块Hierarchical Attention Fusion并提供多种使用方式.
    2. 新增ICLR2025-Token Statistics Transformer改进PSA.
    3. 新增MHAF-YOLO中的RepHMS.<这个是YOLO群内的一个博士新作品>
    4. 更新使用教程.
    5. 百度云视频增加20250308更新说明.

- **20250323-ultralytics-v1.84**
    1. 新增CVPR2025-MambaIR的模块.
    2. 新增CVPR2025-SCSegamba中的模块.
    3. 新增CVPR2025-MambaOut中的模块.
    4. 更新使用教程.
    5. 百度云视频增加20250323更新说明.

- **20250406-ultralytics-v1.85**
    1. 新增CVPR2025-DEIM中的Localization Quality Estimation改进YOLOHead使其分类头同时具备分类score和预测框质量score.
    2. 新增Localization Quality Estimation - Lightweight Shared Convolutional Detection Head.
    3. 新增CVPR2025-EfficientViM和其与CVPR2024-TransNeXt的二次创新后的模块.
    4. 更新使用教程.
    5. 百度云视频增加20250406更新说明.

- **20250426-ultralytics-v1.86**
    1. 新增CVPR2024-EMCAD中的EUCB上采样.
    2. 新增CVPR2024-EMCAD与CVPR2025-BHViT的二次创新模块.
    3. 新增CVPR2024-DCMPNet的多个模块和二次创新的模块.
    4. 新增统计配置文件的计算量和参数量并排序的脚本.
    5. 更新使用教程.
    6. 百度云视频增加20250426更新说明.

- **20250514-ultralytics-v1.87**
    1. 新增LEGNet的LoGStem和LFEModule.
    2. 新增新一代轻量化SOTA的CVPR2025-LSNet的LSNet和LSConv的多个改进和二次创新改进.
    3. 新增CVPR2025-OverLock中的多个模块.
    4. 修改保存权重的逻辑，训练结束(注意是正常训练结束后，手动停止的没有)后统一会保存4个模型，分别是best.pt、last.pt、best_fp32.pt、last_fp32.pt，其中不带fp32后缀的是fp16格式保存的，但由于有些模块对fp16非常敏感，会出现后续使用val.py的时候精度为0的情况，这种情况下可以用后缀带fp32去测试。
    5. 更新使用教程.
    6. 百度云视频增加20250514更新说明.

- **20250601-ultralytics-v1.88**
    1. 新增TransMamba的改进.
    2. 新增CVPR2025-DarkIR的改进.
    3. 新增CVPR2025-EVSSM的改进.
    4. 更新使用教程.
    5. 百度云视频增加20250601更新说明.

- **20250629-ultralytics-v1.89**
    1. 新增ECCV2024-rethinkingfpn中的模块，并对原创改进SOEP再次创新。
    2. 新增CVPR2024-SFSConv的模块.
    3. 新增CVPR2025-GroupMamba中的模块.
    4. 新增CVPR2025-MambaVision中的模块.
    5. 新增AAAI2025-FBRTYOLO中的模块.
    6. 更新使用教程.
    7. 百度云视频增加20250629更新说明.
    8. 修复在torch2.6.0以及以上的版本会出现模型读取失败的问题.

- **20250727-ultralytics-v1.90**
    1. 新增Pyramid Sparse Transformer改进yolo11-neck.
    2. 新增Pyramid Sparse Transformer对SOEP再创新.
    3. 新增MIA2025-FourierConv.
    4. 新增AAAI2025的HS-FPN.
    5. 新增TGRS2025-UMFormer中的模块.
    6. 更新使用教程.
    7. 百度云视频增加20250727更新说明.

- **20250822-ultralytics-v1.91**
    1. 新增ICCV2025-ESC中的多个改进。
    2. 新增ICCV2025-UniConvBlock中的改进。
    3. 更新使用教程.
    4. 百度云视频增加20250822更新说明.

- **20250919-ultralytics-v1.92**
    1. 新增CVPR2025-GCConv模块.
    2. 新增AAAI2024-CFBlock模块.
    3. 新增ICCV2023-FastViT中的RepStem模块.
    4. 更新使用教程.
    5. 百度云视频增加20250919更新说明.

- **20251028-ultralytics-v1.93**
    1. 新增TGRS2025-ASCNet中的模块.
    2. 新增ICCV2025-HFRB模块.
    3. 新增ICIP2025-BEVANET中的模块.
    4. 更新使用教程.
    5. 百度云视频增加20251028更新说明.

- **20251129-ultralytics-v1.94**
    1. 新增GRSL2025-Gaussian Combined Distance,支持在目标框损失和标签分配策略上更改，详细请看LOSS改进系列.md
    2. 新增ACCV2024-PlainUSR中的模块.
    3. 更新使用教程.
    4. 百度云视频增加20251129更新说明.