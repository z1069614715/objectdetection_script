# [基于Ultralytics的RT-DETR改进详细介绍](https://github.com/z1069614715/objectdetection_script)

# 目前自带的一些改进方案(目前拥有合计250+个改进点！持续更新！)

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
10. get_model_erf.py
    绘制模型的有效感受野.[视频链接](https://www.bilibili.com/video/BV1Gx4y1v7ZZ/)
11. export.py
    导出模型脚本
12. test_env.py
    验证一些需要编译的或者难安装的(mmcv)是否成功的代码.[百度云链接](https://pan.baidu.com/s/1sWwvN4UC3blBRVe1twrJAg?pwd=bru5)
13. get_all_yaml_param_and_flops.py
    计算所有yaml的计算量并排序.[百度云链接](https://pan.baidu.com/s/1ZDzglU7EIzzfaUDhAhagBA?pwd=kg8k)

# RT-DETR基准模型

1. ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml(有预训练权重COCO+Objects365,来自RTDETR-Pytorch版本的移植)

    rtdetr-r18 summary: 421 layers, 20184464 parameters, 20184464 gradients, 58.6 GFLOPs
2. ultralytics/cfg/models/rt-detr/rtdetr-r34.yaml(有预训练权重COCO,来自RTDETR-Pytorch版本的移植)

    rtdetr-r34 summary: 525 layers, 31441668 parameters, 31441668 gradients, 90.6 GFLOPs
3. ultralytics/cfg/models/rt-detr/rtdetr-r50-m.yaml(有预训练权重COCO,来自RTDETR-Pytorch版本的移植)

    rtdetr-r50-m summary: 637 layers, 36647020 parameters, 36647020 gradients, 98.3 GFLOPs
4. ultralytics/cfg/models/rt-detr/rtdetr-r50.yaml(有预训练权重COCO+Objects365,来自RTDETR-Pytorch版本的移植)

    rtdetr-r50 summary: 629 layers, 42944620 parameters, 42944620 gradients, 134.8 GFLOPs
5. ultralytics/cfg/models/rt-detr/rtdetr-r101.yaml

    rtdetr-r101 summary: 867 layers, 76661740 parameters, 76661740 gradients, 257.7 GFLOPs
6. ultralytics/cfg/models/rt-detr/rtdetr-l.yaml(有预训练权重)

    rtdetr-l summary: 673 layers, 32970732 parameters, 32970732 gradients, 108.3 GFLOPs
7. ultralytics/cfg/models/rt-detr/rtdetr-x.yaml(有预训练权重)

    rtdetr-x summary: 867 layers, 67468108 parameters, 67468108 gradients, 232.7 GFLOPs
# 专栏改进汇总

### 二次创新系列
1. ultralytics/cfg/models/rt-detr/rtdetr-DCNV2-Dynamic.yaml

    使用自研可变形卷积DCNV2-Dynamic改进resnet18-backbone中的BasicBlock.(详细介绍请看百度云视频-MPCA与DCNV2_Dynamic的说明)
2. ultralytics/cfg/models/rt-detr/rtdetr-iRMB-Cascaded.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进resnet18-backbone中的BasicBlock.(详细介绍请看百度云视频-20231119更新说明)
3. ultralytics/cfg/models/rt-detr/rtdetr-PConv-Rep.yaml

    使用[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv对[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv进行二次创新后改进resnet18-backbone中的BasicBlock.
4. ultralytics/cfg/models/rt-detr/rtdetr-Faster-Rep.yaml

    使用[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv对[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新后改进resnet18-backbone中的BasicBlock.
5. ultralytics/cfg/models/rt-detr/rtdetr-Faster-EMA.yaml

    使用[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)对[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新后改进resnet18-backbone中的BasicBlock.
6. ultralytics/cfg/models/rt-detr/rtdetr-Faster-Rep-EMA.yaml
    
    使用[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv和[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)对[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新后改进resnet18-backbone中的BasicBlock.
7. ultralytics/cfg/models/rt-detr/rtdetr-DWRC3-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)进行二次创新改进rtdetr.
8. ultralytics/cfg/models/rt-detr/rtdetr-ASF-P2.yaml

    在ultralytics/cfg/models/rt-detr/rtdetr-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.
9. ultralytics/cfg/models/rt-detr/rtdetr-slimneck-ASF.yaml

    使用[SlimNeck](https://github.com/AlanLi1997/slim-neck-by-gsconv)中的VoVGSCSP\VoVGSCSPC和GSConv和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进rtdetr中的CCFM.
10. ultralytics/cfg/models/rt-detr/rtdetr-goldyolo-asf.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行改进特征融合模块.
11. ultralytics/cfg/models/rt-detr/rtdetr-HSPAN.yaml

    对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进RTDETR中的CCFM.
12. ultralytics/cfg/models/rt-detr/rtdetr-ASF-Dynamic.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion的上采样模块得到Dynamic Sample Attentional Scale Sequence Fusion改进CCFM.
13. ultralytics/cfg/models/rt-detr/rtdetr-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进resnet18-backbone中的BasicBlock.
14. ultralytics/cfg/models/rt-detr/rtdetr-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进resnet18-backbone中的BasicBlock.
15. ultralytics/cfg/models/rt-detr/rtdetr-DBBNCSPELAN.yaml

    在rtdetr-RepNCSPELAN.yaml使用[Diverse Branch Block CVPR2021](https://arxiv.org/abs/2103.13425)进行二次创新.(详细介绍请看百度云视频-20240225更新说明)

16. ultralytics/cfg/models/rt-detr/rtdetr-OREPANCSPELAN.yaml

    在rtdetr-RepNCSPELAN.yaml使用[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)进行二次创新.(详细介绍请看百度云视频-20240225更新说明)

17. ultralytics/cfg/models/rt-detr/rtdetr-DRBNCSPELAN.yaml

    在rtdetr-RepNCSPELAN.yaml使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock进行二次创新.(详细介绍请看百度云视频-20240225更新说明)

18. ultralytics/cfg/models/rt-detr/rtdetr-Conv3XCNCSPELAN.yaml

    在rtdetr-RepNCSPELAN.yaml使用[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC进行二次创新.(详细介绍请看百度云视频-20240225更新说明)

19. ultralytics/cfg/models/rt-detr/rtdetr-ELA-HSFPN.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN.

20. ultralytics/cfg/models/rt-detr/rtdetr-CA-HSFPN.yaml

    使用[Coordinate Attention CVPR2021](https://github.com/houqb/CoordAttention)改进HSFPN.

21. ultralytics/cfg/models/rt-detr/rtdetr-RepNCSPELAN-CAA.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块改进RepNCSPELAN.

22. ultralytics/cfg/models/rt-detr/rtdetr-CAA-HSFPN.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块HSFPN.

23. ultralytics/cfg/models/rt-detr/rtdetr-CAFMFusion.yaml

    利用具有[HCANet](https://github.com/summitgao/HCANet)中的CAFM，其具有获取全局和局部信息的注意力机制进行二次改进content-guided attention fusion.

24. ultralytics/cfg/models/rt-detr/rtdetr-faster-CGLU.yaml

    使用[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU对CVPR2023中的FasterNet进行二次创新.

25. ultralytics/cfg/models/rt-detr/rtdetr-bifpn-GLSA.yaml

    使用[GLSA](https://github.com/Barrett-python/DuAT)模块对bifpn进行二次创新.

26. ultralytics/cfg/models/rt-detr/rtdetr-BIMAFPN.yaml

    利用BIFPN的思想对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次改进得到BIMAFPN.

27. ultralytics/cfg/models/rt-detr/rtdetr-C2f-AddutuveBlock-CGLU.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU和CSP思想改进backbone.

28. ultralytics/cfg/models/rt-detr/rtdetr-C2f-MSMHSA-CGLU.yaml

    使用[CMTFNet](https://github.com/DrWuHonglin/CMTFNet/tree/main)中的M2SA和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.

29. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SHSA-CGLU.yaml

    使用[SHViT CVPR2024](https://github.com/ysj9909/SHViT)中的SHSABlock与[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU和CSP思想改进backbone.

30. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SMAFB-CGLU.yaml

    使用[SMAFormer BIBM2024](https://github.com/CXH-Research/SMAFormer)中的SMAFormerBlock与[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进与CSP思想改进backbone.

31. ultralytics/cfg/models/rt-detr/rtdetr-MAN-Faster.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新改进rtdetr.

32. ultralytics/cfg/models/rt-detr/rtdetr-MAN-FasterCGLU.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU进行二次创新改进rtdetr.

33. ultralytics/cfg/models/rt-detr/rtdetr-MAN-Star.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock进行二次创新改进rtdetr.

34. ultralytics/cfg/models/rt-detr/rtdetr-MutilBackbone-MSGA.yaml

    使用[MSA^2 Net](https://github.com/xmindflow/MSA-2Net)中的Multi-Scale Adaptive Spatial Attention Gate对自研系列MutilBackbone再次创新.

35. ultralytics/cfg/models/rt-detr/rtdetr-slimneck-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade对slimneck二次创新.

36. ultralytics/cfg/models/rt-detr/rtdetr-CDFA.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的WaveletConv与[AAAI2025 ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)的ContrastDrivenFeatureAggregation结合改进rtdetr.

37. ultralytics/cfg/models/rt-detr/rtdetr-C2f-StripCGLU.yaml

    使用[Strip R-CNN](https://arxiv.org/pdf/2501.03775)中的StripBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU与CSP结合改进backbone.

38. ultralytics/cfg/models/rt-detr/rtdetr-C2f-ELGCA-CGLU.yaml

    使用[ELGC-Net](https://github.com/techmn/elgcnet)中的ELGCA和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU与CSP结合改进backbone.

39. ultralytics/cfg/models/rt-detr/rtdetr-C2f-Faster-KAN.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN对(CVPR2023)fasternet中的FastetBlock进行二次创新.

40. ultralytics/cfg/models/11/yolo11-C3k2-DIMB-KAN.yaml

    在ultralytics/cfg/models/rt-detr/rtdetr-C2f-DIMB.yaml的基础上把mlp模块换成[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN.

41. ultralytics/cfg/models/rt-detr/rtdetr-C2f-EfficientVIM-CGLU.yaml

    使用[CVPR2025 EfficientViM](https://github.com/mlvlab/EfficientViM)中的EfficientViMBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU与CSP结合改进backbone.

42. ultralytics/cfg/models/rt-detr/rtdetr-EUCB-SC.yaml

    使用[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)中的EUCB和[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix改进rtdetr-r18的上采样.

43. ultralytics/cfg/models/rt-detr/rtdetr-EMBSFPN-SC.yaml

    在ultralytics/cfg/models/rt-detr/rtdetr-EMBSFPN.yaml方案上引入[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix.

44. ultralytics/cfg/models/rt-detr/rtdetr-Pola-CGLU.yaml

    使用[ICLR2025 PolaFormer](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention与[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU进行二次创新.

45. ultralytics/cfg/models/rt-detr/rtdetr-Pola-FMFFN.yaml

    使用[ICLR2025 PolaFormer](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention与[ICLR2024-FTIC](https://github.com/qingshi9974/ICLR2024-FTIC)中的的FMFFN进行二次创新.

46. ultralytics/cfg/models/rt-detr/rtdetr-MFMMAFPN.yaml

    利用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次改进得到MFMMAFPN.

47. ultralytics/cfg/models/rt-detr/rtdetr-HyperCompute-MFM.yaml

    利用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的Hypergraph Computation in Semantic Space进行二次创新.

48. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-ASSA-SEFN.yaml

    使用[CVPR2024 Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)中的Adaptive Sparse Self-Attention与[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)改进AIFI.

49. ultralytics/cfg/models/rt-detr/rtdetr-Pola-SEFN.yaml

    使用[ICLR2025 PolaFormer)](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention与[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)改进AIFI.

50. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-ASSA-SEFN-Mona.yaml

    使用[CVPR2024 Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)中的Adaptive Sparse Self-Attention与[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进AIFI.

51. ultralytics/cfg/models/rt-detr/rtdetr-Pola-SEFN-Mona.yaml

    使用[ICLR2025 PolaFormer)](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention与[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进AIFI.

52. ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout-LSConv.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)的LSConv与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进C2f.

53. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-ASSA-SEFN-Mona-DyT.yaml

    使用[CVPR2024 Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)中的Adaptive Sparse Self-Attention与[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进和[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan改进AIFI.

54. ultralytics/cfg/models/rt-detr/rtdetr-Pola-SEFN-Mona-DyT.yaml

    使用[ICLR2025 PolaFormer)](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention与[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进和[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan改进AIFI.

55. ultralytics/cfg/models/rt-detr/rtdetr-Pola-SEFFN-Mona-DyT.yaml

    使用[ICLR2025 PolaFormer)](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention与[TransMamba](https://github.com/sunshangquan/TransMamba)的SpectralEnhancedFFN和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进和[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan改进AIFI.

56. ultralytics/cfg/models/rt-detr/rtdetr-Pola-EDFFN-Mona-DyT.yaml

    使用[ICLR2025 PolaFormer)](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention与[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EDFFN和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进和[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan改进AIFI.

57. ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout-FDConv.yaml

    使用[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv和[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进BackBone.

58. ultralytics/cfg/models/rt-detr/rtdetr-C2f-PFDConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv与[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv二次创新后改进BackBone.

59. ultralytics/cfg/models/rt-detr/rtdetr-C2f-FasterFDConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的FasterBlock与[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv二次创新后改进BackBone.

60. ultralytics/cfg/models/rt-detr/rtdetr-C2f-DSAN-EDFFN.yaml

    使用[DSA: Deformable Spatial Attention](https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks)中的Deformable Spatial Attention Block和[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EDFFN进行二次创新后改进BackBone.

61. ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout-DSA.yaml

    使用[DSA: Deformable Spatial Attention](https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks)中的Deformable Spatial Attention Block与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进BackBone.

62. ultralytics/cfg/models/rt-detr/rtdetr-SOEP-RFPN.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE对原创改进SOEP再次创新.

63. ultralytics/cfg/models/rt-detr/rtdetr-SOEP-MFM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对原创改进SOEP再次创新.

64. ultralytics/cfg/models/rt-detr/rtdetr-SOEP-MFM-RFPN.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE和[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对原创改进SOEP再次创新.

65. ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout-SFSC.yaml

    使用[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进C2f.

66. ultralytics/cfg/models/rt-detr/rtdetr-C2f-PSFSConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv与[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv二次创新后改进C2f.

67. ultralytics/cfg/models/rt-detr/rtdetr-C2f-FasterSFSConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的FasterBlock与[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv二次创新后改进C2f.

### 自研系列
1. ultralytics/cfg/models/rt-detr/rtdetr-PACAPN.yaml

    自研结构, Parallel Atrous Convolution Attention Pyramid Network, PAC-APN
    1. 并行(上/下)采样分支可为网络提供多条特征提取途径，丰富特征表达的多样性、再结合gate机制对采样后的特征进行特征选择，强化更有意义的特征，抑制冗余或不相关的特征，提升特征表达的有效性。
    2. PAC模块通过使用具有不同膨胀率的并行空洞卷积，能够有效地提取不同尺度的特征。这使得网络能够捕捉数据中局部和上下文信息，提高其表示复杂模式的能力。

2. ultralytics/cfg/models/rt-detr/rtdetr-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.

3. ultralytics/cfg/models/rt-detr/rtdetr-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.

4. ultralytics/cfg/models/rt-detr/rtdetr-RGCSPELAN.yaml

    自研RepGhostCSPELAN.
    1. 参考GhostNet中的思想(主流CNN计算的中间特征映射存在广泛的冗余)，采用廉价的操作生成一部分冗余特征图，以此来降低计算量和参数量。
    2. 舍弃yolov5与yolov8中常用的BottleNeck，为了弥补舍弃残差块所带来的性能损失，在梯度流通分支上使用RepConv，以此来增强特征提取和梯度流通的能力，并且RepConv可以在推理的时候进行融合，一举两得。
    3. 可以通过缩放因子控制RGCSPELAN的大小，使其可以兼顾小模型和大模型。

5. ultralytics/cfg/models/rt-detr/rtdetr-ContextGuideFPN.yaml

    Context Guide Fusion Module（CGFM）是一个创新的特征融合模块，旨在改进YOLOv8中的特征金字塔网络（FPN）。该模块的设计考虑了多尺度特征融合过程中上下文信息的引导和自适应调整。
    1. 上下文信息的有效融合：通过SE注意力机制，模块能够在特征融合过程中捕捉并利用重要的上下文信息，从而增强特征表示的有效性，并有效引导模型学习检测目标的信息，从而提高模型的检测精度。
    2. 特征增强：通过权重化的特征重组操作，模块能够增强重要特征，同时抑制不重要特征，提升特征图的判别能力。
    3. 简单高效：模块结构相对简单，不会引入过多的计算开销，适合在实时目标检测任务中应用。
    这期视频讲解在B站:https://www.bilibili.com/video/BV1Vx4y1n7hZ/

6. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SMPCGLU.yaml

    Self-moving Point Convolutional GLU模型改进C2f.
    SMP来源于[CVPR2023-SMPConv](https://github.com/sangnekim/SMPConv),Convolutional GLU来源于[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt).
    1. 普通的卷积在面对数据中的多样性和复杂性时，可能无法捕捉到有效的特征，因此我们采用了SMPConv，其具备最新的自适应点移动机制，从而更好地捕捉局部特征，提高特征提取的灵活性和准确性。
    2. 在SMPConv后添加CGLU，Convolutional GLU 结合了卷积和门控机制，能够选择性地通过信息通道，提高了特征提取的有效性和灵活性。

7. Re-CalibrationFPN

    为了加强浅层和深层特征的相互交互能力，推出重校准特征金字塔网络(Re-CalibrationFPN).
    P2345：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P2345.yaml(带有小目标检测头的ReCalibrationFPN)
    P345：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P345.yaml
    P3456：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P3456.yaml(带有大目标检测头的ReCalibrationFPN)
    1. 浅层语义较少，但细节丰富，有更明显的边界和减少失真。此外，深层蕴藏着丰富的物质语义信息。因此，直接融合低级具有高级特性的特性可能导致冗余和不一致。为了解决这个问题，我们提出了[SBA](https://github.com/Barrett-python/DuAT)模块，它有选择地聚合边界信息和语义信息来描绘更细粒度的物体轮廓和重新校准物体的位置。
    2. 相比传统的FPN结构，[SBA](https://github.com/Barrett-python/DuAT)模块引入了高分辨率和低分辨率特征之间的双向融合机制，使得特征之间的信息传递更加充分，进一步提升了多尺度特征融合的效果。
    3. [SBA](https://github.com/Barrett-python/DuAT)模块通过自适应的注意力机制，根据特征图的不同分辨率和内容，自适应地调整特征的权重，从而更好地捕捉目标的多尺度特征。

8. ultralytics/cfg/models/rt-detr/rtdetr-SOEP.yaml

    小目标在正常的P3、P4、P5检测层上略显吃力，比较传统的做法是加上P2检测层来提升小目标的检测能力，但是同时也会带来一系列的问题，例如加上P2检测层后计算量过大、后处理更加耗时等问题，日益激发需要开发新的针对小目标有效的特征金字塔，我们基于原本的PAFPN上进行改进，提出SmallObjectEnhancePyramid，相对于传统的添加P2检测层，我们使用P2特征层经过SPDConv得到富含小目标信息的特征给到P3进行融合，然后使用CSP思想和基于[AAAI2024的OmniKernel](https://ojs.aaai.org/index.php/AAAI/article/view/27907)进行改进得到CSP-OmniKernel进行特征整合，OmniKernel模块由三个分支组成，即三个分支，即全局分支、大分支和局部分支、以有效地学习从全局到局部的特征表征，最终从而提高小目标的检测性能。

9. ultralytics/cfg/models/rt-detr/rtdetr-CGRFPN.yaml

    Context-Guided Spatial Feature Reconstruction Feature Pyramid Network.
    1. 借鉴[ECCV2024-CGRSeg](https://github.com/nizhenliang/CGRSeg)中的Rectangular Self-Calibration Module经过精心设计,用于空间特征重建和金字塔上下文提取,它在水平和垂直方向上捕获全局上下文，并获得轴向全局上下文来显式地建模矩形关键区域.
    2. PyramidContextExtraction Module使用金字塔上下文提取模块（PyramidContextExtraction），有效整合不同层级的特征信息，提升模型的上下文感知能力。
    3. FuseBlockMulti 和 DynamicInterpolationFusion 这些模块用于多尺度特征的融合，通过动态插值和多特征融合，进一步提高了模型的多尺度特征表示能力和提升模型对复杂背景下目标的识别能力。

10. ultralytics/cfg/models/rt-detr/rtdetr-EMBSFPN.yaml

    基于BIFPN、[MAF-YOLO](https://arxiv.org/pdf/2407.04381)、[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)提出全新的Efficient Multi-Branch&Scale FPN.
    Efficient Multi-Branch&Scale FPN拥有<轻量化>、<多尺度特征加权融合>、<多尺度高效卷积模块>、<高效上采样模块>、<全局异构核选择机制>。
    1. 具有多尺度高效卷积模块和全局异构核选择机制，Trident网络的研究表明，具有较大感受野的网络更适合检测较大的物体，反之，较小尺度的目标则从较小的感受野中受益，因此我们在FPN阶段，对于不同尺度的特征层选择不同的多尺度卷积核以适应并逐步获得多尺度感知场信息。
    2. 借鉴BIFPN中的多尺度特征加权融合，能把Concat换成Add来减少参数量和计算量的情况下，还能通过不同尺度特征的重要性进行自适用选择加权融合。
    3. 高效上采样模块来源于CVPR2024-EMCAD中的EUCB，能够在保证一定效果的同时保持高效性。

11. ultralytics/cfg/models/rt-detr/rtdetr-CSP-PMSFA.yaml

    自研模块:CSP-Partial Multi-Scale Feature Aggregation.
    1. 部分多尺度特征提取：参考CVPR2020-GhostNet、CVPR2024-FasterNet的思想，采用高效的PartialConv，该模块能够从输入中提取多种尺度的特征信息，但它并不是在所有通道上进行这种操作，而是部分（Partial）地进行，从而提高了计算效率。
    2. 增强的特征融合: 最后的 1x1 卷积层通过将不同尺度的特征融合在一起，同时使用残差连接将输入特征与处理后的特征相加，有效保留了原始信息并引入了新的多尺度信息，从而提高模型的表达能力。

12. ultralytics/cfg/models/rt-detr/rtdetr-MutilBackbone-DAF.yaml

    自研MutilBackbone-DynamicAlignFusion.
    1. 为了避免在浅层特征图上消耗过多计算资源，设计的MutilBackbone共享一个stem的信息，这个设计有利于避免计算量过大，推理时间过大的问题。
    2. 为了避免不同Backbone信息融合出现不同来源特征之间的空间差异，我们为此设计了DynamicAlignFusion，其先通过融合来自两个不同模块学习到的特征，然后生成一个名为DynamicAlignWeight去调整各自的特征，最后使用一个可学习的通道权重，其可以根据输入特征动态调整两条路径的权重，从而增强模型对不同特征的适应能力。

13. ultralytics/cfg/models/rt-detr/rtdetr-CSP-MutilScaleEdgeInformationEnhance.yaml

    自研CSP-MutilScaleEdgeInformationEnhance.
    MutilScaleEdgeInformationEnhance模块结合了多尺度特征提取、边缘信息增强和卷积操作。它的主要目的是从不同尺度上提取特征，突出边缘信息，并将这些多尺度特征整合到一起，最后通过卷积层输出增强的特征。这个模块在特征提取和边缘增强的基础上有很好的表征能力.
    1. 多尺度特征提取：通过 nn.AdaptiveAvgPool2d 进行多尺度的池化，提取不同大小的局部信息，有助于捕捉图像的多层次特征。
    2. 边缘增强：EdgeEnhancer 模块专门用于提取边缘信息，使得网络对边缘的敏感度增强，这对许多视觉任务（如目标检测、语义分割等）有重要作用。
    3. 特征融合：将不同尺度下提取的特征通过插值操作对齐到同一尺度，然后将它们拼接在一起，最后经过卷积层融合成统一的特征表示，能够提高模型对多尺度特征的感知。

14. ultralytics/cfg/models/rt-detr/rtdetr-CSP-FreqSpatial.yaml

    FreqSpatial 是一个融合时域和频域特征的卷积神经网络（CNN）模块。该模块通过在时域和频域中提取特征，旨在捕捉不同层次的空间和频率信息，以增强模型在处理图像数据时的鲁棒性和表示能力。模块的主要特点是将 Scharr 算子（用于边缘检测）与 时域卷积 和 频域卷积 结合，通过多种视角捕获图像的结构特征。
    1. 时域特征提取：从原始图像中提取出基于空间结构的特征，主要捕捉图像的细节、边缘信息等。
    2. 频域特征提取：从频率域中提取出频率相关的模式，捕捉到图像的低频和高频成分，能够帮助模型在全局和局部的尺度上提取信息。
    3. 特征融合：将时域和频域的特征进行加权相加，得到最终的输出特征图。这种加权融合允许模型同时考虑空间结构信息和频率信息，从而增强模型在多种场景下的表现能力。

15. ultralytics/cfg/models/rt-detr/rtdetr-CSP-MutilScaleEdgeInformationSelect.yaml

    基于自研CSP-MutilScaleEdgeInformationEnhance再次创新.
    我们提出了一个 多尺度边缘信息选择模块（MutilScaleEdgeInformationSelect），其目的是从多尺度边缘信息中高效选择与目标任务高度相关的关键特征。为了实现这一目标，我们引入了一个具有通过聚焦更重要的区域能力的注意力机制[ICCV2023 DualDomainSelectionMechanism, DSM](https://github.com/c-yn/FocalNet)。该机制通过聚焦图像中更重要的区域（如复杂边缘和高频信号区域），在多尺度特征中自适应地筛选具有更高任务相关性的特征，从而显著提升了特征选择的精准度和整体模型性能。

16. GlobalEdgeInformationTransfer

    总所周知，物体框的定位非常之依赖物体的边缘信息，但是对于常规的目标检测网络来说，没有任何组件能提高网络对物体边缘信息的关注度，我们需要开发一个能让边缘信息融合到各个尺度所提取的特征中，因此我们提出一个名为GlobalEdgeInformationTransfer(GEIT)的模块，其可以帮助我们把浅层特征中提取到的边缘信息传递到整个backbone上，并与不同尺度的特征进行融合。
    1. 由于原始图像中含有大量背景信息，因此从原始图像上直接提取边缘信息传递到整个backbone上会给网络的学习带来噪声，而且浅层的卷积层会帮助我们过滤不必要的背景信息，因此我们选择在网络的浅层开发一个名为MutilScaleEdgeInfoGenetator的模块，其会利用网络的浅层特征层去生成多个尺度的边缘信息特征图并投放到主干的各个尺度中进行融合。
    2. 对于下采样方面的选择，我们需要较为谨慎，我们的目标是保留并增强边缘信息，同时进行下采样，选择MaxPool 会更合适。它能够保留局部区域的最强特征，更好地体现边缘信息。因为 AvgPool 更适用于需要平滑或均匀化特征的场景，但在保留细节和边缘信息方面的表现不如 MaxPool。
    3. 对于融合部分，ConvEdgeFusion巧妙地结合边缘信息和普通卷积特征，提出了一种新的跨通道特征融合方式。首先，使用conv_channel_fusion进行边缘信息与普通卷积特征的跨通道融合，帮助模型更好地整合不同来源的特征。然后采用conv_3x3_feature_extract进一步提取融合后的特征，以增强模型对局部细节的捕捉能力。最后通过conv_1x1调整输出特征维度。

17. ultralytics/cfg/models/rt-detr/rtdetr-C2f-DIMB.yaml

    自研模块DynamicInceptionDWConv2d.(更多解释请看项目内的使用教程.md)

18. ultralytics/cfg/models/rt-detr/rtdetr-HAFB-1.yaml
    
    自研模块Hierarchical Attention Fusion Block.(更多解释请看项目内的使用教程.md)

19. ultralytics/cfg/models/rt-detr/rtdetr-HAFB-2.yaml
     
    HAFB的另外一种使用方式.

20. ultralytics/cfg/models/rt-detr/rtdetr-MutilBackbone-HAFB.yaml

    在rtdetr-MutilBackbone-DAF.yaml上引入HAFB(Hierarchical Attention Fusion Block).

21. ultralytics/cfg/models/rt-detr/rtdetr-RFPN.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE改进rtdetr-neck.

### BackBone系列
1. ultralytics/cfg/models/rt-detr/rt-detr-timm.yaml

    使用[timm](https://github.com/huggingface/pytorch-image-models)库系列的主干替换rtdetr的backbone.(基本支持现有CNN模型)
2. ultralytics/cfg/models/rt-detr/rt-detr-fasternet.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)替换rtdetr的backbone.
3. ultralytics/cfg/models/rt-detr/rt-detr-EfficientViT.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)替换rtdetr的backbone.
4. ultralytics/cfg/models/rt-detr/rtdetr-convnextv2.yaml

    使用[ConvNextV2 2023](https://github.com/facebookresearch/ConvNeXt-V2)替换rtdetr的backbone.
5. ultralytics/cfg/models/rt-detr/rtdetr-EfficientFormerv2.yaml

    使用[EfficientFormerv2 2022](https://github.com/snap-research/EfficientFormer)替换rtdetr的backbone.
6. ultralytics/cfg/models/rt-detr/rtdetr-repvit.yaml

    使用[RepViT ICCV2023](https://github.com/THU-MIG/RepViT)替换rtdetr的backbone.
7. ultralytics/cfg/models/rt-detr/rtdetr-CSwomTramsformer.yaml

    使用[CSwinTramsformer CVPR2022](https://github.com/microsoft/CSWin-Transformer)替换rtdetr的backbone.
8. ultralytics/cfg/models/rt-detr/rtdetr-VanillaNet.yaml

    使用[VanillaNet 2023](https://github.com/huawei-noah/VanillaNet)替换rtdetr的backbone.
9. ultralytics/cfg/models/rt-detr/rtdetr-SwinTransformer.yaml

    使用[SwinTransformer ICCV2021](https://github.com/microsoft/Swin-Transformer)替换rtdetr的backbone.
10. ultralytics/cfg/models/rt-detr/rtdetr-lsknet.yaml

    使用[LSKNet ICCV2023](https://github.com/zcablii/LSKNet)替换rtdetr的backbone.
11. ultralytics/cfg/models/rt-detr/rt-detr-unireplknet.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)替换rtdetr的backbone.
12. ultralytics/cfg/models/rt-detr/rtdetr-TransNeXt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)改进rtdetr的backbone.
13. ultralytics/cfg/models/rt-detr/rtdetr-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN和ADown进行改进RTDETR-R18.
14. ultralytics/cfg/models/rt-detr/rtdetr-rmt.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)改进rtdetr的主干.
15. ultralytics/cfg/models/rt-detr/rtdetr-C2f-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块和C2f改进backbone.
16. ultralytics/cfg/models/rt-detr/rtdetr-C2f-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C2f.
17. ultralytics/cfg/models/rt-detr/rtdetr-mobilenetv4.yaml

    使用[MobileNetV4](https://github.com/jaiwei98/MobileNetV4-pytorch/tree/main)改进rtdetr-backbone.
18. ultralytics/cfg/models/rt-detr/rtdetr-starnet.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)改进yolov8-backbone.

19. ultralytics/cfg/models/rt-detr/rtdetr-C2f-vHeat.yaml

    使用[vHeat](https://github.com/MzeroMiko/vHeat/tree/main)中的HeatBlock和C2f改进backbone.

20. ultralytics/cfg/models/rt-detr/rtdetr-C2f-FMB.yaml

    使用[ECCV2024 SMFANet](https://github.com/Zheng-MJ/SMFANet/tree/main)的Feature Modulation block改进C2f.

21. ultralytics/cfg/models/rt-detr/rtdetr-C2f-gConv.yaml

    使用[Rethinking Performance Gains in Image Dehazing Networks](https://arxiv.org/abs/2209.11448)的gConvblock改进C2f.

22. ultralytics/cfg/models/rt-detr/rtdetr-C2f-AddutuveBlock.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock和CSP思想改进backbone.

23. ultralytics/cfg/models/rt-detr/rtdetr-C2f-MogaBlock.yaml

    使用[MogaNet ICLR2024](https://github.com/Westlake-AI/MogaNet)中的MogaBlock与CSP思想结合改进backbone.

24. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SHSA.yaml

    使用[SHViT CVPR2024](https://github.com/ysj9909/SHViT)中的SHSABlock和CSP思想改进backbone.

25. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SMAFB.yaml

    使用[SMAFormer BIBM2024](https://github.com/CXH-Research/SMAFormer)中的SMAFormerBlock与CSP思想改进backbone.

26. ultralytics/cfg/models/rt-detr/rtdetr-C2f-FFCM.yaml

    使用[Efficient Frequency-Domain Image Deraining with Contrastive Regularization ECCV2024](https://github.com/deng-ai-lab/FADformer)中的Fused_Fourier_Conv_Mixer与CSP思想结合改进rtdetr-backbone.

27. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SFHF.yaml

    使用[SFHformer ECCV2024](https://github.com/deng-ai-lab/SFHformer)中的block与CSP思想结合改进 rtdetr-backbone.

28. ultralytics/cfg/models/rt-detr/rtdetr-C2f-MSM.yaml

    使用[Revitalizing Convolutional Network for Image Restoration TPAMI2024](https://zhuanlan.zhihu.com/p/720777160)中的MSM与CSP思想结合改进rtdetr-backbone.

29. ultralytics/cfg/models/rt-detr/rtdetr-C2f-HDRAB.yaml

    使用[Pattern Recognition 2024|DRANet](https://github.com/WenCongWu/DRANet)中的HDRAB(hybrid dilated residual attention block)结合CSP思想改进backbone.

30. ultralytics/cfg/models/rt-detr/rtdetr-C2f-RAB.yaml

    使用[Pattern Recognition 2024|DRANet](https://github.com/WenCongWu/DRANet)中的RAB( residual attention block)结合CSP思想改进backbone.

31. ultralytics/cfg/models/rt-detr/rtdetr-C2f-FCA.yaml

    使用[FreqFormer](https://github.com/JPWang-CS/FreqFormer)的Frequency-aware Cascade Attention与CSP结合改进backbone.

32. ultralytics/cfg/models/rt-detr/rtdetr-C2f-CAMixer.yaml

    使用[CAMixerSR CVPR2024](https://github.com/icandle/CAMixerSR)中的CAMixer与CSP结合改进backbone.

33. ultralytics/cfg/models/rt-detr/rtdetr-C2f-HFERB.yaml

    使用[ICCV2023 CRAFT-SR](https://github.com/AVC2-UESTC/CRAFT-SR)中的high-frequency enhancement residual block与CSP结合改进backbone.

34. ultralytics/cfg/models/rt-detr/rtdetr-C2f-DTAB.yaml

    使用[AAAI2025 TBSN](https://github.com/nagejacob/TBSN)中的DTAB与CSP结合改进backbone.

35. ultralytics/cfg/models/rt-detr/rtdetr-C2f-JDPM.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的joint domain perception module与CSP结合改进backbone.

36. ultralytics/cfg/models/rt-detr/rtdetr-C2f-ETB.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的entanglement transformer block与CSP结合改进backbone.

37. ultralytics/cfg/models/rt-detr/rtdetr-C2f-FDT.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Full-domain Transformer与CSP结合改进backbone.

38. ultralytics/cfg/models/rt-detr/rtdetr-C2f-AP.yaml

    使用[AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://github.com/JN-Yang/PConv-SDloss-Data)中的Asymmetric Padding bottleneck改进rtdetr.

39. ultralytics/cfg/models/rt-detr/rtdetr-C2f-ELGCA.yaml

    使用[ELGC-Net](https://github.com/techmn/elgcnet)中的ELGCA与CSP结合改进backbone.

40. ultralytics/cfg/models/rt-detr/rtdetr-C2f-Strip.yaml

    使用[Strip R-CNN](https://arxiv.org/pdf/2501.03775)中的StripBlock与CSP结合改进backbone.

41. ultralytics/cfg/models/rt-detr/rtdetr-C2f-KAT.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAT与CSP结合改进backbone.

42. ultralytics/cfg/models/rt-detr/rtdetr-C2f-GlobalFilter.yaml

    使用[T-PAMI Global Filter Networks for Image Classification](https://github.com/raoyongming/GFNet)中的GlobalFilterBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU和CSP改进rtdetr-backbone.

43. ultralytics/cfg/models/rt-detr/rtdetr-C2f-DynamicFilter.yaml

    使用[AAAI2024 FFT-Based Dynamic Token Mixer for Vision](https://github.com/okojoalg/dfformer)中的DynamicFilter与CSP改进rtdetr-backbone.

44. ultralytics/cfg/models/rt-detr/rtdetr-RepHMS.yaml
     
     使用[MHAF-YOLO](https://github.com/yang-0201/MHAF-YOLO)中的RepHMS改进rtdetr.

45. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SAVSS.yaml
    
    使用[CVPR2025 SCSegamba](https://github.com/Karl1109/SCSegamba)中的Structure-Aware Scanning Strategy与CSP结合改进backbone.

46. ultralytics/cfg/models/rt-detr/rtdetr-mambaout.yaml
     
    使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOut替换BackBone.

47. ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout.yaml

    使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOut与CSP结合改进backbone.

48. ultralytics/cfg/models/rt-detr/rtdetr-C2f-EfficientVIM.yaml

    使用[CVPR2025 EfficientViM](https://github.com/mlvlab/EfficientViM)中的EfficientViMBlock与CSP结合改进backbone.

49. ultralytics/cfg/models/rt-detr/rtdetr-C2f-IEL.yaml

    使用[CVPR2025 HVI](https://github.com/Fediory/HVI-CIDNet)中的Intensity Enhancement Layer与CSP改进rtdetr中的BackBone.

50. ultralytics/cfg/models/rt-detr/rtdetr-overlock.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的overlock-backbone替换rtdetr-r18的backbone.

51. ultralytics/cfg/models/rt-detr/rtdetr-C2f-RCB.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的RepConvBlock与CSP改进rtdetr-r18的backbone.

52. ultralytics/cfg/models/rt-detr/rtdetr-C2f-LEGM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的LEGM与CSP改进rtdetr-r18的backbone.

53. ultralytics/cfg/models/rt-detr/rtdetr-C2f-FAT.yaml

    使用[ICLR2024-FTIC](https://github.com/qingshi9974/ICLR2024-FTIC)中的FATBlock与CSP改进rtdetr-r18的backbone.

54. ultralytics/cfg/models/rt-detr/rtdetr-C2f-MobileMamba.yaml

    使用使用[CVPR2025 MobileMamba](https://github.com/lewandofskee/MobileMamba)中的MobileMambaBlock与CSP思想改进backbone.

55. ultralytics/cfg/models/rt-detr/rtdetr-MobileMamba.yaml

    使用[CVPR2025 MobileMamba](https://github.com/lewandofskee/MobileMamba)中的MobileMamba改进Backbone.

56. ultralytics/cfg/models/rt-detr/rtdetr-C2f-LFEM.yaml

    使用[LEGNet](https://github.com/lwCVer/LEGNet)中的LFEModule与CSP思想改进backbone.

57. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SBSM.yaml

    使用[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Snake Bi-Directional Sequence Modelling (SBSM)与CSP思想改进backbone.

58. ultralytics/cfg/models/rt-detr/rtdetr-lsnet.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)的LSNet替换backbone.

59. ultralytics/cfg/models/rt-detr/rtdetr-C2f-LSBlock.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)的LSBlock改进C2f.

60. ultralytics/cfg/models/rt-detr/rtdetr-C2f-TransMamba.yaml

    使用[TransMamba](https://github.com/sunshangquan/TransMamba)的TransMamba与CSP思想改进backbone.

61. ultralytics/cfg/models/rt-detr/rtdetr-C2f-EVS.yaml 

    使用[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EVS与CSP思想改进backbone.

62. ultralytics/cfg/models/rt-detr/rtdetr-C2f-EBlock.yaml

    使用[CVPR2025 DarkIR](https://github.com/cidautai/DarkIR)中的EVS与CSP思想改进backbone.

63. ultralytics/cfg/models/rt-detr/rtdetr-C2f-DBlock.yaml

    使用[CVPR2025 DarkIR](https://github.com/cidautai/DarkIR)中的EVS与CSP思想改进backbone.

64. ultralytics/cfg/models/rt-detr/rtdetr-C2f-FDConv.yaml

    使用[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv与CSP思想改进BackBone.

65. ultralytics/cfg/models/rt-detr/rtdetr-C2f-DSAN.yaml

    使用[DSA: Deformable Spatial Attention](https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks)中的Deformable Spatial Attention Block与CSP改进BackBone.

66. ultralytics/cfg/models/rt-detr/rtdetr-C2f-DSA.yaml

    使用[DSA: Deformable Spatial Attention](https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks)中的Deformable Spatial Attention与CSP改进BackBone.

67. ultralytics/cfg/models/rt-detr/rtdetr-C2f-RMB.yaml

    使用[CVPR2025 MaIR](https://github.com/XLearning-SCU/2025-CVPR-MaIR)中的Residual Mamba Block与CSP思想改进BackBone.

68. ultralytics/cfg/models/rt-detr/rtdetr-C2f-SFSConv.yaml

    使用[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv改进C2f.

69. ultralytics/cfg/models/rt-detr/rtdetr-C2f-GroupMamba.yaml

    使用[CVPR2025 GroupMamba](https://github.com/Amshaker/GroupMamba)中的GroupMambaLayer与CSP思想改进Backbone.

70. ultralytics/cfg/models/rt-detr/rtdetr-C2f-GroupMambaBlock.yaml

    使用[CVPR2025 GroupMamba](https://github.com/Amshaker/GroupMamba)中的GroupMambaBlock与CSP思想改进Backbone.

71. ultralytics/cfg/models/rt-detr/rtdetr-C2f-MambaVision.yaml

    使用[CVPR2025 MambaVision](https://github.com/NVlabs/MambaVision)中的MambaVision与CSP思想改进Backbone.

72. ultralytics/cfg/models/rt-detr/rtdetr-FCM.yaml

    使用[AAAI2025 FBRT-YOLO](https://github.com/galaxy-oss/FCM)的模块改进rtdetr.

### AIFI系列
1. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-LPE.yaml

    使用LearnedPositionalEncoding改进AIFI中的位置编码生成.(详细介绍请看百度云视频-20231119更新说明)
2. ultralytics/cfg/models/rt-detr/rtdetr-CascadedGroupAttention.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention改进rtdetr中的AIFI.(详细请看百度云视频-rtdetr-CascadedGroupAttention说明)
3. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DAttention.yaml

    使用[Vision Transformer with Deformable Attention CVPR2022](https://github.com/LeapLabTHU/DAT)中的DAttention改进AIFI.
4. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-HiLo.yaml

    使用[LITv2](https://github.com/ziplab/LITv2)中具有提取高低频信息的高效注意力对AIFI进行二次改进.
5. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-EfficientAdditive.yaml

    使用[ICCV2023 SwiftFormer](https://github.com/Amshaker/SwiftFormer/tree/main)中的EfficientAdditiveAttention改进AIFI.

6. ultralytics/cfg/models/rt-detr/rtdetr-AIFIRepBN.yaml

    使用[ICML-2024 SLAB](https://github.com/xinghaochen/SLAB)中的RepBN改进AIFI.

7. ultralytics/cfg/models/rt-detr/rtdetr-AdditiveTokenMixer.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock改进AIFI.

8. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-MSMHSA.yaml

    使用[CMTFNet](https://github.com/DrWuHonglin/CMTFNet/tree/main)中的M2SA改进AIFI.

9. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DHSA.yaml

    使用[Histoformer ECCV2024](https://github.com/sunshangquan/Histoformer)中的Dynamic-range Histogram Self-Attention改进AIFI.

10. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DPB.yaml

    使用[CrossFormer](https://arxiv.org/pdf/2108.00154)中的DynamicPosBias-Attention改进AIFI.

11. ultralytics/cfg/models/rt-detr/rtdetr-DTAB.yaml

    使用[AAAI2025 TBSN](https://github.com/nagejacob/TBSN)中的DTAB替换AIFI.

12. ultralytics/cfg/models/rt-detr/rtdetr-ETB.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的entanglement transformer block替换AIFI.

13. ultralytics/cfg/models/rt-detr/rtdetr-FDT.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Full-domain Transformer替换AIFI.

14. ultralytics/cfg/models/rt-detr/rtdetr-Pola.yaml

    使用[ICLR2025 PolaFormer)](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention改进AIFI.

15. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-TSSA.yaml

    使用[Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention改进AIFI.

16. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-ASSA.yaml
    
    使用[CVPR2024 Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)中的Adaptive Sparse Self-Attention改进AIFI.

17. ultralytics/cfg/models/rt-detr/rtdetr-ASSR.yaml
     
    使用[CVPR2025 MambaIR](https://github.com/csguoh/MambaIR)中的Attentive State Space Group改进rtdetr.

18. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-SEFN.yaml

    使用[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)改进AIFI.

19. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DyT.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan改进AIFI.

20. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-SEFFN.yaml

    使用[TransMamba](https://github.com/sunshangquan/TransMamba)的SpectralEnhancedFFN改进AIFI.

21. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-EDFFN.yaml

    使用[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EDFFN改进AIFI.

### Neck系列
1. ultralytics/cfg/models/rt-detr/rtdetr-ASF.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion来改进rtdetr.
2. ultralytics/cfg/models/rt-detr/rtdetr-slimneck.yaml

    使用[SlimNeck](https://github.com/AlanLi1997/slim-neck-by-gsconv)中的VoVGSCSP\VoVGSCSPC和GSConv改进rtdetr中的CCFM.
3. ultralytics/cfg/models/rt-detr/rtdetr-SDI.yaml

    使用[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对CCFM中的feature fusion进行改进.
4. ultralytics/cfg/models/rt-detr/rtdetr-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.
5. ultralytics/cfg/models/rt-detr/rtdetr-HSFPN.yaml

    使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进RTDETR中的CCFM.
6. ultralytics/cfg/models/rt-detr/rtdetr-bifpn.yaml

    添加BIFPN到rtdetr-r18中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        block模块选择,具体可看对应百度云视频-20240302更新公告.
    3. head_channel  
        BIFPN中的通道数,默认设置为256.
7. ultralytics/cfg/models/rt-detr/rtdetr-CSFCN.yaml

    使用[Context and Spatial Feature Calibration for Real-Time Semantic Segmentation](https://github.com/kaigelee/CSFCN/tree/main)中的Context and Spatial Feature Calibration模块改进rtdetr-neck.
8. ultralytics/cfg/models/rt-detr/rtdetr-CGAFusion.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的content-guided attention fusion改进rtdetr-neck.
9. ultralytics/cfg/models/rt-detr/rtdetr-SDFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的superficial detail fusion module改进rtdetr-neck.

10. ultralytics/cfg/models/rt-detr/rtdetr-PSFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的profound semantic fusion module改进yolov8-neck.

11. ultralytics/cfg/models/rt-detr/rtdetr-GLSA.yaml

    使用[GLSA](https://github.com/Barrett-python/DuAT)模块改进rtdetr的neck.

12. ultralytics/cfg/models/rt-detr/rtdetr-CTrans.yaml

    使用[[AAAI2022] UCTransNet](https://github.com/McGregorWwww/UCTransNet/tree/main)中的ChannelTransformer改进rtdetr-neck.

13. ultralytics/cfg/models/rt-detr/rtdetr-p6-CTrans.yaml

    使用[[AAAI2022] UCTransNet](https://github.com/McGregorWwww/UCTransNet/tree/main)中的ChannelTransformer改进rtdetr-neck.(带有p6版本)

14. ultralytics/cfg/models/rt-detr/rtdetr-MAFPN.yaml

    使用[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN改进Neck.

15. Cross-Layer Feature Pyramid Transformer.   

    P345:ultralytics/cfg/models/rt-detr/rtdetr-CFPT.yaml
    P3456:ultralytics/cfg/models/rt-detr/rtdetr-CFPT-P3456.yaml
    使用[CFPT](https://github.com/duzw9311/CFPT/tree/main)改进neck.

16. ultralytics/cfg/models/rt-detr/rtdetr-FreqFFPN.yaml

    使用[FreqFusion TPAMI2024](https://github.com/Linwei-Chen/FreqFusion)中的FreqFusion改进Neck.(这个需要python3.10,不然最后保存模型会出错.)

17. ultralytics/cfg/models/rt-detr/rtdetr-msga.yaml

    使用[MSA^2 Net](https://github.com/xmindflow/MSA-2Net)中的Multi-Scale Adaptive Spatial Attention Gate改进rtdetr-neck.

18. ultralytics/cfg/models/rt-detr/rtdetr-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade改进rtdetr-neck.

19. ultralytics/cfg/models/rt-detr/rtdetr-mpcafsa.yaml

    使用[BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation](https://github.com/nkicsl/SF-UNet)的Frequency-Spatial Attention和Multi-scale Progressive Channel Attention改进rtdetr-neck.

20. ultralytics/cfg/models/rt-detr/rtdetr-fsa.yaml

    使用[BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation](https://github.com/nkicsl/SF-UNet)的Frequency-Spatial Attention改进rtdetr.

21. ultralytics/cfg/models/rt-detr/rtdetr-CAB.yaml

    使用[CVPR2025 HVI](https://github.com/Fediory/HVI-CIDNet)中的CAB改进rtdetr中的特征融合.

22. ultralytics/cfg/models/rt-detr/rtdetr-MFM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM改进neck.

23. ultralytics/cfg/models/rt-detr/rtdetr-GDSAFusion.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的GDSAFusion改进Fusion.

### Head系列
1. ultralytics/cfg/models/rt-detr/rtdetr-p2.yaml

    添加小目标检测头P2到TransformerDecoderHead中.

### RepC3改进系列
1. ultralytics/cfg/models/rt-detr/rtdetr-DWRC3.yaml

    使用[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块构建DWRC3改进rtdetr.
2. ultralytics/cfg/models/rt-detr/rtdetr-Conv3XCC3.yaml

    使用[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC改进RepC3.
3. ultralytics/cfg/models/rt-detr/rtdetr-DRBC3.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进RepC3.
4. ultralytics/cfg/models/rt-detr/rtdetr-DBBC3.yaml

    使用[DiverseBranchBlock CVPR2021](https://github.com/DingXiaoH/DiverseBranchBlock)改进RepC3.
5. ultralytics/cfg/models/rt-detr/rtdetr-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进rtdetr-r18.
6. ultralytics/cfg/models/rt-detr/rtdetr-DGCST2.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer与Dynamic Group Convolution Shuffle Module进行结合改进rtdetr-r18.
7. ultralytics/cfg/models/rt-detr/rtdetr-RetBlockC3.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进RepC3.
8. ultralytics/cfg/models/rt-detr/rtdetr-KANC3.yaml

    使用[Pytorch-Conv-KAN](https://github.com/IvanDrokin/torch-conv-kan)的KAN卷积算子改进RepC3.
    目前支持:
    1. FastKANConv2DLayer
    2. KANConv2DLayer
    3. KALNConv2DLayer
    4. KACNConv2DLayer
    5. KAGNConv2DLayer
9. ultralytics/cfg/models/rt-detr/rtdetr-gConvC3.yaml

    使用[Rethinking Performance Gains in Image Dehazing Networks](https://arxiv.org/abs/2209.11448)的gConvblock改进RepC3.

10. ultralytics/cfg/models/rt-detr/rtdetr-LFEC3.yaml

    使用[Efficient Long-Range Attention Network for Image Super-resolution ECCV2022](https://github.com/xindongzhang/ELAN)中的Local feature extraction改进RepC3.

11. ultralytics/cfg/models/rt-detr/rtdetr-IELC3.yaml

    使用[CVPR2025 HVI](https://github.com/Fediory/HVI-CIDNet)中的Intensity Enhancement Layer改进rtdetr中的RepC3.

12. ultralytics/cfg/models/rt-detr/rtdetr-FDConvC3.yaml

    使用[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv改进RepC3.

### ResNet主干中的BasicBlock/BottleNeck改进系列(以下改进BottleNeck基本都有,就不再重复标注)
1. ultralytics/cfg/models/rt-detr/rtdetr-Ortho.yaml

    使用[OrthoNets](https://github.com/hady1011/OrthoNets/tree/main)中的正交通道注意力改进resnet18-backbone中的BasicBlock.(详细介绍请看百度云视频-20231119更新说明)
2. ultralytics/cfg/models/rt-detr/rtdetr-DCNV2.yaml

    使用可变形卷积DCNV2改进resnet18-backbone中的BasicBlock.
3. ultralytics/cfg/models/rt-detr/rtdetr-DCNV3.yaml

    使用可变形卷积[DCNV3 CVPR2023](https://github.com/OpenGVLab/InternImage)改进resnet18-backbone中的BasicBlock.(安装教程请看百度云视频-20231119更新说明)
4. ultralytics/cfg/models/rt-detr/rtdetr-iRMB.yaml

    使用[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB改进resnet18-backbone中的BasicBlock.(详细介绍请看百度云视频-20231119更新说明)
5. ultralytics/cfg/models/rt-detr/rtdetr-DySnake.yaml

    添加[DySnakeConv](https://github.com/YaoleiQi/DSCNet)到resnet18-backbone中的BasicBlock中.
6. ultralytics/cfg/models/rt-detr/rtdetr-PConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv改进resnet18-backbone中的BasicBlock.
7. ultralytics/cfg/models/rt-detr/rtdetr-Faster.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block改进resnet18-backbone中的BasicBlock.
8. ultralytics/cfg/models/rt-detr/rtdetr-AKConv.yaml

    使用[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进resnet18-backbone中的BasicBlock.

9. ultralytics/cfg/models/rt-detr/rtdetr-RFAConv.yaml

    使用[RFAConv 2023](https://github.com/Liuchen1997/RFAConv)改进resnet18-backbone中的BasicBlock.

10. ultralytics/cfg/models/rt-detr/rtdetr-RFCAConv.yaml

    使用[RFCAConv 2023](https://github.com/Liuchen1997/RFAConv)改进resnet18-backbone中的BasicBlock.

11. ultralytics/cfg/models/rt-detr/rtdetr-RFCBAMConv.yaml

    使用[RFCBAMConv 2023](https://github.com/Liuchen1997/RFAConv)改进resnet18-backbone中的BasicBlock.
12. ultralytics/cfg/models/rt-detr/rtdetr-Conv3XC.yaml

    使用[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC改进resnet18-backbone中的BasicBlock.
13. ultralytics/cfg/models/rt-detr/rtdetr-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进resnet18-backbone中的BasicBlock.
14. ultralytics/cfg/models/rt-detr/rtdetr-DBB.yaml

    使用[DiverseBranchBlock CVPR2021](https://github.com/DingXiaoH/DiverseBranchBlock)改进resnet18-backbone中的BasicBlock.
15. ultralytics/cfg/models/rt-detr/rtdetr-DualConv.yaml

    使用[DualConv](https://github.com/ChipsGuardian/DualConv)改进resnet18-backbone中的BasicBlock.
16. ultralytics/cfg/models/rt-detr/rtdetr-AggregatedAtt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进resnet18中的BasicBlock.(百度云视频-20240106更新说明)
17. ultralytics/cfg/models/rt-detr/rtdetr-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进resnet18中的BasicBlock.
18. ultralytics/cfg/models/rt-detr/rtdetr-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进resnet18中的BasicBlock.
19. ultralytics/cfg/models/rt-detr/rtdetr-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)改进resnet18-backbone中的BasicBlock.
20. ultralytics/cfg/models/rt-detr/rtdetr-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided和Light-weight Context Guided DownSample改进rtdetr-r18.
21. ultralytics/cfg/models/rt-detr/rtdetr-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进resnet18-basicblock.
22. ultralytics/cfg/models/rt-detr/rtdetr-Star.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock改进resnet18-basicblock.
23. ultralytics/cfg/models/rt-detr/rtdetr-KAN.yaml

    使用[Pytorch-Conv-KAN](https://github.com/IvanDrokin/torch-conv-kan)的KAN卷积算子改进resnet18-basicblock.
    目前支持:
    1. FastKANConv2DLayer
    2. KANConv2DLayer
    3. KALNConv2DLayer
    4. KACNConv2DLayer
    5. KAGNConv2DLayer
24. ultralytics/cfg/models/rt-detr/rtdetr-DEConv.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的detail-enhanced convolution改进resnet18-basicblock.
    关于DEConv在运行的时候重参数化后比重参数化前的计算量还要大的问题:是因为重参数化前thop库其计算不准的问题,看重参数化后的参数即可.

25. ultralytics/cfg/models/rt-detr/rtdetr-WTConv.yaml

    使用[ECCV2024 Wavelet Convolutions for Large Receptive Fields](https://github.com/BGU-CS-VIL/WTConv)中的WTConv改进BasicBlock.

26. ultralytics/cfg/models/rt-detr/rtdetr-WDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的WDBB改进BasicBlock.

27. ultralytics/cfg/models/rt-detr/rtdetr-DeepDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的DeepDBB改进BasicBlock.

### 上下采样算子系列
1. ultralytics/cfg/models/rt-detr/rtdetr-DySample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进CCFM中的上采样.
2. ultralytics/cfg/models/rt-detr/rtdetr-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进CCFM中的上采样.
3. ultralytics/cfg/models/rt-detr/rtdetr-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进CCFM的下采样.
4. ultralytics/cfg/models/rt-detr/rtdetr-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample改进rtdetr-r18.
5. ultralytics/cfg/models/rt-detr/rtdetr-SRFD.yaml

    使用[A Robust Feature Downsampling Module for Remote Sensing Visual Tasks](https://ieeexplore.ieee.org/document/10142024)改进rtdetr的下采样.

6. ultralytics/cfg/models/rt-detr/rtdetr-WaveletPool.yaml

    使用[Wavelet Pooling](https://openreview.net/forum?id=rkhlb8lCZ)改进RTDETR的上采样和下采样。

7. ultralytics/cfg/models/rt-detr/rtdetr-LDConv.yaml

    使用[LDConv](https://github.com/CV-ZhangXin/LDConv/tree/main)改进下采样.

8. ultralytics/cfg/models/rt-detr/rtdetr-PSConv.yaml

    使用[AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://github.com/JN-Yang/PConv-SDloss-Data)中的Pinwheel-shaped Convolution改进rtdetr.

9. ultralytics/cfg/models/rt-detr/rtdetr-EUCB.yaml

    使用[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)中的EUCB改进rtdetr-r18的上采样.

10. ultralytics/cfg/models/rt-detr/rtdetr-LoGStem.yaml

    使用[LEGNet](https://github.com/lwCVer/LEGNet)中的LoGStem改进Stem.

### RT-DETR-L改进系列
1. ultralytics/cfg/models/rt-detr/rtdetr-l-GhostHGNetV2.yaml

    使用GhostConv改进HGNetV2.(详细介绍请看百度云视频-20231109更新说明)

2. ultralytics/cfg/models/rt-detr/rtdetr-l-RepHGNetV2.yaml

    使用RepConv改进HGNetV2.(详细介绍请看百度云视频-20231109更新说明)

3. ultralytics/cfg/models/rt-detr/rtdetr-l-attention.yaml

    添加注意力模块到HGBlock中.(手把手教程请看百度云视频-手把手添加注意力教程)

### RT-DETR-Mamba
    集成Mamba-YOLO,并把head改为RTDETR-Head.(需要编译，请看百度云视频)
    ultralytics/cfg/models/rt-detr/rtdetr-mamba-T.yaml
    ultralytics/cfg/models/rt-detr/rtdetr-mamba-B.yaml
    ultralytics/cfg/models/rt-detr/rtdetr-mamba-L.yaml

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
21. HiLo
22. LocalWindowAttention(EfficientViT中的CascadedGroupAttention注意力)
23. Efficient Local Attention
24. CAA(CVPR2024 PKINet中的注意力)
25. CAFM

### IoU系列
1. IoU,GIoU,DIoU,CIoU,EIoU,SIoU(百度云视频-20231125更新说明)
2. MPDIoU[论文链接](https://arxiv.org/pdf/2307.07662.pdf)(百度云视频-20231125更新说明)
3. Inner-IoU,Inner-GIoU,Inner-DIoU,Inner-CIoU,Inner-EIoU,Inner-SIoU[论文链接](https://arxiv.org/abs/2311.02877)(百度云视频-20231125更新说明)
4. Inner-MPDIoU(利用Inner-Iou与MPDIou进行二次创新)(百度云视频-20231125更新说明)
5. Normalized Gaussian Wasserstein Distance.[论文链接](https://arxiv.org/abs/2110.13389)(百度云视频-20231125更新说明)
6. Shape-IoU,Inner-Shape-IoU[论文链接](https://arxiv.org/abs/2110.13389)(百度云视频-20240106更新说明)
7. SlideLoss,EMASlideLoss[创新思路](https://www.bilibili.com/video/BV1W14y1i79U/?vd_source=c8452371e7ca510979593165c8d7ac27).[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)(百度云视频-20240113更新说明)
8. Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)(百度云视频-20240113更新说明)
9. Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)(百度云视频-20240113更新说明)
10. Focaler-IoU,Focaler-GIoU,Focaler-DIoU,Focaler-CIoU,Focaler-EIoU,Focaler-SIoU,Focaler-Shape-IoU,Focaler-MPDIoU[论文链接](https://arxiv.org/abs/2401.10525)(百度云视频-20240128更新说明)
11. Focaler-Wise-IoU(v1,v2,v3)(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)[论文链接](https://arxiv.org/abs/2401.10525)(百度云视频-20240128更新说明)
12. Powerful-IoU,Powerful-IoUV2,Inner-Powerful-IoU,Inner-Powerful-IoUV2,Focaler-Powerful-IoU,Focaler-Powerful-IoUV2,Wise-Powerful-IoU(v1,v2,v3),Wise-Powerful-IoUV2(v1,v2,v3)[论文链接](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640)
13. SlideVarifocalLoss,EMASlideVarifocalLoss[创新思路](https://www.bilibili.com/video/BV1W14y1i79U/?vd_source=c8452371e7ca510979593165c8d7ac27).[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)(百度云视频-20240302更新说明)
14. CVPR2025-DEIM-MAL.(百度云视频-20240315更新说明)

### 以Yolov8为基准模型的改进方案
1. ultralytics/cfg/models/yolo-detr/yolov8-detr.yaml

    使用RT-DETR中的TransformerDecoderHead改进yolov8.

2. ultralytics/cfg/models/yolo-detr/yolov8-detr-DWR.yaml

    使用RT-DETR中的TransformerDecoderHead和[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块改进yolov8.

3. ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)改进yolov8.(支持替换其他主干,请看百度云视频-替换主干示例教程)

4. ultralytics/cfg/models/yolo-detr/yolov8-detr-AIFI-LPE.yaml

    使用RT-DETR中的TransformerDecoderHead和LearnedPositionalEncoding改进yolov8.(详细介绍请看百度云视频-20231119更新说明)

5. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DCNV2.yaml

    使用RT-DETR中的TransformerDecoderHead和可变形卷积DCNV2改进yolov8.

6. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DCNV3.yaml

    使用RT-DETR中的TransformerDecoderHead和可变形卷积[DCNV3 CVPR2023](https://github.com/OpenGVLab/InternImage)改进yolov8.(安装教程请看百度云视频-20231119更新说明)

7. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DCNV2-Dynamic.yaml

    使用RT-DETR中的TransformerDecoderHead和自研可变形卷积DCNV2-Dynamic改进yolov8.(详细介绍请看百度云视频-MPCA与DCNV2_Dynamic的说明)

8. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Ortho.yaml

    使用RT-DETR中的TransformerDecoderHead和[OrthoNets](https://github.com/hady1011/OrthoNets/tree/main)中的正交通道注意力改进yolov8.(详细介绍请看百度云视频-20231119更新说明)

9. ultralytics/cfg/models/yolo-detr/yolov8-detr-attention.yaml

    添加注意力到基于RTDETR-Head中的yolov8中.(手把手教程请看百度云视频-手把手添加注意力教程)

10. ultralytics/cfg/models/yolo-detr/yolov8-detr-p2.yaml

    添加小目标检测头P2到TransformerDecoderHead中.

11. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DySnake.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C2f融合.  

12. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block改进yolov8.

13. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster-Rep.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv二次创新后的Faster-Block-Rep改进yolov8.

14. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster-EMA.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)二次创新后的Faster-Block-EMA的Faster-Block-EMA改进yolov8.

15. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster-Rep-EMA.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv、[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)二次创新后的Faster-Block改进yolov8.

16. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-AKConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进yolov8.

17. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFAConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov8.

18. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFCAConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov8.

19. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFCBAMConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov8.

20. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Conv3XC.yaml

    使用RT-DETR中的TransformerDecoderHead和[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC改进yolov8.

21. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-SPAB.yaml

    使用RT-DETR中的TransformerDecoderHead和[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的SPAB改进yolov8.

22. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DRB.yaml

    使用RT-DETR中的TransformerDecoderHead和[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进yolov8.

23. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-UniRepLKNetBlock.yaml

    使用RT-DETR中的TransformerDecoderHead和[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的UniRepLKNetBlock改进yolov8.

24. ultralytics/cfg/models/yolo-detr/yolov8-detr-DWR-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)进行二次创新改进yolov8.

25. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DBB.yaml

    使用RT-DETR中的TransformerDecoderHead和[DiverseBranchBlock CVPR2021](https://github.com/DingXiaoH/DiverseBranchBlock)改进yolov8.

26. ultralytics/cfg/models/yolo-detr/yolov8-detr-CSP-EDLAN.yaml

    使用RT-DETR中的TransformerDecoderHead和[DualConv](https://github.com/ChipsGuardian/DualConv)打造CSP Efficient Dual Layer Aggregation Networks改进yolov8.

27. ultralytics/cfg/models/yolo-detr/yolov8-detr-ASF.yaml

    使用RT-DETR中的TransformerDecoderHead和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolov8.

28. ultralytics/cfg/models/yolo-detr/yolov8-detr-ASF-P2.yaml

    在ultralytics/cfg/models/yolo-detr/yolov8-detr-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.

29. ultralytics/cfg/models/yolo-detr/yolov8-detr-slimneck.yaml

    使用RT-DETR中的TransformerDecoderHead和[SlimNeck](https://github.com/AlanLi1997/slim-neck-by-gsconv)中VoVGSCSP\VoVGSCSPC和GSConv改进yolov8的neck.

30. ultralytics/cfg/models/yolo-detr/yolov8-detr-slimneck-asf.yaml

    在ultralytics/cfg/models/yolo-detr/yolov8-detr-slimneck.yaml使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新.

31. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-AggregatedAtt.yaml

    使用RT-DETR中的TransformerDecoderHead和[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进C2f.(百度云视频-20240106更新说明)

32. ultralytics/cfg/models/yolo-detr/yolov8-detr-SDI.yaml

    使用RT-DETR中的TransformerDecoderHead和[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对yolov8中的feature fusion进行改进.

33. ultralytics/cfg/models/yolo-detr/yolov8-detr-goldyolo.yaml

    利用RT-DETR中的TransformerDecoderHead和华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.

34. ultralytics/cfg/models/yolo-detr/yolov8-detr-goldyolo-asf.yaml

    利用RT-DETR中的TransformerDecoderHead和华为2023最新GOLD-YOLO中的Gatherand-Distribute和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行改进特征融合模块.

35. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进C2f.

36. ultralytics/cfg/models/yolo-detr/yolov8-detr-HSFPN.yaml

    利用RT-DETR中的TransformerDecoderHead和使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进YOLOV8中的PAN.

37. ultralytics/cfg/models/yolo-detr/yolov8-detr-HSPAN.yaml

    利用RT-DETR中的TransformerDecoderHead和对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进YOLOV8中的PAN.

38. ultralytics/cfg/models/yolo-detr/yolov8-detr-Dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolov8-detr neck中的上采样.

39. ultralytics/cfg/models/yolo-detr/yolov8-detr-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolov8-detr neck中的上采样.

40. ultralytics/cfg/models/yolo-detr/yolov8-detr-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolov8-detr neck的下采样.

41. ultralytics/cfg/models/yolo-detr/yolov8-detr-ASF-Dynamic.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion的上采样模块得到Dynamic Sample Attentional Scale Sequence Fusion改进yolov8-detr中的neck.

42. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进yolov8-detr中的C2f.

43. ultralytics/cfg/models/yolo-detr/yolov8-detr-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进yolov8-detr中的C2f.

44. ultralytics/cfg/models/yolo-detr/yolov8-detr-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进yolov8-detr中的C2f.

45. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)对C2f中的BottleNeck进行改进,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

46. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-LVMB.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)与Cross Stage Partial进行结合,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

47. ultralytics/cfg/models/yolo-detr/yolov8-detr-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行改进yolov8-detr.

48. ultralytics/cfg/models/yolo-detr/yolov8-detr-bifpn.yaml

    添加BIFPN到yolov8中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        block模块选择,具体可看对应百度云视频-20240302更新公告.
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

49. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided和Light-weight Context Guided DownSample改进yolov8-detr.

50. ultralytics/cfg/models/yolo-detr/yolov8-detr-PACAPN.yaml

    自研结构, Parallel Atrous Convolution Attention Pyramid Network, PAC-APN

51. ultralytics/cfg/models/yolo-detr/yolov8-detr-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进yolov8-detr.

52. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-RetBlock.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进C2f.

53. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块改进C2f.

54. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进C2f.

55. ultralytics/cfg/models/yolo-detr/yolov8-detr-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.

56. ultralytics/cfg/models/yolo-detr/yolov8-detr-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.

57. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C2f.

58. ultralytics/cfg/models/yolo-detr/yolov8-detr-SRFD.yaml

    使用[A Robust Feature Downsampling Module for Remote Sensing Visual Tasks](https://ieeexplore.ieee.org/document/10142024)改进yolov8的下采样.

59. ultralytics/cfg/models/yolo-detr/yolov8-detr-CSFCN.yaml

    使用[Context and Spatial Feature Calibration for Real-Time Semantic Segmentation](https://github.com/kaigelee/CSFCN/tree/main)中的Context and Spatial Feature Calibration模块改进yolov8.

60. ultralytics/cfg/models/yolo-detr/yolov8-detr-CGAFusion.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的content-guided attention fusion改进yolov8-neck.

61. ultralytics/cfg/models/yolo-detr/yolov8-detr-CAFMFusion.yaml

    利用具有[HCANet](https://github.com/summitgao/HCANet)中的CAFM，其具有获取全局和局部信息的注意力机制进行二次改进content-guided attention fusion.
 
62. ultralytics/cfg/models/yolo-detr/yolov8-detr-RGCSPELAN.yaml

    自研RepGhostCSPELAN.
    1. 参考GhostNet中的思想(主流CNN计算的中间特征映射存在广泛的冗余)，采用廉价的操作生成一部分冗余特征图，以此来降低计算量和参数量。
    2. 舍弃yolov5与yolov8中常用的BottleNeck，为了弥补舍弃残差块所带来的性能损失，在梯度流通分支上使用RepConv，以此来增强特征提取和梯度流通的能力，并且RepConv可以在推理的时候进行融合，一举两得。
    3. 可以通过缩放因子控制RGCSPELAN的大小，使其可以兼顾小模型和大模型。

63. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster-CGLU.yaml

    使用[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU对CVPR2023中的FasterNet进行二次创新.

64. ultralytics/cfg/models/yolo-detr/yolov8-detr-SDFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的superficial detail fusion module改进yolov8-neck.

65. ultralytics/cfg/models/yolo-detr/yolov8-detr-PSFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的profound semantic fusion module改进yolov8-neck.

66. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Star.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock改进C2f.

67. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Star-CAA.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock和[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA改进C2f.

68. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-KAN.yaml

    使用[Pytorch-Conv-KAN](https://github.com/IvanDrokin/torch-conv-kan)的KAN卷积算子改进C2f.
    目前支持:
    1. FastKANConv2DLayer
    2. KANConv2DLayer
    3. KALNConv2DLayer
    4. KACNConv2DLayer
    5. KAGNConv2DLayer

69. ultralytics/cfg/models/yolo-detr/yolov8-detr-ContextGuideFPN.yaml

    Context Guide Fusion Module（CGFM）是一个创新的特征融合模块，旨在改进YOLOv8中的特征金字塔网络（FPN）。该模块的设计考虑了多尺度特征融合过程中上下文信息的引导和自适应调整。
    1. 上下文信息的有效融合：通过SE注意力机制，模块能够在特征融合过程中捕捉并利用重要的上下文信息，从而增强特征表示的有效性，并有效引导模型学习检测目标的信息，从而提高模型的检测精度。
    2. 特征增强：通过权重化的特征重组操作，模块能够增强重要特征，同时抑制不重要特征，提升特征图的判别能力。
    3. 简单高效：模块结构相对简单，不会引入过多的计算开销，适合在实时目标检测任务中应用。
    这期视频讲解在B站:https://www.bilibili.com/video/BV1Vx4y1n7hZ/

70. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DEConv.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的detail-enhanced convolution改进C2f.
    关于DEConv在运行的时候重参数化后比重参数化前的计算量还要大的问题:是因为重参数化前thop库其计算不准的问题,看重参数化后的参数即可.

71. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-SMPCGLU.yaml

    Self-moving Point Convolutional GLU模型改进C2f.
    SMP来源于[CVPR2023-SMPConv](https://github.com/sangnekim/SMPConv),Convolutional GLU来源于[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt).
    1. 普通的卷积在面对数据中的多样性和复杂性时，可能无法捕捉到有效的特征，因此我们采用了SMPConv，其具备最新的自适应点移动机制，从而更好地捕捉局部特征，提高特征提取的灵活性和准确性。
    2. 在SMPConv后添加CGLU，Convolutional GLU 结合了卷积和门控机制，能够选择性地通过信息通道，提高了特征提取的有效性和灵活性。

### 以Yolov5为基准模型的改进方案
1. ultralytics/cfg/models/yolo-detr/yolov5-detr.yaml

    使用RT-DETR中的TransformerDecoderHead改进yolov5.

2. ultralytics/cfg/models/yolo-detr/yolov5-detr-DWR.yaml

    使用RT-DETR中的TransformerDecoderHead和[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块改进yolov5.

3. ultralytics/cfg/models/yolo-detr/yolov5-detr-fasternet.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)改进yolov5.(支持替换其他主干,请看百度云视频-替换主干示例教程)

4. ultralytics/cfg/models/yolo-detr/yolov5-detr-AIFI-LPE.yaml

    使用RT-DETR中的TransformerDecoderHead和LearnedPositionalEncoding改进yolov5.(详细介绍请看百度云视频-20231119更新说明)

5. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV2.yaml

    使用RT-DETR中的TransformerDecoderHead和可变形卷积DCNV2改进yolov5.

6. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV3.yaml

    使用RT-DETR中的TransformerDecoderHead和可变形卷积[DCNV3 CVPR2023](https://github.com/OpenGVLab/InternImage)改进yolov5.(安装教程请看百度云视频-20231119更新说明)

7. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV2-Dynamic.yaml

    使用RT-DETR中的TransformerDecoderHead和自研可变形卷积DCNV2-Dynamic改进yolov5.(详细介绍请看百度云视频-MPCA与DCNV2_Dynamic的说明)

8. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Ortho.yaml(详细介绍请看百度云视频-20231119更新说明)

    使用RT-DETR中的TransformerDecoderHead和[OrthoNets](https://github.com/hady1011/OrthoNets/tree/main)中的正交通道注意力改进yolov5.

9. ultralytics/cfg/models/yolo-detr/yolov5-detr-attention.yaml

    添加注意力到基于RTDETR-Head中的yolov5中.(手把手教程请看百度云视频-手把手添加注意力教程)

10. ultralytics/cfg/models/yolo-detr/yolov5-detr-p2.yaml

    添加小目标检测头P2到TransformerDecoderHead中.

11. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DySnake.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C3融合.  

12. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block改进yolov5.

13. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster-Rep.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv二次创新后的Faster-Block-Rep改进yolov5.

14. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster-EMA.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)二次创新后的Faster-Block-EMA的Faster-Block-EMA改进yolov5.

15. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster-Rep-EMA.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv、[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)二次创新后的Faster-Block改进yolov5.

16. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-AKConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进yolov5.

17. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFAConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov5.

18. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFCAConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov5.

19. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFCBAMConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov5.

20. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Conv3XC.yaml

    使用RT-DETR中的TransformerDecoderHead和[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC改进yolov5.

21. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-SPAB.yaml

    使用RT-DETR中的TransformerDecoderHead和[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的SPAB改进yolov5.

22. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DRB.yaml

    使用RT-DETR中的TransformerDecoderHead和[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进改进yolov5.

23. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-UniRepLKNetBlock.yaml

    使用RT-DETR中的TransformerDecoderHead和[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的UniRepLKNetBlock改进改进yolov5.

24. ultralytics/cfg/models/yolo-detr/yolov5-detr-DWR-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)进行二次创新改进yolov5.

25. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DBB.yaml

    使用RT-DETR中的TransformerDecoderHead和[DiverseBranchBlock CVPR2021](https://github.com/DingXiaoH/DiverseBranchBlock)改进yolov5.

26. ultralytics/cfg/models/yolo-detr/yolov5-detr-CSP-EDLAN.yaml

    使用RT-DETR中的TransformerDecoderHead和[DualConv](https://github.com/ChipsGuardian/DualConv)打造CSP Efficient Dual Layer Aggregation Networks改进yolov5.

27. ultralytics/cfg/models/yolo-detr/yolov5-detr-ASF.yaml

    使用RT-DETR中的TransformerDecoderHead和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolov5.

28. ultralytics/cfg/models/yolo-detr/yolov5-detr-ASF-P2.yaml

    在ultralytics/cfg/models/yolo-detr/yolov5-detr-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.

29. ultralytics/cfg/models/yolo-detr/yolov5-detr-slimneck.yaml

    使用RT-DETR中的TransformerDecoderHead和[SlimNeck](https://github.com/AlanLi1997/slim-neck-by-gsconv)中VoVGSCSP\VoVGSCSPC和GSConv改进yolov5的neck.

30. ultralytics/cfg/models/yolo-detr/yolov5-detr-slimneck-asf.yaml

    在ultralytics/cfg/models/yolo-detr/yolov5-detr-slimneck.yaml使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新.

31. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-AggregatedAtt.yaml

    使用RT-DETR中的TransformerDecoderHead和[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进C3.(百度云视频-20240106更新说明)

32. ultralytics/cfg/models/yolo-detr/yolov5-detr-SDI.yaml

    使用RT-DETR中的TransformerDecoderHead和[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对yolov5中的feature fusion进行改进.

33. ultralytics/cfg/models/yolo-detr/yolov5-detr-goldyolo.yaml

    利用RT-DETR中的TransformerDecoderHead和华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.

34. ultralytics/cfg/models/yolo-detr/yolov5-detr-goldyolo-asf.yaml

    利用RT-DETR中的TransformerDecoderHead和华为2023最新GOLD-YOLO中的Gatherand-Distribute和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行改进特征融合模块.

35. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进C3.

36. ultralytics/cfg/models/yolo-detr/yolov5-detr-HSFPN.yaml

    利用RT-DETR中的TransformerDecoderHead和使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进YOLOV5中的PAN.

37. ultralytics/cfg/models/yolo-detr/yolov5-detr-HSPAN.yaml

    利用RT-DETR中的TransformerDecoderHead和对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进YOLOV5中的PAN.

38. ultralytics/cfg/models/yolo-detr/yolov8-detr-Dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolov8-detr neck中的上采样.

39. ultralytics/cfg/models/yolo-detr/yolov8-detr-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolov8-detr neck中的上采样.

40. ultralytics/cfg/models/yolo-detr/yolov8-detr-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolov8-detr neck的下采样.

41. ultralytics/cfg/models/yolo-detr/yolov5-detr-ASF-Dynamic.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion的上采样模块得到Dynamic Sample Attentional Scale Sequence Fusion改进yolov5-detr中的neck.

42. ultralytics/cfg/models/yolo-detr/yolov5-detr-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进yolov5-detr中的C3.

43. ultralytics/cfg/models/yolo-detr/yolov5-detr-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进yolov5-detr中的C2f.

44. ultralytics/cfg/models/yolo-detr/yolov5-detr-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进yolov5-detr中的C2f.

45. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)对C3中的BottleNeck进行改进,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

46. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-LVMB.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)与Cross Stage Partial进行结合,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

47. ultralytics/cfg/models/yolo-detr/yolov5-detr-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行改进yolov5-detr.

48. ultralytics/cfg/models/yolo-detr/yolov5-detr-bifpn.yaml

    添加BIFPN到yolov8中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        block模块选择,具体可看对应百度云视频-20240302更新公告.
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

49. ultralytics/cfg/models/yolo-detr/yolov5-detr-C2f-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided和Light-weight Context Guided DownSample改进yolov5-detr.

50. ultralytics/cfg/models/yolo-detr/yolov5-detr-PACAPN.yaml

    自研结构, Parallel Atrous Convolution Attention Pyramid Network, PAC-APN

51. ultralytics/cfg/models/yolo-detr/yolov5-detr-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进yolov5-detr.

52. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-RetBlock.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进C3.

53. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块改进C3.

54. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进C3.

55. ultralytics/cfg/models/yolo-detr/yolov5-detr-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.

56. ultralytics/cfg/models/yolo-detr/yolov5-detr-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.

57. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C3.

58. ultralytics/cfg/models/yolo-detr/yolov5-detr-SRFD.yaml

    使用[A Robust Feature Downsampling Module for Remote Sensing Visual Tasks](https://ieeexplore.ieee.org/document/10142024)改进yolov5的下采样.

59. ultralytics/cfg/models/yolo-detr/yolov5-detr-CSFCN.yaml

    使用[Context and Spatial Feature Calibration for Real-Time Semantic Segmentation](https://github.com/kaigelee/CSFCN/tree/main)中的Context and Spatial Feature Calibration模块改进yolov5.

60. ultralytics/cfg/models/yolo-detr/yolov5-detr-CGAFusion.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的content-guided attention fusion改进yolov5-neck.

61. ultralytics/cfg/models/yolo-detr/yolov5-detr-CAFMFusion.yaml

    利用具有[HCANet](https://github.com/summitgao/HCANet)中的CAFM，其具有获取全局和局部信息的注意力机制进行二次改进content-guided attention fusion.
 
62. ultralytics/cfg/models/yolo-detr/yolov5-detr-RGCSPELAN.yaml

    自研RepGhostCSPELAN.
    1. 参考GhostNet中的思想(主流CNN计算的中间特征映射存在广泛的冗余)，采用廉价的操作生成一部分冗余特征图，以此来降低计算量和参数量。
    2. 舍弃yolov5与yolov8中常用的BottleNeck，为了弥补舍弃残差块所带来的性能损失，在梯度流通分支上使用RepConv，以此来增强特征提取和梯度流通的能力，并且RepConv可以在推理的时候进行融合，一举两得。
    3. 可以通过缩放因子控制RGCSPELAN的大小，使其可以兼顾小模型和大模型。

63. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster-CGLU.yaml

    使用[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU对CVPR2023中的FasterNet进行二次创新.

64. ultralytics/cfg/models/yolo-detr/yolov5-detr-SDFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的superficial detail fusion module改进yolov5-neck.

65. ultralytics/cfg/models/yolo-detr/yolov5-detr-PSFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的profound semantic fusion module改进yolov5-neck.

66. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Star.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock改进C3.

67. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Star-CAA.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock和[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA改进C3.

68. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-KAN.yaml

    使用[Pytorch-Conv-KAN](https://github.com/IvanDrokin/torch-conv-kan)的KAN卷积算子改进C3.
    目前支持:
    1. FastKANConv2DLayer
    2. KANConv2DLayer
    3. KALNConv2DLayer
    4. KACNConv2DLayer
    5. KAGNConv2DLayer

69. ultralytics/cfg/models/yolo-detr/yolov5-detr-ContextGuideFPN.yaml

    Context Guide Fusion Module（CGFM）是一个创新的特征融合模块，旨在改进YOLOv8中的特征金字塔网络（FPN）。该模块的设计考虑了多尺度特征融合过程中上下文信息的引导和自适应调整。
    1. 上下文信息的有效融合：通过SE注意力机制，模块能够在特征融合过程中捕捉并利用重要的上下文信息，从而增强特征表示的有效性，并有效引导模型学习检测目标的信息，从而提高模型的检测精度。
    2. 特征增强：通过权重化的特征重组操作，模块能够增强重要特征，同时抑制不重要特征，提升特征图的判别能力。
    3. 简单高效：模块结构相对简单，不会引入过多的计算开销，适合在实时目标检测任务中应用。
    这期视频讲解在B站:https://www.bilibili.com/video/BV1Vx4y1n7hZ/

70. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DEConv.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的detail-enhanced convolution改进C3.
    关于DEConv在运行的时候重参数化后比重参数化前的计算量还要大的问题:是因为重参数化前thop库其计算不准的问题,看重参数化后的参数即可.

71. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-SMPCGLU.yaml

    Self-moving Point Convolutional GLU模型改进C3.
    SMP来源于[CVPR2023-SMPConv](https://github.com/sangnekim/SMPConv),Convolutional GLU来源于[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt).
    1. 普通的卷积在面对数据中的多样性和复杂性时，可能无法捕捉到有效的特征，因此我们采用了SMPConv，其具备最新的自适应点移动机制，从而更好地捕捉局部特征，提高特征提取的灵活性和准确性。
    2. 在SMPConv后添加CGLU，Convolutional GLU 结合了卷积和门控机制，能够选择性地通过信息通道，提高了特征提取的有效性和灵活性。

# 更新公告
- **20231105-rtdetr-v1.0**
    1. 初版项目发布.

- **20231109-rtdetr-v1.1**
    1. 修复断点训练不能正常使用的bug.
    2. 优化get_FPS.py中的模型导入方法.
    3. 增加以yolov5和yolov8为基准模型更换为RTDETR的Head,后续也会提供yolov5-detr,yolov8-detr相关的改进.
    4. 新增百度云视频-20231109更新说明视频和替换主干说明视频.
    5. 新增GhostHGNetV2,RepHGNetV2,详细请看使用教程中的RT-DETR改进方案.
    6. 新增使用DWRSeg中的Dilation-wise Residual(DWR)模块,加强从网络高层的可扩展感受野中提取特征,详细请看使用教程中的RT-DETR改进方案.

- **20231119-rtdetr-v1.2**
    1. 增加DCNV2,DCNV3,DCNV2-Dynamic,并以RTDETR-R18,RTDETR-R50,YOLOV5-Detr,YOLOV8-Detr多个基准模型进行改进,详细请看使用教程中的RT-DETR改进方案.
    2. 使用CVPR2022-OrthoNets中的正交通道注意力改进resnet18-backbone中的BasicBlock,resnet50-backbone中的BottleNeck,yolov8-C2f,yolov5-C3,详细请看使用教程中的RT-DETR改进方案.
    3. 使用LearnedPositionalEncoding改进AIFI中的位置编码信息生成,详细请看使用教程中的RT-DETR改进方案.
    4. 增加EMO模型中的iRMB模块,并使用(EfficientViT-CVPR2023)中的CascadedAttention对其二次创新得到iRMB_Cascaded,详细请看使用教程中的RT-DETR改进方案.
    5. 百度云视频增加1119更新说明和手把手添加注意力机制视频教学.
    6. 更新使用教程.

- **20231126-rtdetr-v1.3**
    1. 支持IoU,GIoU,DIoU,CIoU,EIoU,SIoU.
    2. 支持MPDIoU,Inner-IoU,Inner-MPDIoU.
    3. 支持Normalized Gaussian Wasserstein Distance.
    4. 支持小目标检测层P2.
    5. 支持DySnakeConv.
    6. 新增Pconv,PConv-Rep(二次创新)优化rtdetr-r18与rtdetr-r50.
    7. 新增Faster-Block,Faster-Block-Rep(二次创新),Faster-Block-EMA(二次创新),Faster-Block-Rep-EMA(二次创新)优化rtdetr-r18、rtdetr-r50、yolov5-detr、yolov8-retr.
    8. 更新使用教程.
    9. 百度云视频增加1126更新说明.

- **20231202-rtdetr-v1.4**
    1. 支持AKConv(具有任意采样形状和任意数目参数的卷积核).
    2. 支持RFAConv,RFCAConv,RFCBAMConv(感受野注意力卷积).
    3. 支持UniRepLKNet(大核CNNRepLK正统续作).
    4. 使用CVPR2022 DAttention改进AIFI.
    4. 更新使用教程.
    5. 百度云视频增加1202更新说明.
    6. 解决训练过程中由于指标出现的nan问题导致best.pt没办法正常保存.

- **20231210-rtdetr-v1.5**
    1. 支持来自Swift Parameter-free Attention Network中的重参数化Conv3XC模块.
    2. 支持UniRepLKNet中的DilatedReparamBlock.
    3. 支持UniRepLKNet中的DilatedReparamBlock对DWRSeg中的Dilation-wise Residual(DWR)模块进行二次创新的DWR_DRB.
    4. 使用ICCV2023 FLatten Transformer中的FocusedLinearAttention改进AIFI.
    5. 更新使用教程.
    6. 百度云视频增加1210更新说明.

- **20231214-rtdetr-v1.6**
    1. 支持DiverseBranchBlock.
    2. 利用DualConv打造CSP Efficient Dual Layer Aggregation Networks(仅支持yolov5-detr和yolov8-detr).
    3. 使用Swift Parameter-free Attention Network中的重参数化Conv3XC和DiverseBranchBlock改进RepC3.
    4. 支持最新的ASF-YOLO中的Attentional Scale Sequence Fusion.
    5. 更新使用教程.
    6. 百度云视频增加1214更新说明.

- **20231223-rtdetr-v1.7**
    1. 增加rtdetr-r18-asf-p2.yaml,使用ASF-YOLO中的Attentional Scale Sequence Fusion与Small Object Detection Head进行二次创新.
    2. 新增rtdetr-slimneck.yaml和rtdetr-slimneck-ASF.yaml.
    3. 新增yolov8-detr-slimneck.yaml,yolov8-detr-slimneck-asf.yaml.
    4. 新增yolov5-detr-slimneck.yaml,yolov5-detr-slimneck-asf.yaml.
    5. 修正热力图计算中预处理.
    6. 更新使用教程.
    7. 百度云视频增加1223更新说明.

- **20240106-rtdetr-v1.8**
    1. 新增Shape-IoU,Inner-Shape-IoU.
    2. 新增支持TransNeXt主干和TransNeXt中的聚焦感知注意力机制.
    3. 新增U-NetV2中的Semantics and Detail Infusion Module对RTDETR的CCFM进行创新.
    4. ASF系列支持attention_add.
    5. 更新使用教程.
    6. 百度云视频增加20240106更新说明.

- **20240113-rtdetr-v1.9**
    1. 支持Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
    2. 支持Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
    3. 支持SlideLoss,EMASlideLoss(利用Exponential Moving Average优化mean iou,可当自研创新模块).
    4. 使用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.
    5. 使用ASF-YOLO中Attentional Scale Sequence Fusion与GOLD-YOLO中的Gatherand-Distribute进行二次创新结合.
    6. 修正rtdetr-r34中检测头参数错误的问题,增加rtdetr-r34,rtdetr-r50-m的预训练权重.
    7. 更新使用教程.
    8. 百度云视频增加20240113更新说明.

- **20240120-rtdetr-v1.10**
    1. 新增DCNV4.
    2. 使用[LITv2](https://github.com/ziplab/LITv2)中具有提取高低频信息的高效注意力对AIFI进行二次改进.
    3. 使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进RTDETR中的CCFM和YOLOV5-DETR、YOLOV8-DETR中的Neck.
    4. 对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进RTDETR中的CCFM和YOLOV5-DETR、YOLOV8-DETR中的Neck.
    5. 修复没有使用wiou时候断点续寻的bug.
    6. 修复plot_result.py画结果图中乱码的问题.
    7. 更新使用教程.
    8. 百度云视频增加20240120更新说明.

- **20240128-rtdetr-v1.11**
    1. 增加CARAFE轻量化上采样算子.
    2. 增加DySample(ICCV2023)动态上采样算子.
    3. 增加Haar wavelet downsampling下采样算子.
    4. 增加Focaler-IoU,Focaler-GIoU,Focaler-DIoU,Focaler-CIoU,Focaler-EIoU,Focaler-SIoU,Focaler-Shape-IoU,Focaler-MPDIoU.
    5. 增加Focaler-Wise-IoU(v1,v2,v3)(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
    6. 使用DySample(ICCV2023)动态上采样算子对ASF-YOLO中的Attentional Scale Sequence Fusion进行二次创新.
    7. 更新使用教程.
    8. 百度云视频增加20240128更新说明.

- **20240206-rtdetr-v1.12**
    1. 新增Shift-ConvNets相关改进内容.(rtdetr-SWC.yaml,rtdetr-R50-SWC.yaml,yolov8-detr-C2f-SWC.yaml,yolov5-detr-C3-SWC.yaml)
    2. 使用UniRepLKNet中的DilatedReparamBlock对EMO中的iRMB进行二次创新.
    3. 使用Shift-ConvNets中的具有移位操作的卷积对EMO中的iRMB进行二次创新.
    4. 更新使用教程.
    5. 百度云视频增加20240206更新说明.

- **20240219-rtdetr-v1.13**
    1. 使用最新的Mamba架构(号称超越Transformer的新架构)改进rtdetr-r18,rtdetr-r50,yolov5-detr,yolov8-detr.
    2. 新增Powerful-IoU,Powerful-IoUV2,Inner-Powerful-IoU,Inner-Powerful-IoUV2,Focaler-Powerful-IoU,Focaler-Powerful-IoUV2,Wise-Powerful-IoU(v1,v2,v3),Wise-Powerful-IoUV2(v1,v2,v3)系列.
    3. 更新热力图脚本,使用方式可参考最新发的yolov5v7-gradcam的视频.
    4. 更新COCO脚本,增加其他指标输出.
    5. 更新使用教程.
    6. 百度云视频增加20240219更新说明.

- **20240225-rtdetr-v1.14**
    1. 新增YOLOV9中的RepNCSPELAN模块.
    2. 使用DBB,OREPA,DilatedReparamBlock,Conv3XC对YOLOV9中的RepNCSPELAN模块进行二次创新.
    3. 更新使用教程.
    4. 百度云视频增加20240225更新说明.

- **20240302-rtdetr-v1.15**
    1. 新增CGNet中的Light-weight Context Guided和Light-weight Context Guided DownSample模块.
    2. Neck模块新增BIFPN,并对其进行创新,支持替换不同的block.
    3. 为RTDETR定制SlideVarifocalLoss,EMASlideVarifocalLoss.
    4. 更新使用教程.
    5. 百度云视频增加20240302更新说明.

- **20240307-rtdetr-v1.16**
    1. 新增自研Neck结构Parallel Atrous Convolution Attention Pyramid Network, PAC-APN.附带模块内结构图
    2. 复现Lightweight Object Detection中的Dynamic Group Convolution Shuffle Transformer.
    3. 更新使用教程.
    4. 百度云视频增加20240307更新说明.

- **20240321-rtdetr-v1.17**
    1. 新增CVPR2024-RMT主干,并支持RetBlock改进RepC3.
    2. 新增2024年新出的Efficient Local Attention,并用其对HSFPN进行二次创新.
    3. 使用CVPR2021-CoordAttention对HSFPN进行二次创新.
    4. 更新使用教程,增加多个常见疑问解答.
    5. 百度云视频增加20240321更新说明.

- **20240404-rtdetr-v1.18**
    1. 新增CVPR2024 PKINet主干.
    2. 新增CVPR2024 PKINet中的PKIModule和CAA模块,提出C2f-PKI.
    3. 使用CVPR2024 PKINet中的Context Anchor Attention改进RepNCSPELAN、HSFPN.
    4. 新增CVPR2024 Frequency-Adaptive Dilated Convolution.
    5. 增加有效感受野可视化脚本.
    6. 更新使用教程
    7. 百度云视频增加20240404更新说明.

- **20240412-rtdetr-v1.19**
    1. 新增自研Focusing Diffusion Pyramid Network.
    2. 新增HCFNet针对小目标分割的Parallelized Patch-Aware Attention Module改进C2f.
    3. 新增HCFNet针对小目标分割的Dimension-Aware Selective Integration Module对自研Focusing Diffusion Pyramid Network再次进行创新.
    4. 更新使用教程.
    5. 百度云视频增加20240412更新说明.

- **20240427-rtdetr-v1.20**
    1. 新增mobilenetv4-backbone.
    2. 新增A Robust Feature Downsampling Module for Remote Sensing Visual Tasks中的下采样.
    3. 新增Context and Spatial Feature Calibration for Real-Time Semantic Segmentation中的Context and Spatial Feature Calibration.
    4. 更新使用教程.
    5. 百度云视频增加20240427更新说明.

- **20240502-rtdetr-v1.21**
    1. 新增支持content-guided attention fusion改进rtdetr-neck.
    2. 新增支持使用CAFM对CGAFusion进行二次改进,得到CAFMFusion改进rtdetr-neck.
    3. get_FPS.py脚本新增可以通过yaml测试推理速度.
    4. 新增自研RGCSPELAN,其比C3、ELAN、C2f、RepNCSPELAN更低参数量和计算量更快推理速度.
    5. 更新使用教程.
    6. 百度云视频增加20240502更新说明.

- **20240518-rtdetr-v1.22**
    1. 新增CVPR2024-StarNet-Backbone以及其衍生的改进(C3-Star、C3-Star-CAA、C2f-Star、C2f-Star-CAA、BasicBlock_Star、BottleNeck_Star).
    2. 使用CVPR2024-TransNext中的Convolutional GLU对CVPR2023-FasterBlock进行二次创新(C3_Faster_CGLU, C2f_Faster_CGLU, BasicBlock_Faster_Block_CGLU, BottleNeck_Faster_Block_CGLU).
    3. 新增PSFusion中的superficial detail fusion module、profound semantic fusion module.
    4. 更新使用教程.
    5. 百度云视频增加20240518更新说明.

- **20240525-rtdetr-v1.23**
    1. KAN In! Mamba Out!,集成pytorch-kan-conv，支持多种KAN变种！
    2. 同步DCNV4-CVPR2024最新代码.
    3. 更新使用教程.
    4. 百度云视频增加20240525更新说明.

- **20240608-rtdetr-v1.24**
    1. 新增自研ContextGuideFPN.
    2. 新增detail-enhanced convolution改进RTDETR.
    3. 新增自研SMPCGLU，里面的模块分别来自CVPR2023和CVPR2024.
    4. 更新使用教程.
    5. 百度云视频增加20240608更新说明.

- **20240618-rtdetr-v1.25**
    1. 新增支持物理传热启发的视觉表征模型vHeat中的vHeatBlock.
    2. 新增自研重校准特征金字塔网络(Re-CalibrationFPN),推出多个版本(P2345,P345,P3456).
    3. 新增WaveletPool改进上采样和下采样.
    4. 更新使用教程.
    5. 百度云视频增加20240618更新说明.

- **20240622-rtdetr-v1.26**
    1. 新增RtDetr-Mamba.
    2. 新增GLSA改进rtdetr-neck.
    3. 新增GLSA对BIFPN进行二次创新.
    4. 更新使用教程.
    5. 百度云视频增加20240622更新说明.

- **20240703-rtdetr-v1.27**
    1. 新增UCTransNet中的ChannelTransformer改进rtdetr-neck.
    2. 新增自研SmallObjectEnhancePyramid.
    3. 新增SwiftFormer的EfficientAdditiveAttention改进AIFI.
    4. 更新使用教程.
    5. 百度云视频增加20240703更新说明.

- **20240715-rtdetr-v1.28**
    1. 新增自研Context-Guided Spatial Feature Reconstruction Feature Pyramid Network.
    2. 新增Wavelet Convolutions for Large Receptive Fields中的WTConv改进BasicBlock.
    3. 新增UBRFC-Net中的Adaptive Fine-Grained Channel Attention.
    4. 更新使用教程.
    5. 百度云视频增加20240715更新说明.

- **20240725-rtdetr-v1.29**
    1. 新增ECCV2024-SMFANet中的Feature Modulation block.
    2. 新增Rethinking Performance Gains in Image Dehazing Networks中的gConvblock.
    3. 更新使用教程.
    4. 百度云视频增加20240725更新说明.

- **20240802-rtdetr-v1.30**
    1. 新增LDConv.
    2. 新增MAF-YOLO中的MAFPN，并利用BIFPN的思想对MAFPN进行二次创新得到BIMAFPN.
    3. 更新使用教程.
    4. 百度云视频增加20240802更新说明.

- **20240815-rtdetr-v1.31**
    1. 新增YOLO-MIF中的WDBB、DeepDBB的重参数化模块.
    2. 新增SLAB中的RepBN改进AIFI.
    3. 更新使用教程.
    4. 百度云视频增加20240815更新说明.

- **20240825-rtdetr-v1.32**
    1. 新增CAS-ViT中的AdditiveBlock和CSP思想改进backbone.
    2. 新增CAS-ViT中的AdditiveBlock改进AIFI.
    3. 新增自研Efficient Multi-Branch&Scale FPN.
    4. 更新使用教程.
    5. 百度云视频增加20240825更新说明.

- **20240902-rtdetr-v1.33**
    1. 新增CMTFUnet和TransNext的二次创新模块.
    2. 新增自研CSP-Partial Multi-Scale Feature Aggregation.
    3. 更新使用教程.
    4. 百度云视频增加20240902更新说明.

- **20240912-rtdetr-v1.34**
    1. 新增Cross-Layer Feature Pyramid Transformer for Small Object Detection in Aerial Images中的CFPT.
    2. 新增ICLR2024中的MogaBlock.
    3. 更新使用教程.
    4. 百度云视频增加20240912更新说明.

- **20240926-rtdetr-v1.35**
    1. 新增CVPR2024-SHViT中的SHSABlock和其的二次创新.
    2. 新增BIBM2024-SMAFormer中的SMAFormerBlock和其的二次创新.
    3. 新增TPAMI2024-FreqFusion中的FreqFusion改进Neck.
    4. 新增自研MutilBackBone-DynamicAlignFusion.
    5. 更新使用教程.
    6. 百度云视频增加20240926更新说明.

- **20241020-rtdetr-v1.36**
    1. 新增Histoformer ECCV2024中的Dynamic-range Histogram Self-Attention改进AIFI.
    2. 新增自研CSP-MutilScaleEdgeInformationEnhance.
    3. 新增Efficient Frequency-Domain Image Deraining with Contrastive Regularization ECCV2024中的Fused_Fourier_Conv_Mixer与CSP思想结合改进rtdetr-backbone.
    4. 更新使用教程.
    5. 百度云视频增加20241020更新说明.

- **20241106-rtdetr-v1.37**
    1. 新增自研CSP-FreqSpatial.
    2. 新增SFHformer ECCV2024中的block与CSP思想结合改进 rtdetr-backbone.
    3. 新增Revitalizing Convolutional Network for Image Restoration TPAMI2024中的MSM与CSP思想结合改进rtdetr-backbone.
    4. 更新使用教程.
    5. 百度云视频增加20241106更新说明.

- **20241118-rtdetr-v1.38**
    1. 基于自研CSP-MutilScaleEdgeInformationEnhance再次创新得到CSP-MutilScaleEdgeInformationSelect.
    2. 新增Pattern Recognition 2024|DRANet中的HDRAB和RAB模块与CSP思想结合改进rtdetr-backbone.
    3. 新增ECCV2022-ELAN中的Local feature extraction改进RepC3.
    4. 更新使用教程.
    5. 百度云视频增加20241118更新说明.

- **20241130-rtdetr-v1.39**
    1. 新增自研GlobalEdgeInformationTransfer.
    2. 新增FreqFormer的Frequency-aware Cascade Attention与CSP结合改进backbone.
    3. 更新使用教程.
    4. 百度云视频增加20241130更新说明.

- **20241215-rtdetr-v1.40**
    1. 新增CrossFormer中的DynamicPosBias-Attention改进AIFI.
    2. 新增CAMixerSR中的CAMixer与CSP结合改进backbone.
    3. 修改保存模型规则,原本为fp16变成fp32,详细请看本期更新视频.
    4. 百度云视频增加20241215更新说明.

- **20241216-rtdetr-v1.41**
    1. 新增Hyper-YOLO中的Hypergraph Computation in Semantic Space和Mixed Aggregation Network改进rtdetr.
    2. 修复已知bug.
    3. 更新使用教程.
    4. 百度云视频增加20241216更新说明.

- **20241228-rtdetr-v1.42**
    1. 新增基于Hyper-YOLO中的Mixed Aggregation Network三个二次改进系列.
    2. 新增使用MSA^2 Net中的Multi-Scale Adaptive Spatial Attention Gate改进rtdetr-neck.
    3. 新增使用MSA^2 Net中的Multi-Scale Adaptive Spatial Attention Gate改进自研系列的MutilBackbone.
    4. 更新使用教程.
    5. 百度云视频增加20241228更新说明.

- **20250111-rtdetr-v1.43**
    1. 新增CRAFT-SR中的high-frequency enhancement residual block与CSP结合改进backbone.
    2. 新增AAAI2025-TBSN中的DTAB改进backbone、AIFI.
    3. 新增ECCV2024-FSEL中的多个模块改进rtdetr.
    4. 新增ACMMM2024-WFEN中的多个模块改进rtdetr.
    5. 更新使用教程.
    6. 百度云视频增加20250111更新说明.

- **20250119-rtdetr-v1.44**
    1. 新增AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection中的Pinwheel-shaped Convolution类型改进.
    2. 新增AAAI2025 ConDSeg中的ContrastDrivenFeatureAggregation与ACMMM2024 WFEN中的小波变换进行创新.
    3. 更新使用教程.
    4. 百度云视频增加20250119更新说明.

- **20250204-rtdetr-v1.45**
    1. 新增ELGC-Net的改进及其二次创新.
    2. 新增ICLR2025 PolaFormer中的PolaAttention改进AIFI.
    3. 新增遥感目标检测Strip R-CNN中的StripBlock及其二次创新.
    4. 新增BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation中的Frequency-Spatial Attention和Multi-scale Progressive Channel Attention.
    5. 更新使用教程.
    6. 百度云视频增加20250204更新说明.

- **20250206-rtdetr-v1.46**
    1. 新增ICLR2025 Kolmogorov-Arnold Transformer中的KAT及其配合FasterBlock的二次创新.<此模块需要编译>
    2. 更新使用教程.
    3. 百度云视频增加20250206更新说明.

- **20250216-rtdetr-v1.47**
    1. 新增自研模块DynamicInceptionDWConv2d.
    2. 新增GlobalFilter和DynamicFilter.
    3. 更新使用教程.
    4. 百度云视频增加20250216更新说明.

- **20250303-rtdetr-v1.48**
    1. 新增自研模块Hierarchical Attention Fusion并提供多种使用方式.
    2. 新增ICLR2025-Token Statistics Transformer中的TSSA改进AIFI.
    3. 新增MHAF-YOLO中的RepHMS.<这个是YOLO群内的一个博士新作品>
    4. 更新使用教程.
    5. 百度云视频增加20250303更新说明.

- **20250315-rtdetr-v1.49**
    1. 新增CVPR2024-Adaptive Sparse Transformer的模块改进aifi.
    2. 新增CVPR2025-MambaIR的模块.
    3. 新增CVPR2025-SCSegamba中的模块.
    4. 新增CVPR2025-MambaOut中的模块.
    5. 新增CVPR2025-DEIM MAL损失函数.
    6. 更新使用教程.
    7. 百度云视频增加20250315更新说明.

- **20250403-rtdetr-v1.50**
    1. 新增CVPR2025-MambaOut与CVPR2024-UniRepLKNet二次创新后的模块.
    2. 新增CVPR2025-EfficientViM和其与CVPR2024-TransNeXt的二次创新后的模块.
    3. 新增CVPR2024-EMCAD中的EUCB.
    4. 新增CVPR2025-BHViT中的ShiftChannelMix和CVPR2024-EMCAD中的EUCB二次创新模块.
    5. 新增rtdetr-EMBSFPN.yaml方案上引入[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix.
    6. 新增CVPR2025-HVI中的Intensity Enhancement Layer.
    7. 新增CVPR2025-OverLock中的模块.
    8. 更新使用教程.
    9. 百度云视频增加20250403更新说明.

- **20250420-rtdetr-v1.51**
    1. 新增ICLR2024-FTIC中的多个模块、以及其与ICLR2025-PolaFormer的二次创新模块.
    2. 新增CVPR2024-DCMPNet中的多个模块.
    3. 新增ICLR2025-PolaFormer与CVPR2024-TransNext的二次创新模块.
    4. 新增CVPR2025-OverLock中的GDSAFusion.
    5. 新增统计配置文件的计算量和参数量并排序的脚本.
    6. 更新使用教程.
    7. 百度云视频增加20250420更新说明.

- **20250508-rtdetr-v1.52**
    1. 新增CVPR2025-MobileMamba的相关改进.
    2. 新增LEGNet中的LFEModule和LoGStem改进.
    3. 新增WACV2025-SEMNet中的Snake Bi-Directional Sequence Modelling (SBSM)和Spatially-Enhanced Feedforward Network (SEFN)的多个改进，并含有二次创新相关内容.
    4. 新增CVPR2025-LSNet中的多个改进，并含有二次创新相关内容.
    5. 新增CVPR2025-DynamicTan中的多个改进，并含有二次创新相关内容.
    6. 更新使用教程.
    7. 百度云视频增加20250508更新说明.

- **20250523-rtdetr-v1.53**
    1. 新增TransMamba中的多个改进.
    2. 新增CVPR2025-EVSSM中的多个改进.
    3. 新增CVPR2025-DarkIR中的多个改进.
    4. 更新使用教程.
    5. 百度云视频增加20250523更新说明.

- **20250606-rtdetr-v1.54**
    1. 新增CVPR2025-FDConv的改进及其多个二次创新模块.
    2. 新增DSA: Deformable Spatial Attention的改进及其多个二次创新模块.
    3. 新增CVPR2025-MaIR中的Residual Mamba Block.
    4. 更新使用教程.
    5. 百度云视频增加20250606更新说明.

- **20250622-rtdetr-v1.55**
    1. 新增ECCV2024-rethinkingfpn中的模块，并对原创改进SOEP再次创新。
    2. 新增CVPR2024-SFSConv的改进及其多个二次创新模块.
    3. 新增CVPR2025-GroupMamba中的模块.
    4. 新增CVPR2025-MambaVision中的模块.
    5. 新增AAAI2025-FBRTYOLO中的模块.
    5. 更新使用教程.
    6. 百度云视频增加20250622更新说明.
    7. 修复在torch2.6.0以及以上的版本会出现模型读取失败的问题.