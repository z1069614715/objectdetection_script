# Ultralytics-YOLO项目详细说明

1. 本项目集成了YOLOv8、v10、11、12乃至前沿的YOLO26等全系列基础模型。 无论是做横向对比实验，还是纵向的版本改进，无需到处找资源，一个项目就能满足你所有的实验需求！
2. 核心代码已实现高度模块化与解耦，专为新手优化。 你完全不需要死磕底层复杂代码，只需像搭积木一样简单修改YAML配置文件，就能轻松实现各种改进模块的自由组合。
3. 在YOLO研究日趋饱和的今天，简单的模块堆砌已难以支撑一篇高质量的毕业论文。我们深知你的痛点——这里不只是给你现成方案，更配套独家「二次创新」系统课程，手把手拆解模块设计的底层逻辑，带你完成从模仿者到创造者的真正蜕变，打造出独属于你、真正有价值的创新成果。担心自己基础薄弱、看不懂怎么创新？完全不用怕，项目内持续更新团队自研的高创新度模块，即便是零基础小白也能直接上手、开箱即用，让你的论文创新度稳稳在线，轻松应对答辩！
4. 针对有代码基础但受困于Ultralytics复杂架构的同学， 本项目引入了来自DFine、DEIM项目中成熟的“万物皆可融”架构思想。你无需纠结模块注册等信息，只需遵循我所提供的标准接口规范，即可将自定义魔改模块无缝融入YAML配置，与各类CSP变种灵活结合。
5. 实验跑通了，却不知道如何写创新点？ 本项目将定期拆解高分论文，传授写作心法，教你如何将实验成果转化为逻辑严密、亮点突出的高质量学术论文，解决写作难题！
6. 毕业设计缺少高大上的展示界面？ 别担心，项目会内置基于HTML的通用可视化界面，开箱即用，完美补齐毕业论文的最后一块拼图，助你从容应对答辩！
7. 项目内有CVPR2026-Does YOLO Really Need to See Every Training Image in Every Epoch中的AFSS加速训练机制，助你在有限的设备内训练速度更快！
8. 购买即享专属技术交流群， 这里有业内公认的高效答疑服务，以及志同道合的伙伴互助交流。拒绝闭门造车，让我们带你避开深坑，高效通关！  

## 针对于已经入手了yolov8/yolo11项目的同学来说，如果你有以下几点需求，可以考虑追加入手！
1. 想用最新的YOLO26做实验！而且本项目支持v8、v10、11、12、26全系列版本！
2. 想深入学习改进创新的同学，本项目会附带二次创新的通用教程，手把手教你设计出属于自己的创新模块！
3. 做完实验不知道怎么写论文？本项目会定期拆解高分论文案例，教你如何把实验结果写成逻辑清晰、亮点突出的高质量学术论文
4. 想自己魔改模块的同学！本项目提供超级简单的模块注册方式，只需按照教程操作，就能轻松注册自己的模块，还能和各种CSP变种随意组合！
5. 想用更多自研的模块让你的论文创新度稳稳在线！

**注意：此项目不是包含之前的YOLO项目，本项目的意思是这里面的创新点都可以用于YOLOV8,YOLOV10,YOLO11,YOLO12,YOLO26**

## 模块列表(这些模块均已在代码中注册好，只需要修改yaml可以直接实验)

- ultralytics/nn/extra_modules/attention(配置文件在ultralytics/cfg/models/improve/attention)

    1. ultralytics/nn/extra_modules/attention/SEAM.py
    2. CVPR2021|ultralytics/nn/extra_modules/attention/ca.py
    3. ICASSP2023|ultralytics/nn/extra_modules/attention/ema.py
    4. ICML2021|ultralytics/nn/extra_modules/attention/simam.py
    5. ICCV2023|ultralytics/nn/extra_modules/attention/lsk.py
    6. WACV2024|ultralytics/nn/extra_modules/attention/DeformableLKA.py
    7. ultralytics/nn/extra_modules/attention/mlca.py
    8. BIBM2024|ultralytics/nn/extra_modules/attention/FSA.py
    9. AAAI2025|ultralytics/nn/extra_modules/attention/CDFA.py
    10. TGRS2025|ultralytics/nn/extra_modules/attention/MCA.py
    11. CVPR2025|ultralytics/nn/extra_modules/attention/CASAB.py 
    12. NN2025|ultralytics/nn/extra_modules/attention/KSFA.py
    13. TGRS2025|ultralytics/nn/extra_modules/attention/ACA.py
    14. TGRS2025|ultralytics/nn/extra_modules/attention/DHPF.py
    15. TGRS2025|ultralytics/nn/extra_modules/attention/ACAB.py
    16. 自研模块|ultralytics/nn/extra_modules/attention/FSDA.py

- ultralytics/nn/extra_modules/conv_module(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第五节,支持与attention部分联合改进CSP模块中的残差块)

    1. CVPR2021|ultralytics/nn/extra_modules/conv_module/dbb.py
    2. TIP2024|ultralytics/nn/extra_modules/conv_module/deconv.py
    3. ICCV2023|ultralytics/nn/extra_modules/conv_module/dynamic_snake_conv.py
    4. CVPR2023|ultralytics/nn/extra_modules/conv_module/pconv.py
    5. AAAI2025|ultralytics/nn/extra_modules/conv_module/psconv.py
    6. CVPR2025|ultralytics/nn/extra_modules/conv_module/ShiftwiseConv.py
    7. ultralytics/nn/extra_modules/conv_module/wdbb.py
    8. ultralytics/nn/extra_modules/conv_module/deepdbb.py
    9. ECCV2024|ultralytics/nn/extra_modules/conv_module/wtconv2d.py
    10. CVPR2023|ultralytics/nn/extra_modules/conv_module/ScConv.py
    11. ultralytics/nn/extra_modules/conv_module/dcnv2.py
    12. CVPR2024|ultralytics/nn/extra_modules/conv_module/DilatedReparamConv.py
    13. ultralytics/nn/extra_modules/conv_module/gConv.py
    14. CVPR2024|ultralytics/nn/extra_modules/conv_module/IDWC.py
    15. ultralytics/nn/extra_modules/conv_module/DSA.py
    16. CVPR2025|ultralytics/nn/extra_modules/conv_module/FDConv.py
    17. CVPR2023|ultralytics/nn/extra_modules/conv_module/dcnv3.py
    18. CVPR2024|ultralytics/nn/extra_modules/conv_module/dcnv4.py
    19. CVPR2024|ultralytics/nn/extra_modules/conv_module/DynamicConv.py
    20. CVPR2024|ultralytics/nn/extra_modules/conv_module/FADC.py
    21. CVPR2023|ultralytics/nn/extra_modules/conv_module/SMPConv.py
    22. MIA2025|ultralytics/nn/extra_modules/conv_module/FourierConv.py
    23. CVPR2024|ultralytics/nn/extra_modules/conv_module/SFSConv.py
    24. ICCV2025|ultralytics/nn/extra_modules/conv_module/ConvAttn.py
    25. ICCV2025|ultralytics/nn/extra_modules/conv_module/Converse2D.py
    26. CVPR2025|ultralytics/nn/extra_modules/conv_module/gcconv.py
    27. ACCV2024|ultralytics/nn/extra_modules/conv_module/RMBC.py
    28. CVPR2026|ultralytics/nn/extra_modules/conv_module/DEGConv.py

- engine/extre_module/custom_nn/stem(配置文件在ultralytics/cfg/models/improve/stem)

    1. ultralytics/nn/extra_modules/stem/SRFD.py
    2. ultralytics/nn/extra_modules/stem/LoG.py
    3. ICCV2023|ultralytics/nn/extra_modules/stem/RepStem.py
    4. 自研模块|ultralytics/nn/extra_modules/stem/ODALStem.py
    5. 自研模块|ultralytics/nn/extra_modules/stem/DPAS.py
    6. 自研模块|ultralytics/nn/extra_modules/stem/WGFS.py

- ultralytics/nn/extra_modules/upsample(配置文件在ultralytics/cfg/models/improve/upsample)

    1. CVPR2024|ultralytics/nn/extra_modules/upsample/eucb.py
    2. CVPR2024|ultralytics/nn/extra_modules/upsample/eucb_sc.py
    3. ultralytics/nn/extra_modules/upsample/WaveletUnPool.py
    4. ICCV2019|ultralytics/nn/extra_modules/upsample/CARAFE.py
    5. ICCV2023|ultralytics/nn/extra_modules/upsample/DySample.py
    6. ICCV2025|ultralytics/nn/extra_modules/upsample/Converse2D_Up.py
    7. CVPR2025|ultralytics/nn/extra_modules/upsample/DSUB.py

- ultralytics/nn/extra_modules/downsample(配置文件在ultralytics/cfg/models/improve/downsample)

    1. TIP2020|ultralytics/nn/extra_modules/downsample/gcnet.py
    2. 自研模块|ultralytics/nn/extra_modules/downsample/lawds.py 
    3. ultralytics/nn/extra_modules/downsample/WaveletPool.py
    4. ultralytics/nn/extra_modules/downsample/ADown.py
    5. ultralytics/nn/extra_modules/downsample/YOLOV7Down.py
    6. ultralytics/nn/extra_modules/downsample/SPDConv.py
    7. ultralytics/nn/extra_modules/downsample/HWD.py
    8. ultralytics/nn/extra_modules/downsample/DRFD.py
    9. TGRS2025|ultralytics/nn/extra_modules/conv_module/FSConv.py
    10. 自研模块|ultralytics/nn/extra_modules/downsample/EdgeLAWDS.py
    11. 自研模块|ultralytics/nn/extra_modules/downsample/FreqLAWDS.py
    12. 自研模块|ultralytics/nn/extra_modules/downsample/RouterLAWDS.py
    13. 自研模块|ultralytics/nn/extra_modules/downsample/FSCGD.py

- ultralytics/nn/extra_modules/module(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. AAAI2025|ultralytics/nn/extra_modules/module/APBottleneck.py
    2. CVPR2025|ultralytics/nn/extra_modules/module/efficientVIM.py
    3. CVPR2023|ultralytics/nn/extra_modules/module/fasterblock.py
    4. CVPR2024|ultralytics/nn/extra_modules/module/starblock.py
    5. ultralytics/nn/extra_modules/module/DWR.py
    6. CVPR2024|ultralytics/nn/extra_modules/module/UniRepLKBlock.py
    7. CVPR2025|ultralytics/nn/extra_modules/module/mambaout.py
    8. AAAI2024|ultralytics/nn/extra_modules/module/DynamicFilter.py
    9. ultralytics/nn/extra_modules/module/StripBlock.py
    10. TGRS2024|ultralytics/nn/extra_modules/module/elgca.py
    11. CVPR2024|ultralytics/nn/extra_modules/module/LEGM.py
    12. ICCV2023|ultralytics/nn/extra_modules/module/iRMB.py
    13. TPAMI2025|ultralytics/nn/extra_modules/module/MSBlock.py
    14. ICLR2024|ultralytics/nn/extra_modules/module/FATBlock.py
    15. CVPR2024|ultralytics/nn/extra_modules/module/MSCB.py
    16. ultralytics/nn/extra_modules/module/LEGBlock.py
    17. ultralytics/nn/extra_modules/module/GLSA.py
    18. CVPR2025|ultralytics/nn/extra_modules/module/RCB.py
    19. ECCV2024|ultralytics/nn/extra_modules/module/JDPM.py
    20. CVPR2025|ultralytics/nn/extra_modules/module/vHeat.py
    21. CVPR2025|ultralytics/nn/extra_modules/module/EBlock.py
    22. CVPR2025|ultralytics/nn/extra_modules/module/DBlock.py
    23. ECCV2024|ultralytics/nn/extra_modules/module/FMB.py
    24. CVPR2024|ultralytics/nn/extra_modules/module/IDWB.py
    25. ECCV2022|ultralytics/nn/extra_modules/module/LFE.py
    26. AAAI2025|ultralytics/nn/extra_modules/module/FCM.py
    27. CVPR2024|ultralytics/nn/extra_modules/module/RepViTBlock.py
    28. CVPR2024|ultralytics/nn/extra_modules/module/PKIModule.py
    29. CVPR2024|ultralytics/nn/extra_modules/module/camixer.py
    30. ICCV2025|ultralytics/nn/extra_modules/module/ESC.py
    31. TGRS2025|ultralytics/nn/extra_modules/module/ARF.py
    32. AAAI2024|ultralytics/nn/extra_modules/module/CFBlock.py
    33. IJCV2024|ultralytics/nn/extra_modules/module/FMA.py
    34. ultralytics/nn/extra_modules/module/LWGA.py
    35. TGRS2025|ultralytics/nn/extra_modules/module/CSSC.py
    36. TGRS2025|ultralytics/nn/extra_modules/module/CNCM.py
    37. ICCV2025|ultralytics/nn/extra_modules/module/HFRB.py
    38. ICIP2025|ultralytics/nn/extra_modules/module/EVA.py
    39. CVPR2025|ultralytics/nn/extra_modules/module/IEL.py
    40. MICCAI2023|ultralytics/nn/extra_modules/module/MFEBlock.py
    41. AAAI2026|ultralytics/nn/extra_modules/module/PartialNetBlock.py
    42. TGRS2025|ultralytics/nn/extra_modules/module/DRG.py
    43. ultralytics/nn/extra_modules/module/Wave2D.py
    44. TGRS2025|ultralytics/nn/extra_modules/module/GLGM.py
    45. TGRS2025|ultralytics/nn/extra_modules/module/MAC.py
    46. AAAI2026|ultralytics/nn/extra_modules/module/SPJFB.py
    47. 自研模块|ultralytics/nn/extra_modules/module/FasterCGABlock.py
    48. CVPR2026|ultralytics/nn/extra_modules/module/sparse_mamba_block.py
    49. CVPR2026|ultralytics/nn/extra_modules/module/MSInit.py
    50. CVPR2026|ultralytics/nn/extra_modules/module/PFG.py
    51. CVPR2026|ultralytics/nn/extra_modules/module/LFP.py
    52. 自研模块|ultralytics/nn/extra_modules/module/AMSI.py
    53. 自研模块|ultralytics/nn/extra_modules/module/CMSI.py
    54. 自研模块|ultralytics/nn/extra_modules/module/FMSI.py
    55. 自研模块|ultralytics/nn/extra_modules/module/HPFGA.py
    56. 自研模块|ultralytics/nn/extra_modules/module/DPFGA.py
    57. 自研模块|ultralytics/nn/extra_modules/module/HOIE.py
    58. 自研模块|ultralytics/nn/extra_modules/module/ADIE.py
    59. TGRS2025|ultralytics/nn/extra_modules/module/DSEBlock.py
    60. TGRS2025|ultralytics/nn/extra_modules/module/LaSEA.py
    61. CVPR2026|ultralytics/nn/extra_modules/module/SFEB.py
    62. TIP2026|ultralytics/nn/extra_modules/module/FourierSR.py
    63. CVPR2026|ultralytics/nn/extra_modules/module/FrequencyCM.py

- ultralytics/nn/extra_modules/block (此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)
    
    1. ultralytics/nn/extra_modules/block/CSPBlock.py
    2. TPAMI2025|ultralytics/nn/extra_modules/block/MANet.py
    3. TPAMI2024|ultralytics/nn/extra_modules/block/MetaFormer.py

- ultralytics/nn/extra_modules/transformer(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. ICLR2025|ultralytics/nn/extra_modules/transformer/PolaLinearAttention.py
    2. CVPR2023|ultralytics/nn/extra_modules/transformer/biformer.py
    3. CVPR2023|ultralytics/nn/extra_modules/transformer/CascadedGroupAttention.py
    4. CVPR2022|ultralytics/nn/extra_modules/transformer/DAttention.py
    5. ICLR2022|ultralytics/nn/extra_modules/transformer/DPBAttention.py
    6. CVPR2024|ultralytics/nn/extra_modules/transformer/AdaptiveSparseSA.py
    7. ultralytics/nn/extra_modules/transformer/GSA.py
    8. ultralytics/nn/extra_modules/transformer/RSA.py
    9. ECCV2024|ultralytics/nn/extra_modules/transformer/FSSA.py
    10. AAAI2025|ultralytics/nn/extra_modules/transformer/DilatedGCSA.py
    11. AAAI2025|ultralytics/nn/extra_modules/transformer/DilatedMWSA.py
    12. CVPR2024|ultralytics/nn/extra_modules/transformer/SHSA.py
    13. IJCAI2024|ultralytics/nn/extra_modules/transformer/CTA.py
    14. IJCAI2024|ultralytics/nn/extra_modules/transformer/SFA.py
    15. ultralytics/nn/extra_modules/transformer/MSLA.py
    16. ACMMM2025|ultralytics/nn/extra_modules/transformer/CPIA_SA.py
    17. NN2025|ultralytics/nn/extra_modules/transformer/TokenSelectAttention.py
    18. CVPR2025|ultralytics/nn/extra_modules/transformer/TAB.py
    19. TPAMI2025|ultralytics/nn/extra_modules/transformer/LRSA.py
    20. ICCV2025|ultralytics/nn/extra_modules/transformer/MALA.py
    21. ICML2023|ultralytics/nn/extra_modules/transformer/MUA.py
    22. ACMMM2025|ultralytics/nn/extra_modules/transformer/EGSA.py
    23. ACMMM2025|ultralytics/nn/extra_modules/transformer/SWSA.py
    24. AAAI2026|ultralytics/nn/extra_modules/transformer/DHOGSA.py
    25. NeurIPS2025|ultralytics/nn/extra_modules/transformer/CBSA.py
    26. TGRS2025|ultralytics/nn/extra_modules/transformer/DPWA.py
    27. TIP2025|ultralytics/nn/extra_modules/transformer/DWM_MSA.py
    28. CVPR2026|ultralytics/nn/extra_modules/transformer/BinaryAttention.py
    29. CVPR2025|ultralytics/nn/extra_modules/transformer/wca.py
    30. TGRS2026|ultralytics/nn/extra_modules/transformer/CGTA.py
    31. TGRS2026|ultralytics/nn/extra_modules/transformer/LCGA.py
    32. AAAI2026|ultralytics/nn/extra_modules/transformer/CirculantAttention.py
    33. CVPR2026|ultralytics/nn/extra_modules/transformer/WDAM.py

- ultralytics/nn/extra_modules/mamba(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. AAAI2025|ultralytics/nn/extra_modules/mamba/SS2D.py
    2. CVPR2025|ultralytics/nn/extra_modules/mamba/ASSM.py
    3. CVPR2025|ultralytics/nn/extra_modules/mamba/SAVSS.py
    4. CVPR2025|ultralytics/nn/extra_modules/mamba/MobileMamba/mobilemamba.py
    5. CVPR2025|ultralytics/nn/extra_modules/mamba/MaIR.py
    6. TGRS2025|ultralytics/nn/extra_modules/mamba/GLVSS.py
    7. ICCV2025|ultralytics/nn/extra_modules/mamba/VSSD.py
    8. ICCV2025|ultralytics/nn/extra_modules/mamba/TinyViM.py
    9. INFFUS2025|ultralytics/nn/extra_modules/mamba/CSI.py
    10. TIP2025|ultralytics/nn/extra_modules/mamba/SFMB.py
    11. TGRS2025|ultralytics/nn/extra_modules/mamba/GLSS.py
    12. TGRS2025|ultralytics/nn/extra_modules/mamba/GLSS2D.py
    13. CVPR2026|ultralytics/nn/extra_modules/mamba/TransMixer.py
    14. CVPR2026|ultralytics/nn/extra_modules/mamba/sparse_state_space.py

- ultralytics/nn/extra_modules/mlp(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. CVPR2024|ultralytics/nn/extra_modules/mlp/ConvolutionalGLU.py
    2. IJCAI2024|ultralytics/nn/extra_modules/mlp/DFFN.py
    3. ICLR2024|ultralytics/nn/extra_modules/mlp/FMFFN.py
    4. CVPR2024|ultralytics/nn/extra_modules/mlp/FRFN.py
    5. ECCV2024|ultralytics/nn/extra_modules/mlp/EFFN.py 
    6. WACV2025|ultralytics/nn/extra_modules/mlp/SEFN.py
    7. ICLR2025|ultralytics/nn/extra_modules/mlp/KAN.py
    8. CVPR2025|ultralytics/nn/extra_modules/mlp/EDFFN.py
    9. ICVJ2024|ultralytics/nn/extra_modules/mlp/DML.py
    10. AAAI2026|ultralytics/nn/extra_modules/mlp/DIFF.py
    11. 自研模块|ultralytics/nn/extra_modules/mlp/MCCG.py
    12. 自研模块|ultralytics/nn/extra_modules/mlp/DSRG.py
    13. CVPR2026|ultralytics/nn/extra_modules/mlp/AFFN.py

- ultralytics/nn/extra_modules/neck(配置文件在ultralytics/cfg/models/improve/neck)

    1. ultralytics/nn/extra_modules/neck/ASF.py
    2. ultralytics/nn/extra_modules/neck/BiFPN.py
    3. AAAI2022|ultralytics/nn/extra_modules/neck/CTrans.py
    4. ultralytics/nn/extra_modules/neck/EfficientRepBiPAN.py
    5. ultralytics/nn/extra_modules/neck/GFPN.py
    6. ultralytics/nn/extra_modules/neck/HSFPN.py
    7. AAAI2025|ultralytics/nn/extra_modules/neck/HS_FPN.py
    8. TPAMI2025|ultralytics/nn/extra_modules/neck/HyperComputeModule.py
    9. ultralytics/nn/extra_modules/neck/SlimNeck.py
    10. ultralytics/nn/extra_modules/neck/GoldYOLO.py
    11. ultralytics/nn/extra_modules/neck/EMBSFPN.py
    12. ultralytics/nn/extra_modules/neck/FDPN.py((里面有三个自研模块FocusFeature、DynamicFrequencyFocusFeature、AlignmentGuidedFocusFeature))
    13. PR2026|ultralytics/nn/extra_modules/neck/A3FPN.py

- ultralytics/nn/extra_modules/featurefusion(配置文件在ultralytics/cfg/models/improve/featurefusion)

    1. 自研模块|ultralytics/nn/extra_modules/featurefusion/cgfm.py
    2. BMVC2024|ultralytics/nn/extra_modules/featurefusion/msga.py
    3. CVPR2024|ultralytics/nn/extra_modules/featurefusion/mfm.py
    4. TIP2023|ultralytics/nn/extra_modules/featurefusion/CSFCN.py
    5. BIBM2024|ultralytics/nn/extra_modules/featurefusion/mpca.py
    6. ACMMM2024|ultralytics/nn/extra_modules/featurefusion/wfu.py
    7. CVPR2025|ultralytics/nn/extra_modules/featurefusion/GDSAFusion.py
    8. ultralytics/nn/extra_modules/featurefusion/PST.py
    9. TGRS2025|ultralytics/nn/extra_modules/featurefusion/MSAM.py
    10. INFFUS2025|ultralytics/nn/extra_modules/featurefusion/DPCF.py
    11. CVRP2025|ultralytics/nn/extra_modules/featurefusion/LCA.py
    12. TGRS2025|ultralytics/nn/extra_modules/featurefusion/HFFE.py
    13. TGRS2025|ultralytics/nn/extra_modules/featurefusion/MFPM.py
    14. TGRS2025|ultralytics/nn/extra_modules/featurefusion/ERM.py
    15. TIP2025|ultralytics/nn/extra_modules/featurefusion/CAFM.py
    16. TIP2024|ultralytics/nn/extra_modules/featurefusion/CGAFusion.py
    17. IF2023|ultralytics/nn/extra_modules/featurefusion/PSFM.py
    18. IF2023|ultralytics/nn/extra_modules/featurefusion/SDFM.py
    19. 自研模块|ultralytics/nn/extra_modules/featurefusion/DAF.py
    20. 自研模块|ultralytics/nn/extra_modules/featurefusion/CIDAF.py
    21. 自研模块|ultralytics/nn/extra_modules/featurefusion/WDAF.py
    22. CVPR2026|ultralytics/nn/extra_modules/featurefusion/SFSFusion.py
    23. CVPR2026|ultralytics/nn/extra_modules/featurefusion/FAAFusion.py
    24. PR2026|ultralytics/nn/extra_modules/featurefusion/HAFFormer.py
    25. 自研模块|ultralytics/nn/extra_modules/featurefusion/LowFrequencyFeatureFusion.py
    26. 自研模块|ultralytics/nn/extra_modules/featurefusion/DCGRM.py
    27. 自研模块|ultralytics/nn/extra_modules/featurefusion/MSCRM.py

- ultralytics/nn/extra_modules/norm(此部分内容教程可以看GuideVideo-MG.md中的改进模块-使用教程的第一和四节)

    1. ICML2024|engine/extre_module/custom_nn/transformer/repbn.py
    2. CVPR2025|engine/extre_module/custom_nn/transformer/dyt.py
    3. engine/extre_module/custom_nn/norm/derf.py

- ultralytics/nn/extra_modules/featurepreprocess(配置文件在ultralytics/cfg/models/improve/featurepreprocess)

    1. TGRS2025|ultralytics/nn/extra_modules/featurepreprocess/FAENet.py

- ultralytics/nn/extra_modules/head(配置文件在ultralytics/cfg/models/improve/head)

    1. ultralytics/nn/extra_modules/head/LSPCD.py
    2. ultralytics/nn/extra_modules/head/LQE.py

## Loss 列表

#### 默认配置（兼容）

- cls_loss=bce
- iou_loss=ciou
- iou_aux=none

- cls_loss（分类损失）

    1. bce
    2. slide
    3. ema_slide
    4. focal
    5. varifocal
    6. qualityfocal

- iou_loss（IoU主损失）

    1. 基础形式：
       iou、giou、diou、ciou、eiou、siou、shapeiou、piou、piou2
    2. Inner形式：
       inner_<base>（例如：inner_diou、inner_ciou、inner_siou）
    3. Focaler形式：
       focaler_<base>（例如：focaler_diou、focaler_ciou、focaler_siou）
    4. MPDIoU家族：
       mpdiou、inner_mpdiou、focaler_mpdiou
    5. WiseIoU家族：
       wiseiou（等价wiseiou_wiou）
       wiseiou_<variant>
       wiseiou_inner_<variant>
       wiseiou_focaler_<variant>
    6. wise <variant> 可选值：
       iou、wiou、giou、diou、ciou、eiou、siou、shapeiou、piou、piou2、mpdiou

- iou_aux（IoU辅助损失）

    1. none
    2. gcd
    3. nwd

## 更新公告

- 20260217

    1. 初版项目发布.
    2. 新增使用教程、模块改进使用教程视频.

- 20260228

    1. 新增常见的cls和iou的损失，并直接支持在train.py里面指定，并且在训练的时候会打印目前的loss.
    2. 对模型改进的yaml扩展到yolov8、yolov10、yolo11、yolo12.
    3. 新增在训练过程中mAP75输出.
    4. 优化detect.py中的特征图保存机制，使其可以单独保存每一个通道的特征图和总通道求和的特征图.
    5. 新增毕业必备-基于web的可视化界面，支持选择模型、检测图片、检测视频，显示目标数量等等功能
    6. 新增web界面的教程视频.
    7. 新增注册module的教程视频.
   
- 20260308

    1. 在val.py脚本中增加auto_coco_eval指标，支持一步到位计算COCO指标，不需要再人为转换标签和对齐标签的问题！
    2. 新增AAAI2026-SPJFB模块.
    3. 新增TGRS2025-GLSS2D模块.
    4. 新增TIP2025-CAFM模块.
    5. 新增TIP2025-DWM_MSA模块.
    6. 新增DynamicERF模块.
    7. 新增CSP、MetaFormer、Module在yaml中的使用教程-20260307补充版的视频.
    8. 修复用户反馈的bug.

- 20260315
    
    1. 新增CVPR2026-DEGConv模块。
    2. 新增CVPR2026-BinaryAttention模块。
    3. 新增CVPR2026-TransMixer模块。
    4. 新增CVPR2025-wca模块。
    5. 新增自研模块-DAF模块。
    6. 新增自研模块-CIDAF模块。
    7. 新增自研模块-WDAF模块。
    8. 新增Neck部分内容(ASF、BIFPN、CTrans、ERepBIFPN、GFPN、HSFPN、HS-FPN、超图FPN、SlimNeck、GoldYOLO、EMBSFPN)。
    9. 补全attention部分的配置文件。
    10. 新增conv、attention的内容如何与CSP模块随意组合的使用教程。
    11. 修复用户反馈的bug。

- 20260327
    1. 新增自研模块-EdgeLAWDS模块。
    2. 新增自研模块-FreqLAWDS模块。
    3. 新增自研模块-RouterLAWDS模块。
    4. 新增自研模块-FasterCGABlock模块。
    5. 新增自研金字塔-FDPN。
    6. 新增自研金字塔中的FDPN模块变种AGFDPN模块。
    7. 新增自研金字塔中的FDPN模块变种DFFDPN模块。
    8. 修复用户反馈的bug。
    9. 补齐了LSPCD的配置文件。
    10. 新增CVPR2026-Does YOLO Really Need to See Every Training Image in Every Epoch?的实现方法，此方法主要用于筛选简单和困难的样本，大部分情况下可以无损加速训练，并新增使用教程视频。

- 20260405
    1. 新增CVPR2026-sparse_state_space模块。
    2. 新增CVPR2026-sparse_mamba_block模块。
    3. 新增CVPR2026-MSInit模块。
    4. 新增CVPR2026-PFG模块。
    5. 新增CVPR2026-SFSFusion模块。
    6. 新增CVPR2026-LFP模块。
    7. 新增CVPR2026-FAAFusion模块。
    8.  新增LQE检测头。
    9.  优化LSPCD检测头。
    10. 新增LSPCD、LQE检测头的讲解视频。

- 20260417
    1. 新增TGRS2026-CGTA模块。
    2. 新增TGRS2026-LCGA模块。
    3. 新增自研模块-FSCGD模块。
    4. 新增自研模块-ADIE模块。
    5. 新增自研模块-AMSI模块。
    6. 新增自研模块-CMSI模块。
    7. 新增自研模块-CSIE模块。
    8. 新增自研模块-DPFGA模块。
    9. 新增自研模块-FMSI模块。
    10. 新增自研模块-HOIE模块。
    11. 新增自研模块-HPFGA模块。
    12. 新增PR2026-HAFFormer模块。
    13. 新增AAAI2026-CirculantAttention模块。
    14. 新增TGRS2025-DSEBlock模块。
    15. 新增TGRS2025-LaSEA模块。
    16. 新增CVPR2026-SFEB模块。
    17. 新增论文系列讲解视频-<用YOLO可以，但你的论文名字不能出现YOLO！>

- 20260505
  
    1. 新增自研模块-FSDA模块。
    2. 新增自研模块-ODALStem模块。
    3. 新增TIP2026-FourierSR模块。
    4. 新增自研模块-MCCG模块。
    5. 新增自研模块-DSRG模块。
    6. 新增CVPR2026-AFFN模块。
    7. 新增CVPR2026-WDAM模块。
    8. 新增自研模块-LowFrequencyFeatureFusion模块。
    9. 新增自研模块-DPAS模块。
    10. 新增自研模块-WGFS模块。
    11. 新增自研模块-MSCRM模块。
    12. 新增自研模块-DCGRM模块。
    13. 新增CVPR2026-FrequencyCM模块。
    14. 新增PR2026-A3FPN。
    15. 修复一些已知问题。
    16. 新增通用二次创新课程-PartialBlock，里面包含原理说明，论文怎么写，实验怎么设计，怎么改代码的流程。
    17. 新增论文系列讲解视频-论文精读｜ESWA2026-ETO-DFGR。