# Ultralytics-YOLO项目详细说明

1. 本项目集成了YOLOv8、v10、v11、v12乃至前沿的YOLO26等全系列基础模型。 无论是做横向对比实验，还是纵向的版本改进，无需到处找资源，一个项目就能满足你所有的实验需求！
2. 核心代码已实现高度模块化与解耦，专为新手优化。 你完全不需要死磕底层复杂代码，只需像搭积木一样简单修改YAML配置文件，就能轻松实现各种改进模块的自由组合。
3. 面对日益内卷的YOLO赛道，简单的“缝合”已难满足毕业要求。 本项目不仅提供现成的创新方案，更配套独家“二次创新”课程，授人以渔。我们将手把手教你掌握模块设计的底层逻辑，助你从“模仿者”进阶为“创造者”，设计出独属于你的创新模块。
4. 针对有代码基础但受困于Ultralytics复杂架构的同学， 本项目引入了来自DFine、DEIM项目中成熟的“万物皆可融”架构思想。你无需纠结模块注册等信息，只需遵循我所提供的标准接口规范，即可将自定义魔改模块无缝融入YAML配置，与各类CSP变种灵活结合。
5. 实验跑通了，却不知道如何写创新点？ 本项目将定期拆解高分论文，传授写作心法，教你如何将实验成果转化为逻辑严密、亮点突出的高质量学术论文，解决写作难题！
6. 毕业设计缺少高大上的展示界面？ 别担心，项目会内置基于PyQt或HTML的通用可视化界面，开箱即用，完美补齐毕业论文的最后一块拼图，助你从容应对答辩！
7. 购买即享专属技术交流群， 这里有业内公认的高效答疑服务，以及志同道合的伙伴互助交流。拒绝闭门造车，让我们带你避开深坑，高效通关！  

## 针对于已经入手了yolov8/yolo11项目的同学来说，如果你有以下几点需求，可以考虑追加入手！
1. 想用最新的YOLO26做实验！而且本项目支持v8、v10、11、12、26全系列版本！
2. 想深入学习改进创新的同学，本项目会附带二次创新的通用教程，手把手教你设计出属于自己的创新模块！
3. 做完实验不知道怎么写论文？本项目会定期拆解高分论文案例，教你如何把实验结果写成逻辑清晰、亮点突出的高质量学术论文
4. 想自己魔改模块的同学！本项目提供超级简单的模块注册方式，只需按照教程操作，就能轻松注册自己的模块，还能和各种CSP变种随意组合！

## 模块列表

- ultralytics/nn/extra_modules/attention 

    1. ultralytics/nn/extra_modules/attention/SEAM.py
    2. CVPR2021|ultralytics/nn/extra_modules/attention/ca.py
    3. ICASSP2023|ultralytics/nn/extra_modules/attention/ema.py
    4. ICML2021|ultralytics/nn/extra_modules/attention/simam.py
    5. ICCV2023|ultralytics/nn/extra_modules/attention/lsk.py
    6. WACV2024|ultralytics/nn/extra_modules/attention/DeformableLKA.py
    7. ultralytics/nn/extra_modules/attention/mlca.py
    8. BIBM2024|ultralytics/nn/extra_modules/attention/FSA.py
    9. AAAI2025|ultralytics/nn/extra_modules/attention/CDFA.py
    10. ultralytics/nn/extra_modules/attention/GLSA.py
    11. TGRS2025|ultralytics/nn/extra_modules/attention/MCA.py
    12. CVPR2025|ultralytics/nn/extra_modules/attention/CASAB.py 
    13. NN2025|ultralytics/nn/extra_modules/attention/KSFA.py
    14. TPAMI2025|ultralytics/nn/extra_modules/attention/GQL.py
    15. TGRS2025|ultralytics/nn/extra_modules/attention/ACA.py
    16. TGRS2025|ultralytics/nn/extra_modules/attention/DHPF.py
    17. TGRS2025|ultralytics/nn/extra_modules/attention/ACAB.py

- ultralytics/nn/extra_modules/conv_module

    1. CVPR2021|ultralytics/nn/extra_modules/conv_module/dbb.py
    2. IEEETIP2024|ultralytics/nn/extra_modules/conv_module/deconv.py
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
    24. ICCV2025|ultralytics/nn/extra_modules/conv_module/MBRConv.py
    25. ICCV2025|ultralytics/nn/extra_modules/conv_module/ConvAttn.py
    26. ICCV2025|ultralytics/nn/extra_modules/conv_module/Converse2D.py
    27. CVPR2025|ultralytics/nn/extra_modules/conv_module/gcconv.py
    28. ACCV2024|ultralytics/nn/extra_modules/conv_module/RMBC.py

- engine/extre_module/custom_nn/stem

    1. ultralytics/nn/extra_modules/stem/SRFD.py
    2. ultralytics/nn/extra_modules/stem/LoG.py
    3. ICCV2023|ultralytics/nn/extra_modules/stem/RepStem.py

- ultralytics/nn/extra_modules/upsample

    1. CVPR2024|ultralytics/nn/extra_modules/upsample/eucb.py
    2. CVPR2024|ultralytics/nn/extra_modules/upsample/eucb_sc.py
    3. ultralytics/nn/extra_modules/upsample/WaveletUnPool.py
    4. ICCV2019|ultralytics/nn/extra_modules/upsample/CARAFE.py
    5. ICCV2023|ultralytics/nn/extra_modules/upsample/DySample.py
    6. ICCV2025|ultralytics/nn/extra_modules/upsample/Converse2D_Up.py
    7. CVPR2025|ultralytics/nn/extra_modules/upsample/DSUB.py

- ultralytics/nn/extra_modules/downsample

    1. IEEETIP2020|ultralytics/nn/extra_modules/downsample/gcnet.py
    2. 自研模块|ultralytics/nn/extra_modules/downsample/lawds.py 
    3. ultralytics/nn/extra_modules/downsample/WaveletPool.py
    4. ultralytics/nn/extra_modules/downsample/ADown.py
    5. ultralytics/nn/extra_modules/downsample/YOLOV7Down.py
    6. ultralytics/nn/extra_modules/downsample/SPDConv.py
    7. ultralytics/nn/extra_modules/downsample/HWD.py
    8. ultralytics/nn/extra_modules/downsample/DRFD.py
    9. TGRS2025|ultralytics/nn/extra_modules/conv_module/FSConv.py

- ultralytics/nn/extra_modules/module

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
    17. CVPR2025|ultralytics/nn/extra_modules/module/RCB.py
    18. ECCV2024|ultralytics/nn/extra_modules/module/JDPM.py
    19. CVPR2025|ultralytics/nn/extra_modules/module/vHeat.py
    20. CVPR2025|ultralytics/nn/extra_modules/module/EBlock.py
    21. CVPR2025|ultralytics/nn/extra_modules/module/DBlock.py
    22. ECCV2024|ultralytics/nn/extra_modules/module/FMB.py
    23. CVPR2024|ultralytics/nn/extra_modules/module/IDWB.py
    24. ECCV2022|ultralytics/nn/extra_modules/module/LFE.py
    25. AAAI2025|ultralytics/nn/extra_modules/module/FCM.py
    26. CVPR2024|ultralytics/nn/extra_modules/module/RepViTBlock.py
    27. CVPR2024|ultralytics/nn/extra_modules/module/PKIModule.py
    28. CVPR2024|ultralytics/nn/extra_modules/module/camixer.py
    29. ICCV2025|ultralytics/nn/extra_modules/module/ESC.py
    30. CVPR2025|ultralytics/nn/extra_modules/module/nnWNet.py
    31. TGRS2025|ultralytics/nn/extra_modules/module/ARF.py
    32. AAAI2024|ultralytics/nn/extra_modules/module/CFBlock.py
    33. IJCV2024|ultralytics/nn/extra_modules/module/FMA.py
    34. ultralytics/nn/extra_modules/module/LWGA.py
    35. TGRS2025|ultralytics/nn/extra_modules/module/CSSC.py
    35. TGRS2025|ultralytics/nn/extra_modules/module/CNCM.py
    36. ICCV2025|ultralytics/nn/extra_modules/module/HFRB.py
    37. ICIP2025|ultralytics/nn/extra_modules/module/EVA.py
    38. CVPR2025|ultralytics/nn/extra_modules/module/IEL.py
    39. MICCAI2023|ultralytics/nn/extra_modules/module/MFEBlock.py
    40. AAAI2026|ultralytics/nn/extra_modules/module/PartialNetBlock.py
    41. TGRS2025|ultralytics/nn/extra_modules/module/DRG.py
    42. ultralytics/nn/extra_modules/module/Wave2D.py
    43. TGRS2025|ultralytics/nn/extra_modules/module/GLGM.py
    44. TGRS2025|ultralytics/nn/extra_modules/module/MAC.py

- ultralytics/nn/extra_modules/block 
    
    1. ultralytics/nn/extra_modules/block/CSPBlock.py
    2. TPAMI2025|ultralytics/nn/extra_modules/block/MANet.py
    3. TPAMI2024|ultralytics/nn/extra_modules/block/MetaFormer.py

- ultralytics/nn/extra_modules/transformer

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
    13. IJCAI2024|ultralytics/nn/extra_modules/transformer/SFA.py
    14. ultralytics/nn/extra_modules/transformer/MSLA.py
    15. ACMMM2025|ultralytics/nn/extra_modules/transformer/CPIA_SA.py
    16. NN2025|ultralytics/nn/extra_modules/transformer/TokenSelectAttention.py
    17. CVPR2025|ultralytics/nn/extra_modules/transformer/TAB.py
    19. TPAMI2025|ultralytics/nn/extra_modules/transformer/LRSA.py
    20. ICCV2025|ultralytics/nn/extra_modules/transformer/MALA.py
    21. ICML2023|ultralytics/nn/extra_modules/transformer/MUA.py
    22. ACMMM2025|ultralytics/nn/extra_modules/transformer/EGSA.py
    23. ACMMM2025|ultralytics/nn/extra_modules/transformer/SWSA.py
    24. AAAI2026|ultralytics/nn/extra_modules/transformer/DHOGSA.py
    25. NeurIPS2025|ultralytics/nn/extra_modules/transformer/CBSA.py
    26. TGRS2025|ultralytics/nn/extra_modules/transformer/DPWA.py

- ultralytics/nn/extra_modules/mlp

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

- ultralytics/nn/extra_modules/featurefusion

    1. 自研模块|ultralytics/nn/extra_modules/featurefusion/cgfm.py
    2. BMVC2024|ultralytics/nn/extra_modules/featurefusion/msga.py
    3. CVPR2024|ultralytics/nn/extra_modules/featurefusion/mfm.py
    4. IEEETIP2023|ultralytics/nn/extra_modules/featurefusion/CSFCN.py
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

- ultralytics/nn/extra_modules/featurepreprocess

    1. TGRS2025|ultralytics/nn/extra_modules/featurepreprocess/FAENet.py

- ultralytics/nn/extra_modules/head(ultralytics/cfg/models/improve/head)

    1. ultralytics/nn/extra_modules/head/LSPCD.py

## 更新公告

- 20260217

    1. 初版项目发布.
    2. 新增使用教程、模块改进使用教程视频.