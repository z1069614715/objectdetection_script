# 2025-SOTA目标检测模型项目(2025发论文必备项目)

鉴于目前YOLO系列模型反映的拒稿率越来越高且YOLO模型确实非常泛滥，无论是不是计算机专业、是不是小白都基本可以快速上手YOLO模型，导致计算机专业和有期刊级别要求的小伙伴日益难受，简单来说就是YOLO在学术界的红利已经基本吃透，目前开始越来越多人转CVPR2024-RTDETR，而且目前研究生毕业一年比一年难，不像以前随便结合点深度学习就可以毕业，就像越来越多人反馈，导师已经明确禁止不能用YOLO，再加上这么多年来YOLO对学术的灌水已经让审稿人出现视觉疲劳，带上了”有色”眼镜看待YOLO，所以结合以上众多原因，因此我们需要一个有一定上手难度且是顶会的模型来支撑我们后续的大小论文的工作。
PS:20250614版本更新后，本项目的dfine和deim已经支持Ultralytics同款的配置文件形式，大大降低上手难度！[B站介绍链接](https://www.bilibili.com/video/BV1Q4MHzXEdd/)

### 1. 这个项目包含什么模型？

这个项目的源代码来自：[DEIM](https://github.com/ShihuaHuang95/DEIM)  
其内部可以跑以下模型(以下仅支持目标检测、不支持实例分割、姿态检测、旋转目标检测)：
1. CVPR2025-DEIM
2. ICLR2025-DFine
3. RTDETRV2

选择这个课程，这些模型都可以改进，不限于DEIM，这些都是顶会的模型，不要说2025，就算是2026都不落后！还有一个重点就是像CVPR2024-RTDETR，最小的模型也有50GFLOPs，但是现在的DEIM和DFine都有像YOLO一样的Nano大小版本的模型，变相降低了训练成本和设备要求！(建议最低12G显存的显卡起步)

### 2. 这个项目会以什么形式开展？

1. 这个项目跟以往区别比较大，我们其他改进项目都是直接提供好修改好的代码，用户不需要懂代码的情况下也可以开始做实验，甚至可以做完实验，但是这样也有一个不好的点，就是会大幅度降低上手门槛，这特别对计算机专业的同学来说是非常不利的，因此这个项目在代码工程方面，这个项目我们会有教程教大家怎么去调试程序、修改代码、添加模块。
2. 这个项目会**不定时(直播时间到时候会群里进行通知，没有硬性规定多久一次，不方便看的会有录播)**有**直播**，详细直播内容请看第三大点。
3. 这个项目会持续更新创新点，如果创新点是来源于现有的模型，还会提供对应的论文及其中文翻译版本（假设像FasterNet中的FasterBlock，会提供好对应的py文件、原论文及其中文翻译版本），用户可以根据从本课程学习到的缝合模块（代指第一点）去定制或者创新自己的网络。
4. 附带答疑群，答疑群主要答疑的内容是实验、代码操作、代码报错等相关问题(经过YOLO、RTDETR大量的经验，我没法保证每一个问题都能回复到大家，只能保证遇到过的问题会给大家提供建议和方向，当然群内的一些高频问题，我也会收集起来挑出部分出视频或者直播给大家进行解答)。
5. 如果后续有剪枝、蒸馏，不需要额外付费，本项目会包含在内，所以性价比真的非常高，YOLO改进剪枝蒸馏三件套也要200多了。

### 3. 直播内容

1. 解答群内一些高频疑问，比如很多人都会遇到的报错、或者注意点。
2. 教大家如何去做二次创新(PS:这个不是口头给大家说怎么二次创新，而是从代码的层面带大家去实践二次创新。可能这里会有同学问，那自研创新呢？你会自研模块的前提是必须要懂如何二次创新，首先这是一个过程，然后我有很多自研模块是突然有的想法或者看论文看到某些结构与之前看到的论文联合后有新的想法，所以也很难描述我为什么就想到这个结构，大多数情况下，只需要会有一定复杂度的二次创新就足够，当然自研模块有机会我也会去讲)
3. 给大家从浅到深解说一些我认为比较经典的模块，提高自己能创新新模块的能力和基础，因为很多模块都是相通的，本质没有变，只是模块上的组合体替换。(有不少人私聊我说，能不能出些你是如何结合一些现有的模块去创新的，虽然现在B站上也有不少讲创新点的，但是他们的感觉就是从头到尾读一篇代码，我看了几次之后觉得我把代码扔给GPT给我打上注释的感觉是一样的，看的时候感觉哦哦哦这样，看完后就不知所然)

### 3. 入手本项目需要注意些什么？

1. 因为本项目完全不是像之前YOLO项目这样傻瓜式操作，所以本项目有一定难度，具有以下特征的小伙伴不建议入手。（看到这里可能有人会问，为什么不考虑把DEIM、DFine、RTDETRV2都移植到Ultralytics？因为这个不确定性太大，DETR类型的模型对参数非常敏感，可能有一点参数不合适，效果就会大打折扣，但是对于这种较为复杂的模型移植过程中又很难保证一比一全过程移植） 
- 未入门、100%纯小白(如果你有心学，这个不是问题)
- 不太想花太多时间去学，搞这个只是想为了水个无要求的论文就行
- 没有任何解决问题的能力(如果你有心学，这个不是问题)
- 从来不看使用文档、说明之类的(强烈不建议入手)  
- 此项目上手需要时间，如果想无脑直接跑就不合适购入  
最后补充！如果你具有以上特征，但又要求期刊不能太水或者不能做yolo的问题，尽早入手CVPR2024-RTDETR吧，去年没抓上，今年不能再等了，模型红利可不等人。
2. 入手前可以先去B站看一下[CVPR025-DEIM合集里面的教程](https://space.bilibili.com/286900343/lists/4909499)，最起码先跑通过DEIM原始模型，能跟着视频训练和测试，然后也把合集里面的基础课程都先看一下，为后面打好基础。
3. 我认为这个不是什么不可达到的事，就看你想不想毕业了，有志者事竟成。
PS:20250614版本更新后，本项目的dfine和deim已经支持Ultralytics同款的配置文件形式，大大降低上手难度！[B站介绍链接](https://www.bilibili.com/video/BV1Q4MHzXEdd/)

### 4. 价格

1. 本项目价格为268，没有时效限制。（与其150、200买个YOLO纯模型改进专栏，不如268买个2025-SOTA专栏，最起码不用怕花了钱，最后做的YOLO还投不出去，还毕不了业）
2. 虚拟项目一经售出不退不换，需要入手前考虑清楚，如果你是初次入手我的项目，怕我不靠谱，可以先考虑入手个YOLO和RTDETR看下。

### 5. 项目使用问题

1. 购买本项目的使用者都会得到一个独一无二的用于解压7z的密码，到时候用于解压对应的压缩包，此密码自己妥善保管，请勿告诉他人。
2. 本项目的视频和直播回放统一都是加密视频，每个购买者都可以得到一个激活码，激活码在每个人专属的7z压缩文件内。

### 6. 项目更新公告

- 20250330

    1. 初版项目发布.

- 20250413

    1. 新增多个改进模块并新增模块简介，位置在engine/extre_module/module_images内。
    2. 新增训练和测试阶段的进度条显示。
    3. 优化tensorboard中的精度名称显示。
    4. 优化输出，把重要信息换颜色显示。
    5. 新增plot_train_batch_freq参数，用于控制间隔多少epoch保存第一个batch中的数据增强后的图像，默认为12。
    6. 新增保存当前参数信息，会自动保存到output_dir中的args.json文件内。
    7. 优化output_dir保存逻辑，当判断output_dir路径存在的时候，会自动在后缀加1，避免覆盖原先代码。

- 20250419

    1. 新增verbose_type参数，用于控制使用默认还是进度条输出，默认为官方默认输出形式。
    2. 新增thop计算模型计算量方式，避免calflops对于部分算子出现不支持报错的操作。
    3. 完善每个模块的py文件，增加输出计算量和参数量等数值，方便用户后续调试。
    4. 给DataLoader中添加pin_memory参数为True，可以在训练时候如果是数据加载成为瓶颈，可以提高速度。
    5. 修复用户反馈的已知问题。
    6. 新增多个改进模块。

- 20250429

    1. 修复engine/extre_module/custom_nn/attention/SEAM.py模块，应该是MutilSEAM。
    2. 新增一些进阶课程的视频。
    3. 新增多个改进模块。
    4. 修复用户反馈的已知问题。
    5. 修复续训时候会新增一个保存路径的问题。
    6. 修复多卡训练Stage2的时候会出现部分进程找不到权重文件的问题。

- 20250514

    1. 新增一些进阶课程的视频。
    2. 新增多个改进模块。
    3. 修复用户反馈的已知问题。

- 20250526

    1. 新增一些进阶课程的视频。
    2. 新增多个改进模块。
    3. 新增cache_ram参数，详细可以看userguide。
    4. 修复在torch2.7.0下出现的NotImplementedError问题。

- 20250609

    1. 修复新增了cache_ram功能后训练COCO数据集精度不正常的问题。
    2. 修复在训练COCO数据集中数据增强的绘制BUG。
    3. 新增多个改进模块。
    4. 新增一些进阶课程的视频。
    5. 修复用户反馈的已知问题。

- 20250614

    1. 新增Ultralytics的配置文件方式，大大降低改进难度。
    2. 新增一些<Ultralytics的配置文件方式>进阶课程的视频。
    3. 新增多个改进模块。

- 20250617

    1. 修复配置文件中层序号有误的问题。

- 20250619

    1. 修复配置文件中层序号有误的问题。
    2. 新增多个改进模块。
    3. 新增一些<Ultralytics的配置文件方式>进阶课程的视频。

- 20250625

    1. 修复best_stg2保存异常的问题。
    2. 新增YOLOV13中的HyperACE模块。
    3. 新增多个关于<Ultralytics的配置文件方式>进阶课程的视频。

- 20250705

    1. 新增多个改进模块。
    2. 新增多个关于<Ultralytics的配置文件方式>进阶课程的视频。
    3. 新增20250704基础疑问解答直播回放链接。

- 20250714

    1. 新增多个改进模块。
    2. 新增多个关于<Ultralytics的配置文件方式>进阶课程的视频。
    3. 新增小目标检测网络架构专题一群课题直播回放。

- 20250726

    1. 新增在test-only的状态下输出每个类别的'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'。
    2. 新增多个改进模块。
    3. 修复用户反馈的已知问题。
    4. 新增一个JSON格式数据集脚本。(输出类别数和类别id、输出每个类别的实例数量)

- 20250817

    1. 新增支持蒸馏学习，蒸馏学习支持断点续训使用方法跟正常训练一样。
    2. 蒸馏学习支持特征蒸馏、逻辑蒸馏、特征+逻辑蒸馏 这三种方式。
    3. 无论是Ultralytics配置文件方式、还是原始的代码方式都支持相互蒸馏。
    4. 蒸馏学习支持控制epoch，例如只有前50epoch进行蒸馏学习，后50epoch关闭蒸馏学习。
    5. 更多细节请看关于<知识蒸馏教学视频>的进阶课程。
    6. 支持输出YOLO指标(Precision、Recall、F1-Score、mAP50、mAP75、mAP50-95)，详细请看userguide。
    7. 新增多个改进模块。
    8. 新增小目标检测网络架构专题二链接。

- 20250823

    1. 修复YOLO指标在一些图片没真实标签的时候报错的bug。
    2. 开放逻辑蒸馏，在项目内有对应的课程。
    3. 新增多个改进模块。
    4. 新增<知识蒸馏教学视频>的进阶课程。

- 20250907

    1. 新增多个改进模块。
    2. 修复蒸馏学习中教师信息输出错误的问题。

### 7. 进阶视频教程

------------------------------------- 进阶教程 -------------------------------------  

-------------------------------------------- 基础疑问直播回放和一些功能性文件的使用教程 <这部分没有观看顺序> --------------------------------------------  
其中基础疑问解答系列的视频可以在群文件中的群在线文档查看这期讲了什么问题。
1. 20250402基础疑问解答直播回放
2. 20250416基础疑问解答直播回放
3. featuremap脚本使用教程
4. heatmap脚本使用教程
5. tools/inference/torch_inf.py
6. 20250513基础疑问解答直播回放链接
7. 20250618基础疑问解答直播回放链接
8. 20250704基础疑问解答直播回放链接
9. 小目标检测网络架构专题一群课题直播回放链接
10. 小目标检测网络架构专题二

-------------------------------------------- 基础教程补充版 (这部分建议理解完基础课程全部再看) --------------------------------------------  
1. 如何快速得知每个改进模块的输入输出格式相关信息
2. 这个项目内的通道数是怎么传递的？特别是backbone与encoder部分，对后续的改进很重要！
3. 主干进阶改进方案一-给每个stage设定不同的参数
4. 主干进阶改进方案二-给每个stage使用不同的改进结构
5. 改进模型后参数量和计算量变得非常大怎么办？为什么会这样？怎么解决？

-------------------------------------------- 特殊配置文件-进阶教程(这部分必须要看完理解完基础课程全部才能看，不然百分百不看懂) <这部分没有观看顺序>  --------------------------------------------  
1. engine/extre_module/custom_nn/featurefusion/mpca.py
2. 如何使用engine/extre_module/custom_nn/transformer改进HybridEncoder中的Transformer
3. engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程-上集
3. engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程-中集
4. engine/extre_module/custom_nn/module.py的搭积木神器(万物皆可融)教程-下集
5. engine/extre_module/custom_nn/mlp/SEFN.py
6. engine/extre_module/custom_nn/neck/FDPN.py

-------------------------------------------- Ultralytics配置文件版本教程 <即使你使用配置文件方式去跑也建议B站的基础视频和项目内的其他视频都看一下，对后面做二次创新时候的代码有帮助> --------------------------------------------
1. 原始配置文件讲解
2. 怎么使用预训练权重
3. Ultralytics配置文件版本的热力图和特征图脚本使用教程
4. 注册模块示例教程一
5. 注册模块示例教程二(Conv、attention部分)
6. 注册模块示例教程三(module、block部分)
7. 注册模块示例教程四(搭积木神器(万物皆可融)部分)
8. 注册模块示例教程五(HyperACE)
9. 注册模块示例教程六(featurefusion部分)
10. 自研模块FDPN在配置文件中的实现讲解
11. YOLO13中的HyperACE在DEIM中的应用
12. 注册模块示例教程七(transformer部分)
13. 注册模块示例教程八(GOLD-YOLO)
14. GOLO-YOLO在DEIM中的应用(包含怎么对其二次创新)
15. HS-FPN怎么融合到DEIM？
16. 怎么用module、block中的模块去改进主干网络？
17. 从0搭建一个yaml！以CVPR2025-nnWNet为例.

-------------------------------------------- 知识蒸馏 --------------------------------------------
1. 知识蒸馏原理讲解
2. 知识蒸馏使用教程
3. 逻辑蒸馏的讲解1-DETR检测头大白话讲解
4. 逻辑蒸馏的讲解2-逻辑蒸馏的讲解
5. 一些关于蒸馏的注意点(蒸馏实验前必看)
6. 蒸馏中出现教师模型和配置文件一直显示不匹配的问题解决思路

### 8. 目前已有的模块

- engine/extre_module/custom_nn/attention 

    1. engine/extre_module/custom_nn/attention/SEAM.py
    2. CVPR2021|engine/extre_module/custom_nn/attention/ca.py
    3. ICASSP2023|engine/extre_module/custom_nn/attention/ema.py
    4. ICML2021|engine/extre_module/custom_nn/attention/simam.py
    5. ICCV2023|engine/extre_module/custom_nn/attention/lsk.py
    6. WACV2024|engine/extre_module/custom_nn/attention/DeformableLKA.py
    7. engine/extre_module/custom_nn/attention/mlca.py
    8. BIBM2024|engine/extre_module/custom_nn/attention/FSA.py
    9. AAAI2025|engine/extre_module/custom_nn/attention/CDFA.py
    10. engine/extre_module/custom_nn/attention/GLSA.py
    11. TGRS2025|engine/extre_module/custom_nn/attention/MCA.py
    12. CVPR2025|engine/extre_module/custom_nn/attention/CASAB.py 

- engine/extre_module/custom_nn/block

    1. engine/extre_module/custom_nn/block/RepHMS.py
    2. 自研模块|engine/extre_module/custom_nn/block/rgcspelan.py
    3. TPAMI2025|engine/extre_module/custom_nn/block/MANet.py

- engine/extre_module/custom_nn/conv_module

    1. CVPR2021|engine/extre_module/custom_nn/conv_module/dbb.py
    2. IEEETIP2024|engine/extre_module/custom_nn/conv_module/deconv.py
    3. ICCV2023|engine/extre_module/custom_nn/conv_module/dynamic_snake_conv.py
    4. CVPR2023|engine/extre_module/custom_nn/conv_module/pconv.py
    5. AAAI2025|engine/extre_module/custom_nn/conv_module/psconv.py
    6. CVPR2025|engine/extre_module/custom_nn/conv_module/ShiftwiseConv.py
    7. engine/extre_module/custom_nn/conv_module/wdbb.py
    8. engine/extre_module/custom_nn/conv_module/deepdbb.py
    9. ECCV2024|engine/extre_module/custom_nn/conv_module/wtconv2d.py
    10. CVPR2023|engine/extre_module/custom_nn/conv_module/ScConv.py
    11. engine/extre_module/custom_nn/conv_module/dcnv2.py
    12. CVPR2024|engine/extre_module/custom_nn/conv_module/DilatedReparamConv.py
    13. engine/extre_module/custom_nn/conv_module/gConv.py
    14. CVPR2024|engine/extre_module/custom_nn/conv_module/IDWC.py
    15. engine/extre_module/custom_nn/conv_module/DSA.py
    16. CVPR2025|engine/extre_module/custom_nn/conv_module/FDConv.py
    17. CVPR2023|engine/extre_module/custom_nn/conv_module/dcnv3.py
    18. CVPR2024|engine/extre_module/custom_nn/conv_module/dcnv4.py
    19. CVPR2024|engine/extre_module/custom_nn/conv_module/DynamicConv.py
    20. CVPR2024|engine/extre_module/custom_nn/conv_module/FADC.py
    21. CVPR2023|engine/extre_module/custom_nn/conv_module/SMPConv.py
    22. MIA2025|engine/extre_module/custom_nn/conv_module/FourierConv.py
    23. CVPR2024|engine/extre_module/custom_nn/conv_module/SFSConv.py
    24. ICCV2025|engine/extre_module/custom_nn/conv_module/MBRConv.py
    25. ICCV2025|engine/extre_module/custom_nn/conv_module/ConvAttn.py
    26. ICCV2025|engine/extre_module/custom_nn/conv_module/Converse2D.py
    27. CVPR2025|engine/extre_module/custom_nn/conv_module/gcconv.py

- engine/extre_module/custom_nn/upsample

    1. CVPR2024|engine/extre_module/custom_nn/upsample/eucb.py
    2. CVPR2024|engine/extre_module/custom_nn/upsample/eucb_sc.py
    3. engine/extre_module/custom_nn/upsample/WaveletUnPool.py
    4. ICCV2019|engine/extre_module/custom_nn/upsample/CARAFE.py
    5. ICCV2023|engine/extre_module/custom_nn/upsample/DySample.py
    6. ICCV2025|engine/extre_module/custom_nn/upsample/Converse2D_Up.py
    7. CVPR2025|engine/extre_module/custom_nn/upsample/DSUB.py

- engine/extre_module/custom_nn/downsample

    1. IEEETIP2020|engine/extre_module/custom_nn/downsample/gcnet.py
    2. 自研模块|engine/extre_module/custom_nn/downsample/lawds.py 
    3. engine/extre_module/custom_nn/downsample/WaveletPool.py
    4. engine/extre_module/custom_nn/downsample/ADown.py
    5. engine/extre_module/custom_nn/downsample/YOLOV7Down.py
    6. engine/extre_module/custom_nn/downsample/SPDConv.py
    7. engine/extre_module/custom_nn/downsample/HWD.py
    8. engine/extre_module/custom_nn/downsample/DRFD.py

- engine/extre_module/custom_nn/stem

    1. engine/extre_module/custom_nn/stem/SRFD.py
    2. engine/extre_module/custom_nn/stem/LoG.py
    3. ICCV2023|engine/extre_module/custom_nn/stem/RepStem.py

- engine/extre_module/custom_nn/featurefusion

    1. 自研模块|engine/extre_module/custom_nn/featurefusion/cgfm.py
    2. BMVC2024|engine/extre_module/custom_nn/featurefusion/msga.py
    3. CVPR2024|engine/extre_module/custom_nn/featurefusion/mfm.py
    4. IEEETIP2023|engine/extre_module/custom_nn/featurefusion/CSFCN.py
    5. BIBM2024|engine/extre_module/custom_nn/featurefusion/mpca.py
    6. ACMMM2024|engine/extre_module/custom_nn/featurefusion/wfu.py
    7. CVPR2025|engine/extre_module/custom_nn/featurefusion/GDSAFusion.py
    8. engine/extre_module/custom_nn/featurefusion/PST.py
    9. TGRS2025|engine/extre_module/custom_nn/featurefusion/MSAM.py
    10. INFFUS2025|engine/extre_module/custom_nn/featurefusion/DPCF.py

- engine/extre_module/custom_nn/module

    1. AAAI2025|engine/extre_module/custom_nn/module/APBottleneck.py
    2. CVPR2025|engine/extre_module/custom_nn/module/efficientVIM.py
    3. CVPR2023|engine/extre_module/custom_nn/module/fasterblock.py
    4. CVPR2024|engine/extre_module/custom_nn/module/starblock.py
    5. engine/extre_module/custom_nn/module/DWR.py
    6. CVPR2024|engine/extre_module/custom_nn/module/UniRepLKBlock.py
    7. CVPR2025|engine/extre_module/custom_nn/module/mambaout.py
    8. AAAI2024|engine/extre_module/custom_nn/module/DynamicFilter.py
    9. engine/extre_module/custom_nn/module/StripBlock.py
    10. IEEETGRS2024|engine/extre_module/custom_nn/module/elgca.py
    11. CVPR2024|engine/extre_module/custom_nn/module/LEGM.py
    12. ICCV2023|engine/extre_module/custom_nn/module/iRMB.py
    13. TPAMI2025|engine/extre_module/custom_nn/module/MSBlock.py
    14. ICLR2024|engine/extre_module/custom_nn/module/FATBlock.py
    15. CVPR2024|engine/extre_module/custom_nn/module/MSCB.py
    16. engine/extre_module/custom_nn/module/LEGBlock.py
    17. CVPR2025|engine/extre_module/custom_nn/module/RCB.py
    18. ECCV2024|engine/extre_module/custom_nn/module/JDPM.py
    19. CVPR2025|engine/extre_module/custom_nn/module/vHeat.py
    20. CVPR2025|engine/extre_module/custom_nn/module/EBlock.py
    21. CVPR2025|engine/extre_module/custom_nn/module/DBlock.py
    22. ECCV2024|engine/extre_module/custom_nn/module/FMB.py
    23. CVPR2024|engine/extre_module/custom_nn/module/IDWB.py
    24. ECCV2022|engine/extre_module/custom_nn/module/LFE.py
    25. AAAI2025|engine/extre_module/custom_nn/module/FCM.py
    26. CVPR2024|engine/extre_module/custom_nn/module/RepViTBlock.py
    27. CVPR2024|engine/extre_module/custom_nn/module/PKIModule.py
    28. CVPR2024|engine/extre_module/custom_nn/module/camixer.py
    29. ICCV2025|engine/extre_module/custom_nn/module/ESC.py
    30. CVPR2025|engine/extre_module/custom_nn/module/nnWNet.py
    31. TGRS2025|engine/extre_module/custom_nn/module/ARF.py
    32. AAAI2024|engine/extre_module/custom_nn/module/CFBlock.py

- engine/extre_module/custom_nn/neck

    1. 自研模块|engine/extre_module/custom_nn/neck/FDPN.py

- engine/extre_module/custom_nn/neck_module

    1. TPAMI2025|engine/extre_module/custom_nn/neck_module/HyperCompute.py
    2. engine/extre_module/custom_nn/neck_module/HyperACE.py
    3. engine/extre_module/custom_nn/neck_module/GoldYOLO.py
    4. AAAI2025|engine/extre_module/custom_nn/neck_module/HS_FPN.py

- engine/extre_module/custom_nn/norm

    1. ICML2024|engine/extre_module/custom_nn/transformer/repbn.py
    2. CVPR2025|engine/extre_module/custom_nn/transformer/dyt.py

- engine/extre_module/custom_nn/transformer

    1. ICLR2025|engine/extre_module/custom_nn/transformer/PolaLinearAttention.py
    2. CVPR2023|engine/extre_module/custom_nn/transformer/biformer.py
    3. CVPR2023|engine/extre_module/custom_nn/transformer/CascadedGroupAttention.py
    4. CVPR2022|engine/extre_module/custom_nn/transformer/DAttention.py
    5. ICLR2022|engine/extre_module/custom_nn/transformer/DPBAttention.py
    6. CVPR2024|engine/extre_module/custom_nn/transformer/AdaptiveSparseSA.py
    7. engine/extre_module/custom_nn/transformer/GSA.py
    8. engine/extre_module/custom_nn/transformer/RSA.py
    9. ECCV2024|engine/extre_module/custom_nn/transformer/FSSA.py
    10. AAAI2025|engine/extre_module/custom_nn/transformer/DilatedGCSA.py
    11. AAAI2025|engine/extre_module/custom_nn/transformer/DilatedMWSA.py
    12. CVPR2024|engine/extre_module/custom_nn/transformer/SHSA.py
    13. IJCAI2024|engine/extre_module/custom_nn/transformer/CTA.py
    13. IJCAI2024|engine/extre_module/custom_nn/transformer/SFA.py
    14. engine/extre_module/custom_nn/transformer/MSLA.py
    15. ACMMM2025|engine/extre_module/custom_nn/transformer/CPIA_SA.py

- engine/extre_module/custom_nn/mlp

    1. CVPR2024|engine/extre_module/custom_nn/mlp/ConvolutionalGLU.py
    2. IJCAI2024|engine/extre_module/custom_nn/mlp/DFFN.py
    3. ICLR2024|engine/extre_module/custom_nn/mlp/FMFFN.py
    4. CVPR2024|engine/extre_module/custom_nn/mlp/FRFN.py
    5. ECCV2024|engine/extre_module/custom_nn/mlp/EFFN.py 
    6. WACV2025|engine/extre_module/custom_nn/mlp/SEFN.py
    7. ICLR2025|engine/extre_module/custom_nn/mlp/KAN.py
    8. CVPR2025|engine/extre_module/custom_nn/mlp/EDFFN.py

- engine/extre_module/custom_nn/mamba

    1. AAAI2025|engine/extre_module/custom_nn/mamba/SS2D.py
    2. CVPR2025|engine/extre_module/custom_nn/mamba/ASSM.py
    3. CVPR2025|engine/extre_module/custom_nn/mamba/SAVSS.py
    4. CVPR2025|engine/extre_module/custom_nn/mamba/MobileMamba/mobilemamba.py
    5. CVPR2025|engine/extre_module/custom_nn/mamba/MaIR.py
    6. TGRS2025|engine/extre_module/custom_nn/mamba/GLVSS.py
    7. ICCV2025|engine/extre_module/custom_nn/mamba/VSSD.py
    8. ICCV2025|engine/extre_module/custom_nn/mamba/TinyViM.py
    9. INFFUS2025|engine/extre_module/custom_nn/mamba/CSI.py

- 积木模块,示例教程engine/extre_module/custom_nn/module/example.py

    1. YOLOV5|C3
    2. YOLOV8|C2f
    3. YOLO11|C3k2
    4. TPAMI2025|MANet
    5. TPAMI2024|MetaFormer_Block
    6. TPAMI2024+CVPR2025|MetaFormer_Mona
    7. TPAMI2024+CVPR2025+WACV2025|MetaFormer_SEFN
    8. TPAMI2024+CVPR2025+WACV2025|MetaFormer_Mona_SEFN