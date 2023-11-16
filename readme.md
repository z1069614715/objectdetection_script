# Object Detection Script
这个项目主要是提供一些关于目标检测的代码和改进思路参考.

# Project <需要入手请加企鹅1615905974/1069614715,如添加不上可bilibili私聊直发企鹅号码,最好好友请求也设置不需要验证就可以加上>
1. 基于整合yolov7的项目.(12.9¥)[项目详解](https://t.bilibili.com/798697826625257472#reply245141740)  

        YOLOV5-Backbone中的主干基本都支持到我个人整合的v7代码中,并且整合后的v7配置文件非常简洁,方便进行改进
        购买了本项目联系我qq,我会拉你进一个交流群,详细可看[哔哩哔哩链接](https://www.bilibili.com/opus/798697826625257472?spm_id_from=333.999.0.0)

2. 基于整合yolov8的项目.(目前69.9¥)
    
    [目前已有的改进方案和更新详细公告](https://github.com/z1069614715/objectdetection_script/blob/master/yolo-improve/yolov8-project.md)  
    项目简单介绍，详情请看项目详解.
    1. 提供修改好的代码和每个改进点的配置文件,相当于积木都给大家准备好,大家只需要做实验和搭积木(修改yaml配置文件组合创新点)即可,装好环境即可使用.
    2. 后续的改进方案都会基于这个项目更新进行发布，在群公告进行更新百度云链接.
    3. 购买了yolov8项目的都会赠送yolov5-PAGCP通道剪枝算法代码和相关实验参数命令.
    4. 购买后进YOLOV8交流群(代码视频均在群公告),群里可交流代码和论文相关,目前1900+人,气氛活跃.
    5. 项目因为(价格问题)不附带一对一私人答疑服务,平时私聊问点小问题和群里的问题,有空我都会回复.

3. 基于YOLOV5,YOLOV7的(剪枝+知识蒸馏)项目.(129.9¥)[项目详解](https://github.com/z1069614715/objectdetection_script/blob/master/yolo-improve/yolov5v7-light.md)

    1. 模型轻量化,部署必备之一!
    2. 项目里面配套几个剪枝和蒸馏的示例,并且都配有视频讲解,供大家理解如何进行剪枝和蒸馏.
    3. 购买后进YOLOV5V7轻量化交流群(代码视频均在群公告),轻量化问题都可在群交流,因为剪枝问题比较困难,所以剪枝蒸馏问题可以群里提问,我都会群里回复相关问题.

4. 基于Ultralytics的RT-DETR改进项目.(89.9¥)

    [目前已有的改进方案和更新详细公告](https://github.com/z1069614715/objectdetection_script/blob/master/yolo-improve/rtdetr-project.md)  
    项目简单介绍，详情请看项目详解.
    1. 提供修改好的代码和每个改进点的配置文件,相当于积木都给大家准备好,大家只需要做实验和搭积木(修改yaml配置文件组合创新点)即可,装好环境即可使用.
    2. 后续的改进方案都会基于这个项目更新进行发布,在群公告进行更新百度云链接.
    3. 购买了RT-DETR项目的都会赠送yolov5-PAGCP通道剪枝算法代码和相关实验参数命令.
    4. 购买后进RT-DETR交流群(代码视频均在群公告),群里可交流代码和论文相关.
    5. 项目因为(价格问题)不附带一对一私人答疑服务,平时私聊问点小问题和群里的问题,有空我都会回复.
    6. RT-DETR项目包含多种基准模型改进方案(RT-DETR-R18,RT-DETR-L,Yolov8-Detr,Yolov5-Detr),具体可点击[目前已有的改进方案和更新详细公告](https://github.com/z1069614715/objectdetection_script/blob/master/yolo-improve/rtdetr-project.md)看详细.

# Advertising Board
人工智能-工作室长期对外接单，范围主要是:
1. 目标检测.
2. 图像分类.
3. 图像分割.
4. NLP领域.
5. 图像超分辨.
6. 图像去噪.
7. GAN.
8. 模型部署.
9. 模型创新. 
10. 目标跟踪.

等等. 价格公道,三年TB老店,TB付款安全可靠.  
有需要的可加企鹅 1615905974详聊.  
另外也有有偿指导答疑.  

# Explanation
- **yolo**  
    yolo文件夹是针对yolov5,yolov7,yolov8的数据集处理脚本，具体可看[readme.md](https://github.com/z1069614715/objectdetection_script/blob/master/yolo/readme.md).  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1tM411a7it/).  

- **damo-yolo**  
    damo-yolo文件夹是针对DAMO-YOLO的数据集处理脚本，具体可看[readme.md](https://github.com/z1069614715/objectdetection_script/blob/master/damo-yolo/readme.md).  
    目前只支持voc转coco.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1M24y1v7Uf/).   

- **yolo-improve**  
    yolo-improve文件夹是提供一些关于yolo系列模型改进思路的源码，具体可看[readme.md](https://github.com/z1069614715/objectdetection_script/blob/master/yolo-improve/readme.md).   

- **yolo-gradcam**  
    yolo-gradcam文件夹是提供一些关于可视化yolo模型的热力图的源码，具体可看[readme.md](https://github.com/z1069614715/objectdetection_script/blob/master/yolo-gradcam/README.md).

- **cv-attention**  
    cv-attention文件夹是关于CV的一些经典注意力机制，具体可看[readme.md](https://github.com/z1069614715/objectdetection_script/blob/master/cv-attention/readme.md).

- **objectdetection-tricks**  
    objectdetection-tricks文件夹是关于目标检测中各种小技巧，具体可看[readme.md](https://github.com/z1069614715/objectdetection_script/blob/master/objectdetection-tricks/readme.md).
    
[![Forkers repo roster for @z1069614715/objectdetection_script](https://reporoster.com/forks/z1069614715/objectdetection_script)](https://github.com/z1069614715/objectdetection_script/network/members)
[![Stargazers repo roster for @z1069614715/objectdetection_script](https://reporoster.com/stars/z1069614715/objectdetection_script)](https://github.com/z1069614715/objectdetection_script/stargazers)

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=z1069614715/objectdetection_script&type=Date)](https://star-history.com/#z1069614715/objectdetection_script&Date)

<a id="0"></a>

# Alipay(需要购买直接加我QQ:1615905974(添加不上请在B站上私信发我qq号))
<!-- ![Alipay](images/ZFB.jpg)   -->