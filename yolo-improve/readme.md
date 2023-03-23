# YOLO-Improve
这个项目主要是提供一些关于yolo系列模型的改进思路，效果因数据集和参数而异，仅作参考。

# Explanation
- **iou**  
    添加EIOU，SIOU，ALPHA-IOU, FocalEIOU, Wise-IOU到yolov5,yolov8的box_iou中.  
    1. yolov5
        视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1KM411b7Sz/).  
        博客地址：[CSDN](https://blog.csdn.net/qq_37706472/article/details/128737484?spm=1001.2014.3001.5501).

        #### 2023-2-8 更新: 新增[Wise-IoU](https://arxiv.org/abs/2301.10051) 视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1tG4y1N7Gk/). reference:[github](https://github.com/Instinct323/wiou)  
    2. yolov8
        视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1PY4y1o7Hm/).  
        博客地址：[CSDN](https://blog.csdn.net/qq_37706472/article/details/128743012?spm=1001.2014.3001.5502).

        #### 2023-2-7 更新: 新增[Wise-IoU](https://arxiv.org/abs/2301.10051) 视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1De4y1N7Mb/). reference:[github](https://github.com/Instinct323/wiou)   
- **yolov5-GFPN**   
    使用DAMO-YOLO中的GFPN替换YOLOV5中的Head.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1iR4y1a7bx/).  
- **yolov5-C2F**  
    使用yolov8中的C2F模块替换yolov5中的C3模块.(这个操作比较简单，因此就不提供代码，直接看视频操作一下即可)  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1rx4y1g7xt/).  
- **yolov7-iou**  
    添加EIOU，SIOU，ALPHA-IOU, FocalEIOU, Wise-IOU到yolov7的box_iou中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1zx4y177EF/).  
    博客地址：[CSDN](https://blog.csdn.net/qq_37706472/article/details/128780275?spm=1001.2014.3001.5502).  
    #### 2023-2-11 更新: 新增[Wise-IoU](https://arxiv.org/abs/2301.10051) 视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1yv4y147kf/). reference:[github](https://github.com/Instinct323/wiou)  
- **yolov5-OTA**  
    添加Optimal Transport Assignment到yolov5的Loss中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1zx4y177EF/).  
- **yolov5-DCN**  
    添加Deformable convolution V2到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1rT411Q76q/).  
- **yolov8-DCN**  
    添加Deformable convolution V2到yolov8中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1Fo4y1i7Mm/).  
- **yolov7-DCN**  
    添加Deformable convolution V2到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV17R4y1q7vr/).  
- **yolov5-AUX**
    添加辅助训练分支到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1Fo4y1v7bi/).  
    原理参考链接：[知乎](https://zhuanlan.zhihu.com/p/588947172)
- **CAM**  
    添加context augmentation module到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV17b411d7ef/).  
    paper：[链接](https://openreview.net/pdf?id=q2ZaVU6bEsT)
- **yolov5-SAConv**  
    添加SAC到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1xD4y1u7NU/).  
    paper：[链接](https://arxiv.org/pdf/2006.02334.pdf)  
    reference: [链接](https://github.com/joe-siyuan-qiao/DetectoRS)
- **yolov7-SAConv**  
    添加SAC到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1xD4y1u7NU/).  
    paper：[链接](https://arxiv.org/pdf/2006.02334.pdf)  
    reference: [链接](https://github.com/joe-siyuan-qiao/DetectoRS)
- **yolov5-CoordConv**  
    添加CoordConv到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1ng4y1E7rS/).   
    reference: [链接](https://blog.csdn.net/qq_35608277/article/details/125257225)
- **yolov5-soft-nms**  
    添加soft-nms(IoU,GIoU,DIoU,CIoU,EIoU,SIoU)到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1cM41147Ry/).  
- **yolov7-CoordConv**  
    添加CoordConv到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1K54y1g7ye/).   
    reference: [链接](https://blog.csdn.net/qq_35608277/article/details/125257225)
- **yolov7-soft-nms**  
    添加soft-nms(IoU,GIoU,DIoU,CIoU,EIoU,SIoU)到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1ZY41167iC/). 
- **yolov5-DSConv**  
    添加DSConv到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1iT411a7Mi/).   
    paper: [链接](https://arxiv.org/abs/1901.01928)  
    reference: [链接](https://github.com/ActiveVisionLab/DSConv)
- **yolov7-DSConv**  
    添加DSConv到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1724y1b7PD/).   
    paper: [链接](https://arxiv.org/abs/1901.01928)  
    reference: [链接](https://github.com/ActiveVisionLab/DSConv)
- **yolov5-DCNV3**  
    添加DCNV3到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1LY411z7iE/).   
    补充事项-视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1Dv4y1j7ij/).   
    paper: [链接](https://arxiv.org/abs/2211.05778)  
    reference: [链接](https://github.com/OpenGVLab/InternImage)  
- **yolov5-NWD**  
    添加Normalized Gaussian Wasserstein Distance到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1zY4y197UP/).   
    paper: [链接](https://arxiv.org/abs/2110.13389)  
    reference: [链接](https://github.com/jwwangchn/NWD)  
- **yolov7-DCNV3**  
    添加DCNV3到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1LY411z7iE/).   
    paper: [链接](https://arxiv.org/abs/2211.05778)  
    reference: [链接](https://github.com/OpenGVLab/InternImage) 
- **yolov7-DCNV3**  
    添加DCNV3到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1mk4y1h7us/).   
    paper: [链接](https://arxiv.org/abs/2211.05778)  
    reference: [链接](https://github.com/OpenGVLab/InternImage) 
- **yolov5-DecoupledHead**  
    添加Efficient-DecoupledHead到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1mk4y1h7us/).   
    paper: [yolov6链接](https://arxiv.org/pdf/2301.05586.pdf)  
    reference: [链接](https://github.com/meituan/YOLOv6/blob/main/yolov6/models/effidehead.py) 
- **yolov5-FasterBlock**  
    添加FasterNet中的Faster-Block到yolov5中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1Bs4y1H7Ph/).   
    paper: [链接](https://arxiv.org/abs/2303.03667)  
    reference: [链接](https://github.com/JierunChen/FasterNet) 
- **yolov7-NWD**  
    添加Normalized Gaussian Wasserstein Distance到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1kM411H7g1/).   
    paper: [链接](https://arxiv.org/abs/2110.13389)  
    reference: [链接](https://github.com/jwwangchn/NWD)
- **yolov7-DecoupledHead**  
    添加具有隐式知识学习的Efficient-DecoupledHead到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1tg4y1x7ha/).   
    paper: [yolov6链接](https://arxiv.org/pdf/2301.05586.pdf) [yolor链接](https://arxiv.org/abs/2105.04206) [yolor参考博客](https://blog.csdn.net/AaronYKing/article/details/123804988)  
    reference: [链接](https://github.com/meituan/YOLOv6/blob/main/yolov6/models/effidehead.py) 
- **yolov5-backbone**  
    添加Timm支持的主干到yolov5中.  
    需要安装timm库. 命令: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1Mx4y1A7jy/).   
    reference: [链接](https://github.com/huggingface/pytorch-image-models#:~:text=I%20missed%20anything.-,Models,-All%20model%20architecture)
- **yolov7-PConv**  
    添加FasterNet中的PConv到yolov7中.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1Bs4y1H7Ph/).   
    paper: [链接](https://arxiv.org/abs/2303.03667)  
    reference: [链接](https://github.com/JierunChen/FasterNet) 