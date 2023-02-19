# YOLO-Improve
这个项目主要是提供一些关于yolo系列模型的改进思路，效果因数据集和参数而异，仅作参考。

# Explanation
- **iou**  
    添加EIOU，SIOU，ALPHA-IOU, FocalEIOU, Wise-IOU到yolov5,yolov8的box_iou中.  
    1. yolov5
        视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1KM411b7Sz/).  
        博客地址：[CSDN](https://blog.csdn.net/qq_37706472/article/details/128737484?spm=1001.2014.3001.5501).

        #### 2023-2-8 更新: 新增[Wise-IoU](https://arxiv.org/abs/2301.10051) 视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1tG4y1N7Gk/).  
    2. yolov8
        视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1PY4y1o7Hm/).  
        博客地址：[CSDN](https://blog.csdn.net/qq_37706472/article/details/128743012?spm=1001.2014.3001.5502).

        #### 2023-2-7 更新: 新增[Wise-IoU](https://arxiv.org/abs/2301.10051) 视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1De4y1N7Mb/).  
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
    #### 2023-2-11 更新: 新增[Wise-IoU](https://arxiv.org/abs/2301.10051) 视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1yv4y147kf/).     
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
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV17R4y1q7vr/).  
    原理参考链接：[知乎](https://zhuanlan.zhihu.com/p/588947172)