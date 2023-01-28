# YOLO-Improve
这个项目主要是提供一些关于yolo系列模型的改进思路，效果因数据集和参数而异，仅作参考。

# Explanation
- **iou**  
    添加EIOU，SIOU，ALPHA-IOU, FocalEIOU到yolov5,yolov8的box_iou中.  
    1. yolov5
        视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1KM411b7Sz/).  
        博客地址：[CSDN](https://blog.csdn.net/qq_37706472/article/details/128737484?spm=1001.2014.3001.5501).
    2. yolov8
        视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1PY4y1o7Hm/).  
        博客地址：[CSDN](https://blog.csdn.net/qq_37706472/article/details/128743012?spm=1001.2014.3001.5502).
- **yolov5-GFPN**  
    使用DAMO-YOLO中的GFPN替换YOLOV5中的Head.  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1iR4y1a7bx/).  
- **yolov5-C2F**  
    使用yolov8中的C2F模块替换yolov5中的C3模块.(这个操作比较简单，因此就不提供代码，直接看视频操作一下即可)  
    视频教学地址：[哔哩哔哩](https://www.bilibili.com/video/BV1rx4y1g7xt/).  
- **yolov7-iou**  
    添加EIOU，SIOU，ALPHA-IOU, FocalEIOU到yolov7的box_iou中.  