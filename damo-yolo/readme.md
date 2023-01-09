# DAMO-YOLO的数据集处理文件
本目录下的脚本是针对与DAMO-YOLO的数据集处理脚本，支持如下：
1. VOC标注格式转换为COCO标注格式，并生成train.json,val.json,test.json.

# 使用方法
1. 把图片存放在JPEGImages中，图片后缀需要一致，比如都是jpg或者png等等，不支持混合的图片后缀格式，比如一些是jpg，一些是png。
2. 把VOC标注格式的XML文件存放在Annotations中。
3. 运行voc2coco.py,其中postfix参数是JPEGImages的图片后缀，train_ratio是训练集的比例，val_ratio是验证集的比例，剩下的就是测试集的比例。