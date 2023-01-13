# YOLOV5,YOLOV7,YOLOV8的数据集处理文件
本目录下的脚本是针对与yolov5,v7,v8的数据集处理脚本，支持如下：
1. VOC标注格式转换为YOLO标注格式。
2. 对数据集进行划分训练集，验证集，测试集。

# VOC标注格式数据集使用示例
1. 把图片存放在dataset\VOCdevkit\JPEGImages中，图片后缀需要一致，比如都是jpg或者png等等，不支持混合的图片后缀格式，比如一些是jpg，一些是png。
2. 把VOC标注格式的XML文件存放在dataset\VOCdevkit\Annotations中。
3. 运行xml2txt.py,在这个文件中其会把Annotations中的XML格式标注文件转换到txt中的yolo格式标注文件。其中xml2txt.py中的postfix参数是JPEGImages的图片后缀,修改成图片的后缀即可，默认为jpg。比如我的图片都是png后缀的，需要把postfix修改为png即可。其中运行这个文件的时候，输出信息会输出你的数据集的类别，你需要把类别列表复制到data.yaml中的names中，并且修改nc为你的类别数，也就是names中类别个数。
4. 运行split_data.py,这个文件是划分训练、验证、测试集。其中支持修改val_size**验证集比例**和test_size**测试集比例**，可以在split_data.py中找到对应的参数进行修改，然后postfix参数也是你的图片数据集后缀格式，默认为jpg，如果你的图片后缀不是jpg结尾的话，需要修改一下这个参数。

# YOLO标注格式数据集使用示例
1. 把图片存放在dataset\VOCdevkit\JPEGImages中，图片后缀需要一致，比如都是jpg或者png等等，不支持混合的图片后缀格式，比如一些是jpg，一些是png。
2. 把YOLO标注格式的TXT文件存放在dataset\VOCdevkit\txt中。
3. 运行split_data.py,这个文件是划分训练、验证、测试集。其中支持修改val_size**验证集比例**和test_size**测试集比例**，可以在split_data.py中找到对应的参数进行修改，然后postfix参数也是你的图片数据集后缀格式，默认为jpg，如果你的图片后缀不是jpg结尾的话，需要修改一下这个参数。
4. 在data.yaml中的names设置你的类别，其为一个list，比如我的YOLO标注格式数据集中，0代表face，1代表body，那在data.yaml中就是names:['face', 'body']，然后nc:2，nc就是类别个数。
