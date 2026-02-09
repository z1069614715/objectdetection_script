import json
import os
from pathlib import Path
from PIL import Image


class YOLOtoCOCO:
    def __init__(self, yolo_dir, image_dir, class_names, output_json='coco_annotations.json'):
        """
        初始化YOLO到COCO转换器
        
        Args:
            yolo_dir: YOLO标签文件目录
            image_dir: 图片文件目录
            class_names: 类别名称列表，索引对应YOLO的类别ID
            output_json: 输出的COCO格式JSON文件路径
        """
        self.yolo_dir = Path(yolo_dir)
        self.image_dir = Path(image_dir)
        self.class_names = class_names
        self.output_json = output_json
        
        # COCO格式的基本结构
        self.coco_format = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.annotation_id = 0
    
    def create_categories(self):
        """创建类别信息"""
        for i, class_name in enumerate(self.class_names):
            category = {
                "id": i,
                "name": class_name,
                "supercategory": "object"
            }
            self.coco_format["categories"].append(category)
    
    def yolo_to_coco_bbox(self, yolo_bbox, img_width, img_height):
        """
        将YOLO格式的bbox转换为COCO格式
        
        YOLO格式: [x_center, y_center, width, height] (归一化)
        COCO格式: [x_min, y_min, width, height] (像素值)
        """
        x_center, y_center, width, height = yolo_bbox
        
        # 转换为像素值
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # 转换为COCO格式 (左上角坐标 + 宽高)
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        
        return [x_min, y_min, width, height]
    
    def bbox_to_segmentation(self, bbox):
        """
        将bbox转换为segmentation格式
        矩形四个顶点，从左上角开始顺时针
        
        Args:
            bbox: [x_min, y_min, width, height]
        
        Returns:
            segmentation: [[x1, y1, x2, y2, x3, y3, x4, y4]]
        """
        x_min, y_min, width, height = bbox
        
        # 计算四个顶点坐标（从左上角开始顺时针）
        # 左上角
        x1, y1 = x_min, y_min
        # 右上角
        x2, y2 = x_min + width, y_min
        # 右下角
        x3, y3 = x_min + width, y_min + height
        # 左下角
        x4, y4 = x_min, y_min + height
        
        # COCO segmentation格式: [[x1, y1, x2, y2, x3, y3, x4, y4]]
        segmentation = [[x1, y1, x2, y2, x3, y3, x4, y4]]
        
        return segmentation
    
    def process_image(self, image_path, label_path):
        """处理单张图片及其标签"""
        # 使用文件名(不含扩展名)作为image_id
        image_id = image_path.stem
        
        # 读取图片获取尺寸
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"无法读取图片 {image_path}: {e}")
            return
        
        # 添加图片信息
        image_info = {
            "id": image_id,
            "file_name": image_path.name,
            "width": img_width,
            "height": img_height
        }
        self.coco_format["images"].append(image_info)
        
        # 读取YOLO标签文件
        if not label_path.exists():
            print(f"标签文件不存在: {label_path}")
            return
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 处理每个标注
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            
            # 转换bbox格式
            coco_bbox = self.yolo_to_coco_bbox(bbox, img_width, img_height)
            
            # 计算面积
            area = coco_bbox[2] * coco_bbox[3]
            
            # 生成segmentation（矩形四个顶点）
            segmentation = self.bbox_to_segmentation(coco_bbox)
            
            # 创建标注信息
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation
            }
            self.coco_format["annotations"].append(annotation)
            self.annotation_id += 1
    
    def convert(self):
        """执行转换"""
        print("开始转换YOLO格式到COCO格式...")
        
        # 创建类别信息
        self.create_categories()
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.image_dir.glob(f'*{ext}'))
            image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 处理每张图片
        for image_path in image_files:
            # 对应的标签文件
            label_path = self.yolo_dir / f"{image_path.stem}.txt"
            self.process_image(image_path, label_path)
        
        # 保存为JSON文件
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(self.coco_format, f, indent=2, ensure_ascii=False)
        
        print(f"转换完成！")
        print(f"图片数量: {len(self.coco_format['images'])}")
        print(f"标注数量: {len(self.coco_format['annotations'])}")
        print(f"类别数量: {len(self.coco_format['categories'])}")
        print(f"输出文件: {self.output_json}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    yolo_label_dir = "/root/dataset/dataset_visdrone/VisDrone2019-DET-test-dev/labels"  # YOLO标签文件目录
    image_dir = "/root/dataset/dataset_visdrone/VisDrone2019-DET-test-dev/images"  # 图片目录
    
    # 类别名称列表（索引对应YOLO的类别ID）
    class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    
    output_json = "/root/dataset/dataset_visdrone/VisDrone2019-DET-test-dev/coco_annotations.json"  # 输出文件名
    
    # 创建转换器并执行转换
    converter = YOLOtoCOCO(
        yolo_dir=yolo_label_dir,
        image_dir=image_dir,
        class_names=class_names,
        output_json=output_json
    )
    
    converter.convert()