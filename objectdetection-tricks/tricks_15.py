import os, glob, cv2, tqdm
from prettytable import PrettyTable

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

image_postfix = ['jpg', 'png', 'bmp', 'tif']
images_folder_path = ['/home/dataset/dataset_visdrone/VisDrone2019-DET-train/images', 
                      '/home/dataset/dataset_visdrone/VisDrone2019-DET-val/images',
                      '/home/dataset/dataset_visdrone/VisDrone2019-DET-test-dev/images']
labels_folder_path = ['/home/dataset/dataset_visdrone/VisDrone2019-DET-train/labels',
                      '/home/dataset/dataset_visdrone/VisDrone2019-DET-val/labels',
                      '/home/dataset/dataset_visdrone/VisDrone2019-DET-test-dev/labels']
classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
# classes = ['people', 'bicycle', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
object_info = [32*32, 96*96]
COLOR_LIST = [
    (255, 0, 0),         # 红色 (person)
    (0, 255, 0),         # 绿色 (car)
    (0, 0, 255),         # 蓝色 (bike)
    (255, 165, 0),       # 橙色 (motorcycle)
    (255, 255, 0),       # 黄色 (truck)
    (0, 255, 255),       # 青色 (bus)
    (255, 0, 255),       # 品红 (train)
    (255, 255, 255),     # 白色 (airplane)
    (128, 0, 0),         # 棕色 (dog)
    (0, 128, 0),         # 深绿色 (cat)
    (0, 0, 128),         # 深蓝色 (horse)
    (128, 128, 0),       # 橄榄色 (sheep)
    (0, 128, 128),       # 蓝绿色 (cow)
    (128, 0, 128),       # 紫色 (elephant)
    (192, 192, 192),     # 银色 (giraffe)
    (255, 99, 71),       # 番茄色 (zebra)
    (0, 255, 127),       # 春绿色 (monkey)
    (255, 105, 180),     # 深粉色 (bird)
    (70, 130, 180),      # 钢蓝色 (fish)
]

def get_color_by_class(class_id):
    # 根据类别的索引返回固定颜色
    return COLOR_LIST[class_id % len(COLOR_LIST)]  # 确保索引不越界

def draw_detections(box, name, color, img):
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))
    
    # 根据图像大小调整矩形框的线宽和文本的大小
    line_thickness = max(1, int(min(height, width) / 400))
    font_scale = min(height, width) / 1000
    font_thickness = max(1, int(min(height, width) / 400))
    # 根据图像大小调整文本的纵向位置
    text_offset_y = int(min(height, width) / 100)
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, line_thickness)
    cv2.putText(img, str(name), (xmin, ymin - text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
    return img

def get_images_and_labels_path(images_folder_path, labels_folder_path):
    labels_path_list, labels_filename = [], {}
    for folder_path in labels_folder_path:
        glob_list = glob.glob(os.path.join(folder_path, '*.txt'))
        filename = {os.path.splitext(os.path.basename(i))[0]:i for i in glob_list}
        labels_path_list.extend(glob_list)
        labels_filename.update(filename)
    
    images_path_list, images_filename = [], {}
    for folder_path in images_folder_path:
        for p in image_postfix:
            glob_list = glob.glob(os.path.join(folder_path, f'*.{p}'))
            filename = {os.path.splitext(os.path.basename(i))[0]:i for i in glob_list}
            images_path_list.extend(glob_list)
            images_filename.update(filename)
    
    print(ORANGE + f'image_path_length:{len(images_filename)} label_path_length:{len(labels_filename)}')

    image_label_dict = {}
    for i in labels_filename:
        if i in images_filename:
            image_label_dict[labels_filename[i]] = images_filename[i]
    
    print(f'After matching. data_length:{len(image_label_dict)}' + RESET)

    return image_label_dict, labels_path_list

def show_dataset_info(image_label_dict, visual_box=False, save_path='visual_box'):
    if visual_box and not os.path.exists(save_path):
        os.makedirs(save_path)

    classes_dict = {cls:{'s':0, 'm':0, 'l':0, 'num':0} for cls in classes}
    for label_path in tqdm.tqdm(image_label_dict):
        image_path = image_label_dict[label_path]

        image = cv2.imread(image_path)
        try:
            h, w = image.shape[:2]
        except:
            print(RED + f'{image_path} read failure. skip.' + RESET)
        
        with open(label_path) as f:
            label = list(map(lambda x:x.strip().split(), f.readlines()))
        
        for cls_id,x_c,y_c,width,height in label:
            classes_dict[classes[int(float(cls_id))]]['num'] += 1
            width = float(width) * w
            height = float(height) * h
            obj_area = width * height

            if obj_area < object_info[0]:
                classes_dict[classes[int(float(cls_id))]]['s'] += 1
            elif obj_area > object_info[1]:
                classes_dict[classes[int(float(cls_id))]]['l'] += 1
            else:
                classes_dict[classes[int(float(cls_id))]]['m'] += 1
            
            if visual_box:
                x_c, y_c = float(x_c) * w, float(y_c) * h
                x_min, y_min, x_max, y_max = x_c - width / 2, y_c - height / 2, x_c + width / 2, y_c + height / 2
                image = draw_detections([x_min, y_min, x_max, y_max], classes[int(float(cls_id))], get_color_by_class(int(float(cls_id))), image)
                cv2.imwrite(os.path.join(save_path, os.path.basename(image_path)), image)
    
    # 统计总和
    total_s = sum(v['s'] for v in classes_dict.values())
    total_m = sum(v['m'] for v in classes_dict.values())
    total_l = sum(v['l'] for v in classes_dict.values())
    total_num = sum(v['num'] for v in classes_dict.values())

    # 创建表格
    table = PrettyTable()
    table.field_names = ["Category", "Small (s)", "Medium (m)", "Large (l)", "Total (num)"]

    # 添加每一行
    for category, values in classes_dict.items():
        s, m, l, num = values['s'], values['m'], values['l'], values['num']
        row = [
            category,
            f"{s} ({s/num:.1%})",
            f"{m} ({m/num:.1%})",
            f"{l} ({l/num:.1%})",
            num
        ]
        table.add_row(row)

    # 添加总计行
    row_total = [
        "All",
        f"{total_s} ({total_s/total_num:.1%})",
        f"{total_m} ({total_m/total_num:.1%})",
        f"{total_l} ({total_l/total_num:.1%})",
        total_num
    ]
    table.add_row(row_total)

    # 可选：左对齐类别列
    table.align["Category"] = "l"

    # 打印表格
    print(table)

def remap_yolo_dataset_class(labels_path_list, delete_label=[0, 1, 3, 5]):
    classes = []
    for label_path in tqdm.tqdm(labels_path_list, desc='scan dataset class'):
        with open(label_path) as f:
            label = list(map(lambda x:x.strip().split(), f.readlines()))
            
        for cls_id,x_c,y_c,width,height in label:
            classes.append(int(float(cls_id)))
    classes = sorted(list(set(classes)))
    filter_classes = list(sorted(set(classes) - set(delete_label)))
    print(ORANGE + f'now classes:{classes} delete classes:{delete_label} filter_classes:{filter_classes}' + RESET)

    for label_path in tqdm.tqdm(labels_path_list, desc='process dataset class'):
        with open(label_path) as f:
            label = list(map(lambda x:x.strip().split(), f.readlines()))
        
        new_label = []
        for cls_id,x_c,y_c,width,height in label:
            if int(float(cls_id)) in delete_label:
                continue

            new_label.append(' '.join([str(filter_classes.index(int(float(cls_id)))),x_c,y_c,width,height]))
        
        with open(label_path, 'w+') as f:
            f.write('\n'.join(new_label))

if __name__ == '__main__':
    image_label_dict, labels_path_list = get_images_and_labels_path(images_folder_path, labels_folder_path)
    
    show_dataset_info(image_label_dict, visual_box=True)
    # remap_yolo_dataset_class(labels_path_list, delete_label=[0, 3])