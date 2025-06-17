import json, tqdm, cv2, shutil, os
import numpy as np
import matplotlib.pyplot as plt

# 1. 标签文件类别有问题，例如类别从1开始，不是从0开始。
# 2. image_id不匹配。
# 3. 标签的box异常。

SAVE_PATH = 'coco_visual'
LABEL_COCO_PATH = '/Users/moguimianju/Downloads/data.json'
PRED_COCO_PATH = '/Users/moguimianju/Downloads/predictions.json'
SCORE_THR = 0.2
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

if __name__ == '__main__':
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    os.makedirs(SAVE_PATH)

    with open(LABEL_COCO_PATH) as f:
        label = json.load(f)

    with open(PRED_COCO_PATH) as f:
        predictions = json.load(f)

    print(f'label json classes info:{label["categories"]}')

    label_dict = {}
    for data in label['images']:
        image_id = data['id']
        label_dict[image_id] = {'file_name':data['file_name'], 'width':data['width'], 'height':data['height'], 'bbox_info':[]}
    
    for data in tqdm.tqdm(label['annotations'], desc='process annotations'):
        image_id = data['image_id']
        label_dict[image_id]['bbox_info'].append({'class_id':data['category_id'], 'bbox':data['bbox']})
    
    pred_classes_set = []
    pred_dict = {}
    for data in tqdm.tqdm(predictions, desc='process predictions'):
        image_id = data['image_id']
        if image_id not in pred_dict:
            pred_dict[image_id] = []
        if data['category_id'] not in pred_classes_set:
            pred_classes_set.append(data['category_id'])
        if data['score'] < SCORE_THR:
            continue
        pred_dict[image_id].append({'class_id':data['category_id'], 'bbox':data['bbox'], 'score':data['score']})

    print(f'predictions json classes set:{sorted(pred_classes_set)}')

    # print('-'*40 + 'label image_id' + '-'*40)
    # print(label_dict.keys())
    # print('-'*40 + 'pred image_id' + '-'*40)
    # print(pred_dict.keys())

    for image_id in tqdm.tqdm(label_dict, desc='process draw func'):
        if image_id not in pred_dict:
            print(f'image id:{image_id} not in predictions.json')
            continue

        label_img = np.ones((label_dict[image_id]['height'], label_dict[image_id]['width'], 3), dtype=np.uint8) * 255
        pred_img = np.ones((label_dict[image_id]['height'], label_dict[image_id]['width'], 3), dtype=np.uint8) * 255

        for bbox_info in label_dict[image_id]['bbox_info']:
            class_id = bbox_info['class_id']
            x, y, w, h = bbox_info['bbox']
            x_min, y_min, x_max, y_max = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            draw_detections([x_min, y_min, x_max, y_max], f'{class_id}', get_color_by_class(class_id), label_img)
        
        for bbox_info in pred_dict[image_id]:
            class_id = bbox_info['class_id']
            score = bbox_info['score']
            x, y, w, h = bbox_info['bbox']
            x_min, y_min, x_max, y_max = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            draw_detections([x_min, y_min, x_max, y_max], f'{class_id} {score:.2f}', get_color_by_class(class_id), pred_img)
        
        plt.figure(figsize=(12, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('label')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('predictions')

        plt.tight_layout()
        plt.savefig(f'{SAVE_PATH}/{image_id}.png')
        plt.close()