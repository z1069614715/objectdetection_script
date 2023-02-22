import os, cv2, tqdm, shutil
import numpy as np

def xywh2xyxy(box):
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box

def iou(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
 
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
 
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
 
    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter
 
    iou = area_inter / area_union
    return iou

def draw_box(img, box, color):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=2)
    return img

if __name__ == '__main__':
    postfix = 'jpg'
    img_path = 'image'
    label_path = 'label'
    predict_path = 'predict'
    save_path = 'vis'
    classes = ['train', 'diningtable', 'person', 'bus', 'pottedplant', 'chair', 'cat', 'tvmonitor', 'motorbike', 'sofa', 'cow', 'bottle', 'aeroplane', 'dog', 'horse', 'car', 'boat', 'sheep', 'bicycle', 'bird']
    detect_color, missing_color, error_color  = (0, 255, 0), (0, 0, 255), (255, 0, 0)
    iou_threshold = 0.45
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    for path in tqdm.tqdm(os.listdir(label_path)):
        image = cv2.imread(f'{img_path}/{path[:-4]}.{postfix}')
        if image is None:
            print(f'image:{img_path}/{path[:-4]}.{postfix} not found.')
        h, w = image.shape[:2]
        
        try:
            with open(f'{predict_path}/{path}') as f:
                pred = np.array(list(map(lambda x:np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])
                pred[:, [1, 3]] *= w
                pred[:, [2, 4]] *= h
                pred = list(pred)
        except:
            pred = []
        
        with open(f'{label_path}/{path}') as f:
            label = np.array(list(map(lambda x:np.array(x.strip().split(), dtype=np.float32), f.readlines())))
            label[:, 1:] = xywh2xyxy(label[:, 1:])
            label[:, [1, 3]] *= w
            label[:, [2, 4]] *= h
        
        label_id, pred_id = list(range(label.shape[0])), [] if len(pred) == 0 else list(range(len(pred)))
        for i in range(label.shape[0]):
            if len(pred) == 0: break
            ious = iou(label[i:i+1, 1:], np.array(pred)[:, 1:5])[0]
            ious_argsort = ious.argsort()[::-1]
            missing = True
            for j in ious_argsort:
                if ious[j] < iou_threshold: break
                if label[i, 0] == pred[j][0]:
                    image = draw_box(image, pred[j][1:5], detect_color)
                    pred.pop(j)
                    missing = False
                    break
            
            if missing: image = draw_box(image, label[i][1:5], missing_color)
        
        if len(pred):
            for j in range(len(pred)):
                image = draw_box(image, pred[j][1:5], error_color)
        
        cv2.imwrite(f'{save_path}/{path[:-4]}.{postfix}', image)
            
