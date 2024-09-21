import os, torch, cv2, math, tqdm, time, shutil, argparse, json, pickle
import numpy as np
from prettytable import PrettyTable

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_coco', type=str, default='/home/hjj/Desktop/dataset/dataset_visdrone/test_coco.json', help='label coco path')
    parser.add_argument('--pred_coco', type=str, default='runs/val/exp/predictions.json', help='pred coco path')
    # parser.add_argument('--pred_coco', type=str, default='/home/hjj/Desktop/github_code/mmdetection-visdrone/work_dirs/dino-4scale_r50_8xb2-12e_visdrone/test/prediction.pickle', help='pred coco path')
    parser.add_argument('--iou', type=float, default=0.7, help='iou threshold')
    parser.add_argument('--conf', type=float, default=0.001, help='conf threshold')
    opt = parser.parse_known_args()[0]
    return opt
    
if __name__ == '__main__':
    opt = parse_opt()
    
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    stats = []
    
    label_coco_json_path, pred_coco_json_path = opt.label_coco, opt.pred_coco
    with open(label_coco_json_path) as f:
        label = json.load(f)
    
    classes = []
    for data in label['categories']:
        classes.append(data['name'])
    
    image_id_hw_dict = {}
    for data in label['images']:
        image_id_hw_dict[data['id']] = [data['height'], data['width']]
    
    label_id_dict = {}
    for data in tqdm.tqdm(label['annotations'], desc='Process label...'):
        if data['image_id'] not in label_id_dict:
            label_id_dict[data['image_id']] = []
        
        category_id = data['category_id']
        x_min, y_min, w, h = data['bbox'][0], data['bbox'][1], data['bbox'][2], data['bbox'][3]
        x_max, y_max = x_min + w, y_min + h
        label_id_dict[data['image_id']].append(np.array([int(category_id), x_min, y_min, x_max, y_max]))
    
    if pred_coco_json_path.endswith('json'):
        with open(pred_coco_json_path) as f:
            pred = json.load(f)
        pred_id_dict = {}
        for data in tqdm.tqdm(pred, desc='Process pred...'):
            if data['image_id'] not in pred_id_dict:
                pred_id_dict[data['image_id']] = []
            
            score = data['score']
            category_id = data['category_id']
            x_min, y_min, w, h = data['bbox'][0], data['bbox'][1], data['bbox'][2], data['bbox'][3]
            x_max, y_max = x_min + w, y_min + h
            
            pred_id_dict[data['image_id']].append(np.array([x_min, y_min, x_max, y_max, float(score), int(category_id)]))
    else:
        with open(pred_coco_json_path, 'rb') as f:
            pred = pickle.load(f)
        pred_id_dict = {}
        for data in tqdm.tqdm(pred, desc='Process pred...'):
            image_id = os.path.splitext(os.path.basename(data['img_path']))[0]
            if image_id not in pred_id_dict:
                pred_id_dict[image_id] = []
            
            for i in range(data['pred_instances']['labels'].size(0)):
                score = data['pred_instances']['scores'][i]
                category_id = data['pred_instances']['labels'][i]
                bboxes = data['pred_instances']['bboxes'][i]
                
                x_min, y_min, x_max, y_max = bboxes.cpu().detach().numpy()
                # x_min, x_max = x_min / data['scale_factor'][0], x_max / data['scale_factor'][0]
                # y_min, y_max = y_min / data['scale_factor'][1], y_max / data['scale_factor'][1]
                
                pred_id_dict[image_id].append(np.array([x_min, y_min, x_max, y_max, float(score), int(category_id)]))
    
    for idx, image_id in enumerate(tqdm.tqdm(list(image_id_hw_dict.keys()), desc="Cal mAP...")):
        label = np.array(label_id_dict[image_id])
        
        if image_id not in pred_id_dict:
            pred = np.empty((0, 6))
        else:
            pred = torch.from_numpy(np.array(pred_id_dict[image_id]))
        
        nl, npr = label.shape[0], pred.shape[0]
        correct = torch.zeros(npr, niou, dtype=torch.bool)
        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0)), torch.from_numpy(label[:, 0])))
            continue
        
        if nl:
            correct = process_batch(pred, torch.from_numpy(label), iouv)
        stats.append((correct, pred[:, 4], pred[:, 5], torch.from_numpy(label[:, 0])))
    
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
    print(f'precision:{p}')
    print(f'recall:{r}')
    print(f'mAP@0.5:{ap[:, 0]}')
    
    table = PrettyTable()
    table.title = f"Metrice"
    table.field_names = ["Classes", 'Precision', 'Recall', 'mAP50', 'mAP50-95']
    table.add_row(['all', f'{np.mean(p):.3f}', f'{np.mean(r):.3f}', f'{np.mean(ap[:, 0]):.3f}', f'{np.mean(ap):.3f}'])
    for cls_idx, classes in enumerate(classes):
        table.add_row([classes, f'{p[cls_idx]:.3f}', f'{r[cls_idx]:.3f}', f'{ap[cls_idx, 0]:.3f}', f'{ap[cls_idx, :].mean():.3f}'])
    print(table)