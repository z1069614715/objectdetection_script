def bbox_mpdiou(box1, box2, x1y1x2y2=True, mpdiou_hw=None, grid=None, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    box1[:2] += grid
    box2[:2] += grid

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return iou - d1 / mpdiou_hw - d2 / mpdiou_hw  # MPDIoU

# ComputeLoss
iou = bbox_mpdiou(pbox.T, tbox[i], x1y1x2y2=False, mpdiou_hw=pi.size(2) ** 2 + pi.size(3) ** 2, grid=torch.stack([gj, gi]))  # iou(prediction, target)

# ComputeLossOTA
iou = bbox_mpdiou(pbox.T, selected_tbox, x1y1x2y2=False, mpdiou_hw=pi.size(2) ** 2 + pi.size(3) ** 2, grid=torch.stack([gj, gi]))  # iou(prediction, target)