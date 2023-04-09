import cv2
import numpy as np
import matplotlib.pylab as plt
from segment_anything import SamPredictor, sam_model_registry

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

class Select_RoI:
    def __init__(self, img) -> None:
        self.mouseWindowName = 'Select_RoI'
        self.last_img, self.cur_img = img.copy(), img.copy()
        
        self.point_lefttop, self.point_rightbottom, self.center_point, self.count = [], [], [], 0
        
        cv2.namedWindow(self.mouseWindowName, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.mouseWindowName, self.on_mouse)
        while True:
            cv2.imshow(self.mouseWindowName, self.cur_img)
            key = cv2.waitKey(5)
            if key == 13:  # 按回车键13表示完成绘制
                break
            elif key == 99:  # 按键盘c退回上一次的状态
                self.clear()
            elif key == 32:
                self.confirm()
        
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.point_lefttop) == len(self.point_rightbottom):
                self.point_lefttop.append([x, y])
                cv2.circle(self.cur_img, (x, y), 5, (0, 255, 0), -1)
            else:
                self.point_rightbottom.append([x, y])
                cv2.circle(self.cur_img, (x, y), 5, (0, 255, 0), -1)
                cv2.rectangle(self.cur_img, (tuple(self.point_lefttop[-1])), (tuple(self.point_rightbottom[-1])), (0, 0, 255), 3)
            cv2.imshow(self.mouseWindowName, self.cur_img)
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(self.cur_img, (x, y), 5, (255, 0, 0), -1)
            self.center_point.append([x, y])
    
    def clear(self):
        if len(self.center_point) == len(self.point_lefttop) == len(self.point_rightbottom):
            min_len = len(self.center_point) - 1
        else:
            min_len = np.min([len(self.center_point), len(self.point_lefttop), len(self.point_rightbottom)])
        
        if len(self.center_point) > min_len:
            self.center_point.pop(-1)
        if len(self.point_lefttop) > min_len:
            self.point_lefttop.pop(-1)
        if len(self.point_rightbottom) > min_len:
            self.point_rightbottom.pop(-1)
        
        if len(self.center_point) == len(self.point_lefttop) == len(self.point_rightbottom):
            self.count = min_len
            self.cur_img = self.last_img.copy()
        else:
            raise "center_point point_lefttop point_rightbottom not equal."
        print(f'point_lefttop:{self.point_lefttop}\npoint_rightbottom:{self.point_rightbottom}\ncenter_point:{self.center_point}\ncount:{self.count}')
    
    def confirm(self):
        self.last_img = self.cur_img.copy()
        if len(self.center_point) == len(self.point_lefttop) == len(self.point_rightbottom):
                self.count = len(self.center_point)
        else:
            raise "center_point point_lefttop point_rightbottom not equal."
        print(f'point_lefttop:{self.point_lefttop}\npoint_rightbottom:{self.point_rightbottom}\ncenter_point:{self.center_point}\ncount:{self.count}')
        
    def get_result(self):
        return np.array([np.array([*i, *j]) for i, j in zip(self.point_lefttop, self.point_rightbottom)]), np.array([np.array(i) for i in self.center_point])

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

path = '1.jpg'
image = cv2.imread(path)
roi = Select_RoI(image.copy())
box, point = roi.get_result()
label = np.array([0 for i in point])
predictor.set_image(image)
if point.shape[0] != 0:
    masks, scores, logits = predictor.predict(box=box, point_coords=point, point_labels=label)
else:
    masks, scores, logits = predictor.predict(box=box)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    if point.shape[0] != 0:
        show_points(point, label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()