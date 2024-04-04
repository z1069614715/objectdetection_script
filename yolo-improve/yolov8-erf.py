import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys, glob
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from timm.utils import AverageMeter
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns

def get_activation(feat, backbone_idx=-1):
    def hook(model, inputs, outputs):
        if backbone_idx != -1:
            for _ in range(5 - len(outputs)): outputs.insert(0, None)
            feat.append(outputs[backbone_idx])
        else:
            feat.append(outputs)
    return hook

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None

def heatmap(data, camp='RdYlGn', figsize=(10, 10.75), ax=None, save_path=None):
    plt.figure(figsize=figsize, dpi=40)
    ax = sns.heatmap(data,
                xticklabels=False,
                yticklabels=False, cmap=camp,
                center=0, annot=False, ax=ax, cbar=True, annot_kws={"size": 24}, fmt='.2f')
    plt.tight_layout()
    plt.savefig(save_path)

class yolov8_erf:
    feature, hooks = [], []
    
    def __init__(self, weight, device, layer, dataset, num_images, save_path) -> None:
        device = torch.device(device)
        ckpt = torch.load(weight)
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0, weight_decay=0)
        meter = AverageMeter()
        optimizer.zero_grad()
        
        if '-' in layer:
            layer_first, layer_second = layer.split('-')
            self.hooks.append(model.model[int(layer_first)].register_forward_hook(get_activation(self.feature, backbone_idx=int(layer_second))))
        else:
            self.hooks.append(model.model[int(layer)].register_forward_hook(get_activation(self.feature)))
    
        self.__dict__.update(locals())
    
    def get_input_grad(self, samples):
        _ = self.model(samples)
        outputs = self.feature[-1]
        self.feature.clear()
        out_size = outputs.size()
        central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map
    
    def process(self):
        for image_path in os.listdir(self.dataset):
            if self.meter.count == self.num_images:
                break
            
            img = cv2.imread(f'{self.dataset}/{image_path}')
            img = letterbox(img, auto=False)[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255.0
            samples = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
            samples.requires_grad = True
            self.optimizer.zero_grad()
            contribution_scores = self.get_input_grad(samples)
            
            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                print(f'{self.meter.count}/{self.num_images} calculate....')
                self.meter.update(contribution_scores)
        
        #   Set figure parameters
        large = 24; med = 24; small = 24
        params = {'axes.titlesize': large,
                'legend.fontsize': med,
                'figure.figsize': (16, 10),
                'axes.labelsize': med,
                'xtick.labelsize': med,
                'ytick.labelsize': med,
                'figure.titlesize': large}
        plt.rcParams.update(params)
        plt.style.use('seaborn-whitegrid')
        sns.set_style("white")
        plt.rc('font', **{'family': 'Times New Roman'})
        plt.rcParams['axes.unicode_minus'] = False
        
        data = self.meter.avg
        print(f'max value:{np.max(data):.3f} min value:{np.min(data):.3f}')
        
        data = np.log10(data + 1)       #   the scores differ in magnitude. take the logarithm for better readability
        data = data / np.max(data)      #   rescale to [0,1] for the comparability among models
        print('======================= the high-contribution area ratio =====================')
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            side_length, area_ratio = get_rectangle(data, thresh)
            print('thresh, rectangle side length, area ratio: ', thresh, side_length, area_ratio)
        heatmap(data, save_path=self.save_path)


def get_params():
    params = {
        'weight': 'yolov8n.pt', # 只需要指定权重即可
        'device': 'cuda:0',
        'layer': '10', # string
        'dataset': '',
        'num_images': 50,
        'save_path': 'result.png'
    }
    return params

if __name__ == '__main__':
    cfg = get_params()
    yolov8_erf(**cfg).process()