import warnings
warnings.filterwarnings('ignore')
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# BILIBILI UP 魔傀面具
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt') # 选择训练好的权重路径
    result = model.val(data='data.yaml',
                        split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
                        imgsz=640,
                        batch=16,
                        iou=0.7,
                        project='runs/val',
                        name='exp',
                        )
    
    if model.task == 'detect': # 仅目标检测任务适用
        model_names = list(result.names.values())
        time_per_image = result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']
        
        n_l, n_p, n_g, flops = model_info(model.model)
        
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "FPS"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', f'{1000 / time_per_image:.2f}'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx, cls_name in enumerate(model_names):
            model_metrice_table.add_row([
                                        cls_name, 
                                        f"{result.box.p[idx]:.4f}", 
                                        f"{result.box.r[idx]:.4f}", 
                                        f"{result.box.f1[idx]:.4f}", 
                                        f"{result.box.ap50[idx]:.4f}", 
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[idx]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all", 
                                    f"{result.results_dict['metrics/precision(B)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(B)']:.4f}", 
                                    f"{np.mean(result.box.f1):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
                                ])
        print(model_metrice_table)