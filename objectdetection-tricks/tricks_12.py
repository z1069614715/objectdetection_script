import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def deal_yolov7_result(data_path):
    with open(data_path) as f:
        data = np.array(list(map(lambda x:np.array(x.strip().split()), f.readlines())))
    return data

if __name__ == '__main__':
    epoch = 50
    yolov5_result_csv = '/home/hjj/Desktop/github_code/yolov5/runs/train/yolov5n-crowdhuman/results.csv'
    yolov7_result_csv = '/home/hjj/Desktop/github_code/yolov7/runs/train/yolov7-tiny-crowdhuman/results.txt'
    yolov8_result_csv = '/home/hjj/Desktop/github_code/ultralytics/runs/train/yolov8n-crowdhuman/results.csv'
    yolov9_result_csv = '/home/hjj/Desktop/github_code/yolov9/runs/train/yolov9s-corwdhuman/results.csv'
    yolov10_result_csv = '/home/hjj/Desktop/github_code/yolov10/runs/train/yolov10n-crowdhuman/results.csv'
    
    yolov5_result_data = pd.read_csv(yolov5_result_csv)
    yolov7_result_data = deal_yolov7_result(yolov7_result_csv)
    yolov8_result_data = pd.read_csv(yolov8_result_csv)
    yolov9_result_data = pd.read_csv(yolov9_result_csv)
    yolov10_result_data = pd.read_csv(yolov10_result_csv)
    
    plt.figure(figsize=(10, 8))  # 调整图形大小
    plt.plot(np.arange(epoch), yolov5_result_data['     metrics/mAP_0.5'], label='yolov5n', linewidth=2)
    plt.plot(np.arange(epoch), np.array(yolov7_result_data[:, 11], dtype=float), label='yolov7-tiny', linewidth=2)
    plt.plot(np.arange(epoch), yolov8_result_data['       metrics/mAP50(B)'], label='yolov8n', linewidth=2)
    plt.plot(np.arange(epoch), yolov9_result_data['     metrics/mAP_0.5'], label='yolov9s', linewidth=2)
    plt.plot(np.arange(epoch), yolov10_result_data['       metrics/mAP50(B)'], label='yolov10n', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=14)  # 调整x轴标签字体大小
    plt.ylabel('mAP@0.5', fontsize=14)  # 调整y轴标签字体大小
    plt.legend(fontsize=20)  # 调整图例字体大小
    plt.xticks(fontsize=12)  # 调整x轴刻度字体大小
    plt.yticks(fontsize=12)  # 调整y轴刻度字体大小
    plt.title('YOLO CrowdHuman mAP50 Curve', fontsize=20)
    plt.tight_layout()
    plt.savefig('mAP50-curve.png')
    
    data_dict = {
        'yolov5n':[0.672, 0.1+3.2+0.7, '+'], 
        'yolov7-tiny':[0.74, 4.0, '*'],
        'yolov8n':[0.711, 4.5, 'x'],
        'yolov9s':[0.772, 9.9, 'D'],
        'yolov10n':[0.727, 5.3, '_']
    }
    
    plt.figure(figsize=(10, 8))  # 调整图形大小
    for model_name in data_dict:
        print(data_dict[model_name][1], data_dict[model_name][0])
        plt.scatter(data_dict[model_name][1], data_dict[model_name][0], label=model_name, marker=data_dict[model_name][2], s=500)
    plt.xlabel('Inference Time(ms/img)', fontsize=14)  # 调整x轴标签字体大小
    plt.ylabel('mAP@0.5', fontsize=14)  # 调整y轴标签字体大小
    plt.legend(fontsize=20, loc=4)  # 调整图例字体大小
    plt.xticks(fontsize=12)  # 调整x轴刻度字体大小
    plt.yticks(fontsize=12)  # 调整y轴刻度字体大小
    plt.title('inferencetimevsmAP50', fontsize=20)
    plt.tight_layout()
    plt.savefig('inferencetimevsmAP50.png')
