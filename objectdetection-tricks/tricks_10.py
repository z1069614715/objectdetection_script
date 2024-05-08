import torch, thop
from thop import profile
from ultralytics import YOLO, RTDETR
from prettytable import PrettyTable

if __name__ == '__main__':
    batch_size, height, width = 1, 640, 640

    model = YOLO(r'ultralytics/cfg/models/yolov8/yolov8n.yaml').model # select your model.pt path
    # model = RTDETR(r'ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml').model
    model.fuse()
    input = torch.randn(batch_size, 3, height, width)
    total_flops, total_params, layers = profile(model, [input], verbose=True, ret_layer_info=True)
    FLOPs, Params = thop.clever_format([total_flops * 2 / batch_size, total_params], "%.3f")
    table = PrettyTable()
    table.title = f'Model Flops:{FLOPs} Params:{Params}'
    table.field_names = ['Layer ID', "FLOPs", "Params"]
    for layer_id in layers['model'][2]:
        data = layers['model'][2][layer_id]
        FLOPs, Params = thop.clever_format([data[0] * 2 / batch_size, data[1]], "%.3f")
        table.add_row([layer_id, FLOPs, Params])
    print(table)