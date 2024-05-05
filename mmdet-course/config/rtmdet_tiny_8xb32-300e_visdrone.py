_base_ = 'rtmdet_tiny_8xb32-300e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=10
    )
)

# 修改数据集相关配置
data_root = '/home/hjj/Desktop/dataset/dataset_visdrone/'
metainfo = {
    'classes': ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-train/annotations/train.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/')))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-val/annotations/val.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/')))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-test-dev/annotations/test.json',
        data_prefix=dict(img='VisDrone2019-DET-test-dev/images/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'VisDrone2019-DET-val/annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'VisDrone2019-DET-test-dev/annotations/test.json')

# optim_wrapper = dict(type='AmpOptimWrapper')

default_hooks = dict(logger=dict(type='LoggerHook', interval=200))
load_from='rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

# nohup python tools/train.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py > rtmdet-tiny-visdrone.log 2>&1 & tail -f rtmdet-tiny-visdrone.log
# python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py work_dirs/rtmdet_tiny_8xb32-300e_visdrone/epoch_300.pth --show --show-dir test_save
# python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py work_dirs/rtmdet_tiny_8xb32-300e_visdrone/epoch_300.pth --tta 
# python tools/analysis_tools/get_flops.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py