_base_ = 'atss_r50_fpn_dyhead_1x_coco.py'

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
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-train/annotations/train.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/')))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-val/annotations/val.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/')))
test_dataloader = dict(
    batch_size=8,
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

load_from='atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth'

# nohup python tools/train.py configs/dyhead/atss_r50_fpn_dyhead_1x_visdrone.py > atss-dyhead-visdrone.log 2>&1 & tail -f atss-dyhead-visdrone.log
# python tools/test.py configs/dyhead/atss_r50_fpn_dyhead_1x_visdrone.py work_dirs/tood_r50_fpn_1x_visdrone/epoch_12.pth --show --show-dir test_save
# python tools/test.py configs/dyhead/atss_r50_fpn_dyhead_1x_visdrone.py work_dirs/tood_r50_fpn_1x_visdrone/epoch_12.pth --tta 