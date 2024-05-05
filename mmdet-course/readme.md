# mmdet使用教程

### mmdet教程命令

1. conda create -n mmdet_py39 python=3.9 anaconda
2. https://mmdetection.readthedocs.io/en/latest/get_started.html
3. https://pytorch.org/get-started/previous-versions/  
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
4. https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/train.html#id7

### mmdet运行命令

1. 训练

        python tools/train.py <your-config-file>
2. 测试  

        python tools/test.py <your-config-file> <your-model-weights-file> --out <save-pickle-path>
3. 计算量、参数量计算脚本  

        python tools/analysis_tools/get_flops.py <your-config-file>
4. 推理时间、fps、gpu memory计算脚本  

        python tools/analysis_tools/benchmark.py <your-config-file> --checkpoint <your-model-weights-file> --task inference --fuse-conv-bn
5. 绘制曲线图脚本  

        python tools/analysis_tools/analyze_logs.py plot_curve <train-json-file> --keys <keys> --legend <legend> --out <save-path>
6. 结果分析脚本  

        python tools/analysis_tools/analyze_results.py <your-config-file> <test-pickle-path> <save-path>

### mmdet视频教程链接(可按顺序观看)

1. [一库打尽目标检测对比实验！mmdetection环境、训练、测试手把手教程！](https://www.bilibili.com/video/BV1xA4m1c7H8/)
2. [一库打尽目标检测对比实验！mmdetection参数量、计算量、FPS、绘制logs手把手教程](https://www.bilibili.com/video/BV17C41137dW/)

### mmdet实验数据(指标均为COCO指标)

以下实验数据环境:  
python:3.9.19  
torch:2.1.0+cu121  
torchvision:0.16.0  
mmdet:3.3.0  
mmcv:2.1.0  
mmengine:0.10.3  
硬件环境:  
Platform:Ubuntu  
CPU:i7-12700K  
RAM:32G  
GPU:RTX3090  

#### VisDrone2019-testset

| model | Input Shape | GFlops | Params | coco/bbox_mAP | coco/bbox_mAP_50 | coco/bbox_mAP_s | coco/bbox_mAP_m | coco/bbox_mAP_l |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Faster-RCNN-R50-FPN-CIOU | (768, 1344) | 208G | 41.39M | 0.194 | 0.329 | 0.095 | 0.309 | 0.429 |
| Cascade-RCNN-R50-FPN | (768, 1344) | 236G | 69.29M | 0.197 | 0.326 | 0.099 | 0.309 | 0.406 |
| ATSS-R50-FPN-DyHead | (768, 1344) | 110G | 38.91M | 0.204 | 0.338 | 0.100 | 0.317 | 0.485 |
| TOOD-R50 | (768, 1344) | 199G | 32.04M | 0.204 | 0.339 | 0.102 | 0.317 | 0.403 |
| DINO | (750, 1333) | 274G | 47.56M | 0.253 | 0.445 | 0.150 | 0.371 | 0.503 |
| DDQ | (768, 1333) | - | - | 0.268 | 0.463 | 0.159 | 0.390 | 0.526 |
| YOLOX-Tiny | (640, 640) | 7.578G | 5.035M | 0.148 | 0.278 | 0.076 | 0.221 | 0.278 |
| GFL | (768, 1344) | 206G | 32.279M | 0.193 | 0.321 | 0.094 | 0.300 | 0.409 |
| RTMDet-Tiny | (640, 640) | 8.033G | 4.876M | 0.184 | 0.312 | 0.077 | 0.288 | 0.445 |
| RetinaNet-R50-FPN | (768, 1344) | 210G | 36.517M | 0.164 | 0.276 | 0.060 | 0.274 | 0.427 |