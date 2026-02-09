import warnings
warnings.filterwarnings('ignore')
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets

# COCO指标如果一直生成不出来之类的问题可以看这期视频排查：https://www.bilibili.com/video/BV1SdNizEE4X/
# 出现缺失的info健的问题请装pycocotools==2.0.8

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='data.json', help='label coco json path') # 数据集coco格式的json标签文件
    parser.add_argument('--pred_json', type=str, default='', help='pred coco json path') # 数据集coco格式的json模型推理文件
    
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir='tide_result')