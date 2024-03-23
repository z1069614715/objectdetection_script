import warnings
warnings.filterwarnings('ignore')
import cv2, os, shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import DeepOCSORT, BYTETracker, BoTSORT, StrongSORT, OCSORT, HybridSORT

def get_video_cfg(path):
    video = cv2.VideoCapture(path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter_fourcc(*'XVID'), size, fps

def counting(image_plot, result):
    box_count = result.boxes.shape[0]
    cv2.putText(image_plot, f'Object Counts:{box_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return image_plot

def transform_mot(result):
    mot_result = []
    for i in range(result.boxes.shape[0]):
        mot_result.append(result.boxes.xyxy[i].cpu().detach().cpu().numpy().tolist() + [float(result.boxes.conf[i]), float(result.boxes.cls[i])])
    return np.array(mot_result)

# boxmot                        10.0.57
if __name__ == '__main__':
    output_dir = 'result'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO('runs/train/yolov8m-crowdhuman/weights/best.pt') # select your model.pt path
    
    video_base_path = 'video'
    for video_path in os.listdir(video_base_path):
        
        tracker = DeepOCSORT(
        model_weights=Path('osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pt'), # which ReID model to use
        device='cuda:0',
        fp16=False,
    )
        # tracker = BoTSORT(
        #     model_weights=Path('osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pt'), # which ReID model to use
        #     device='cuda:0',
        #     fp16=False,
        # )
        # tracker = StrongSORT(
        #     model_weights=Path('osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pt'), # which ReID model to use
        #     device='cuda:0',
        #     fp16=False,
        # )
        # tracker = HybridSORT(
        #     reid_weights=Path('osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pt'), # which ReID model to use
        #     device='cuda:0',
        #     half=False,
        #     det_thresh=0.3,
        # )
        # tracker = BYTETracker()
        # tracker = OCSORT()
        
        fourcc, size, fps = get_video_cfg(f'{video_base_path}/{video_path}')
        video_output = cv2.VideoWriter(f'{output_dir}/{video_path}', fourcc, fps, size)
        for result in model.predict(source=f'{video_base_path}/{video_path}',
                      stream=True,
                      imgsz=640,
                      save=False,
                      # conf=0.2,
                      classes=1
                      ):
            image_plot = result.orig_img
            mot_input = transform_mot(result)
            try:
                tracker.update(mot_input, image_plot)
                tracker.plot_results(image_plot, show_trajectories=True)
            except:
                continue
            counting(image_plot, result)
            video_output.write(image_plot)
        video_output.release()