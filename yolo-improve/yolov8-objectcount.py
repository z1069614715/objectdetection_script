import warnings
warnings.filterwarnings('ignore')
import cv2, os, shutil
import numpy as np
from ultralytics import YOLO

def get_video_cfg(path):
    video = cv2.VideoCapture(path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter_fourcc(*'XVID'), size, fps

def plot_and_counting(result):
    image_plot = result.plot()
    box_count = result.boxes.shape[0]
    cv2.putText(image_plot, f'Object Counts:{box_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return image_plot

if __name__ == '__main__':
    output_dir = 'result'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO('yolov8n.pt') # select your model.pt path
    
    # ----------------------for images or images-folder----------------------
    for result in model.predict(source='ultralytics/assets',
                  stream=True,
                  imgsz=640,
                  save=False,
                  # conf=0.2,
                  ):
        image_plot = plot_and_counting(result)
        cv2.imwrite(f'{output_dir}/{os.path.basename(result.path)}', image_plot)
    
    # ----------------------for video-folder----------------------
    # video_base_path = 'video'
    # for video_path in os.listdir(video_base_path):
    #     fourcc, size, fps = get_video_cfg(f'{video_base_path}/{video_path}')
    #     video_output = cv2.VideoWriter(f'{output_dir}/{video_path}', fourcc, fps, size)
    #     for result in model.predict(source=f'{video_base_path}/{video_path}',
    #                   stream=True,
    #                   imgsz=640,
    #                   save=False,
    #                   # conf=0.2,
    #                   ):
    #         image_plot = plot_and_counting(result)
    #         video_output.write(image_plot)
    #     video_output.release()