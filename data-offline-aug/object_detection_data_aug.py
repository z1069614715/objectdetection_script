import warnings
warnings.filterwarnings('ignore')
import os, shutil, cv2, tqdm
import numpy as np
import albumentations as A
from PIL import Image
from multiprocessing import Pool
from typing import Callable, Dict, List, Union

# https://github.com/albumentations-team/albumentations
# https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#geometric-transforms-augmentationsgeometrictransforms:~:text=Contributing%20to%20Albumentations-,Geometric%20transforms%20(augmentations.geometric.transforms),-%C2%B6

IMAGE_PATH = 'dataset/object_detection/images'
LABEL_PATH = 'dataset/object_detection/labels'
AUG_IMAGE_PATH = 'dataset/object_detection/images_aug'
AUG_LABEL_PATH = 'dataset/object_detection/labels_aug'
SHOW_SAVE_PATH = 'results'
CLASSES = ['head', 'person']

ENHANCEMENT_LOOP = 1
ENHANCEMENT_STRATEGY = A.Compose([
    A.Compose([
        A.Affine(scale=[0.5, 1.5], translate_percent=[0.0, 0.3], rotate=[-360, 360], shear=[-45, 45], keep_ratio=True, p=0.5), # Augmentation to apply affine transformations to images.
        A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.1), # Crop a random part of the input without loss of bboxes.
        A.D4(p=0.1), # Applies one of the eight possible D4 dihedral group transformations to a square-shaped input, maintaining the square shape. These transformations correspond to the symmetries of a square, including rotations and reflections.
        A.ElasticTransform(p=0.1), # Elastic deformation of images as described in [Simard2003]_ (with modifications).
        A.Flip(p=0.1), # Flip the input either horizontally, vertically or both horizontally and vertically.
        A.GridDistortion(p=0.1), # Applies grid distortion augmentation to images, masks, and bounding boxes. This technique involves dividing the image into a grid of cells and randomly displacing the intersection points of the grid, resulting in localized distortions.
        A.Perspective(p=0.1), # Perform a random four point perspective transform of the input.
    ], p=1.0),
    
    A.Compose([
        A.GaussNoise(p=0.1), # Apply Gaussian noise to the input image.
        A.ISONoise(p=0.1), # Apply camera sensor noise.
        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.1), # Decreases image quality by Jpeg, WebP compression of an image.
        A.RandomBrightnessContrast(p=0.1), # Randomly change brightness and contrast of the input image.
        A.RandomFog(p=0.1), # Simulates fog for the image.
        A.RandomRain(p=0.1), # Adds rain effects to an image.
        A.RandomSnow(p=0.1), # Bleach out some pixel values imitating snow.
        A.RandomShadow(p=0.1), # Simulates shadows for the image
        A.RandomSunFlare(p=0.1), # Simulates Sun Flare for the image
        A.ToGray(p=0.1), # Convert the input RGB image to grayscale
    ], p=1.0)
    
    # A.OneOf([
    #     A.GaussNoise(p=1.0), # Apply Gaussian noise to the input image.
    #     A.ISONoise(p=1.0), # Apply camera sensor noise.
    #     A.ImageCompression(quality_lower=50, quality_upper=100, p=1.0), # Decreases image quality by Jpeg, WebP compression of an image.
    #     A.RandomBrightnessContrast(p=1.0), # Randomly change brightness and contrast of the input image.
    #     A.RandomFog(p=1.0), # Simulates fog for the image.
    #     A.RandomRain(p=1.0), # Adds rain effects to an image.
    #     A.RandomSnow(p=1.0), # Bleach out some pixel values imitating snow.
    #     A.RandomShadow(p=1.0), # Simulates shadows for the image
    #     A.RandomSunFlare(p=1.0), # Simulates Sun Flare for the image
    #     A.ToGray(p=1.0), # Convert the input RGB image to grayscale
    # ], p=1.0),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))

def parallelise(function: Callable, data: List, chunksize=100, verbose=True, num_workers=os.cpu_count()) -> List:
    num_workers = 1 if num_workers < 1 else num_workers  # Pool needs to have at least 1 worker.
    pool = Pool(processes=num_workers)
    results = list(
        tqdm.tqdm(pool.imap(function, data, chunksize), total=len(data), disable=not verbose)
    )
    pool.close()
    pool.join()
    return results

def draw_detections(box, name, img):
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))
    
    # 根据图像大小调整矩形框的线宽和文本的大小
    line_thickness = max(1, int(min(height, width) / 200))
    font_scale = min(height, width) / 500
    font_thickness = max(1, int(min(height, width) / 200))
    # 根据图像大小调整文本的纵向位置
    text_offset_y = int(min(height, width) / 50)
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), line_thickness)
    cv2.putText(img, str(name), (xmin, ymin - text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
    return img

def show_labels(images_base_path, labels_base_path):
    if os.path.exists(SHOW_SAVE_PATH):
        shutil.rmtree(SHOW_SAVE_PATH)
    os.makedirs(SHOW_SAVE_PATH, exist_ok=True)
    
    for images_name in tqdm.tqdm(os.listdir(images_base_path)):
        file_heads, _ = os.path.splitext(images_name)
        images_path = f'{images_base_path}/{images_name}'
        labels_path = f'{labels_base_path}/{file_heads}.txt'
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels = np.array(list(map(lambda x:np.array(x.strip().split(), dtype=np.float64), f.readlines())), dtype=np.float64)
            images = cv2.imread(images_path)
            height, width, _ = images.shape
            for cls, x_center, y_center, w, h in labels:
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                draw_detections([x_center - w // 2, y_center - h // 2, x_center + w // 2, y_center + h // 2], CLASSES[int(cls)], images)
            cv2.imwrite(f'{SHOW_SAVE_PATH}/{images_name}', images)
            print(f'{SHOW_SAVE_PATH}/{images_name} save success...')
        else:
            print(f'{labels_path} label file not found...')

def data_aug_single(images_name):
    file_heads, postfix = os.path.splitext(images_name)
    images_path = f'{IMAGE_PATH}/{images_name}'
    labels_path = f'{LABEL_PATH}/{file_heads}.txt'
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = np.array(list(map(lambda x:np.array(x.strip().split(), dtype=np.float64), f.readlines())), dtype=np.float64)
        images = Image.open(images_path)
        for i in range(ENHANCEMENT_LOOP):
            new_images_name = f'{AUG_IMAGE_PATH}/{file_heads}_{i:0>3}{postfix}'
            new_labels_name = f'{AUG_LABEL_PATH}/{file_heads}_{i:0>3}.txt'
            try:
                transformed = ENHANCEMENT_STRATEGY(image=np.array(images), bboxes=np.minimum(np.maximum(labels[:, 1:], 0), 1), class_labels=labels[:, 0])
            except:
                continue
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
            
            cv2.imwrite(new_images_name, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            with open(new_labels_name, 'w+') as f:
                for bbox, cls in zip(transformed_bboxes, transformed_class_labels):
                    f.write(f'{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')
            print(f'{new_images_name} and {new_labels_name} save success...')
    else:
        print(f'{labels_path} label file not found...')

def data_aug():
    if os.path.exists(AUG_IMAGE_PATH):
        shutil.rmtree(AUG_IMAGE_PATH)
    if os.path.exists(AUG_LABEL_PATH):
        shutil.rmtree(AUG_LABEL_PATH)
        
    os.makedirs(AUG_IMAGE_PATH, exist_ok=True)
    os.makedirs(AUG_LABEL_PATH, exist_ok=True)

    for images_name in tqdm.tqdm(os.listdir(IMAGE_PATH)):
        data_aug_single(images_name)
    
if __name__ == '__main__':
    # data_aug()
    
    # show_labels(IMAGE_PATH, LABEL_PATH)
    show_labels(AUG_IMAGE_PATH, AUG_LABEL_PATH)
    