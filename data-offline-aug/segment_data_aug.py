import warnings
warnings.filterwarnings('ignore')
import os, shutil, cv2, tqdm
import numpy as np
np.random.seed(0)
import albumentations as A
from PIL import Image
from multiprocessing import Pool
from typing import Callable, Dict, List, Union

# https://github.com/albumentations-team/albumentations

def generate_color_map(num_classes):
    hsv_colors = [(i * 180 // num_classes, 255, 255) for i in range(num_classes)]
    rgb_colors = [[0, 0, 0]] + [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2BGR)[0][0] for color in hsv_colors]
    return np.array(rgb_colors, dtype=np.uint8)

IMAGE_PATH = 'dataset/segment/images'
LABEL_PATH = 'dataset/segment/labels'
AUG_IMAGE_PATH = 'dataset/segment/images_aug'
AUG_LABEL_PATH = 'dataset/segment/labels_aug'
SHOW_SAVE_PATH = 'results'
COLORS = generate_color_map(20)

ENHANCEMENT_LOOP = 1
ENHANCEMENT_STRATEGY = A.Compose([
    A.Compose([
        A.Affine(scale=[0.5, 1.5], translate_percent=[0.0, 0.3], rotate=[-360, 360], shear=[-45, 45], keep_ratio=True, cval_mask=0, p=0.5), # Augmentation to apply affine transformations to images.
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
], is_check_shapes=False)

def draw_segments(image, mask):
    blended_image = cv2.addWeighted(image, 0.7, COLORS[mask], 0.3, 0)
    return blended_image

def show_labels(images_base_path, labels_base_path):
    if os.path.exists(SHOW_SAVE_PATH):
        shutil.rmtree(SHOW_SAVE_PATH)
    os.makedirs(SHOW_SAVE_PATH, exist_ok=True)
    
    for images_name in tqdm.tqdm(os.listdir(images_base_path)):
        file_heads, _ = os.path.splitext(images_name)
        # images_path = f'{images_base_path}/{images_name}'
        images_path = os.path.join(images_base_path, images_name)
        # labels_path = f'{labels_base_path}/{file_heads}.png'
        labels_path = os.path.join(labels_base_path, f'{file_heads}.png')
        if os.path.exists(labels_path):
            images = cv2.imread(images_path)
            masks = np.array(Image.open(labels_path))
            print(np.unique(masks))
            images = draw_segments(images, masks)
            cv2.imwrite(f'{SHOW_SAVE_PATH}/{images_name}', images)
            print(f'{SHOW_SAVE_PATH}/{images_name} save success...')
        else:
            print(f'{labels_path} label file not found...')

def data_aug_single(images_name):
    file_heads, postfix = os.path.splitext(images_name)
    # images_path = f'{IMAGE_PATH}/{images_name}'
    images_path = os.path.join(IMAGE_PATH, images_name)
    # labels_path = f'{LABEL_PATH}/{file_heads}.jpg'
    labels_path = os.path.join(LABEL_PATH, f'{file_heads}.jpg')
    if os.path.exists(labels_path):
        images = Image.open(images_path)
        masks = np.array(Image.open(labels_path))
        for i in range(ENHANCEMENT_LOOP):
            # new_images_name = f'{AUG_IMAGE_PATH}/{file_heads}_{i:0>3}{postfix}'
            new_images_name = os.path.join(AUG_IMAGE_PATH, f'{file_heads}_{i:0>3}{postfix}')
            # new_labels_name = f'{AUG_LABEL_PATH}/{file_heads}_{i:0>3}.png'
            new_labels_name = os.path.join(AUG_LABEL_PATH, f'{file_heads}_{i:0>3}.png')
            try:
                transformed = ENHANCEMENT_STRATEGY(image=np.array(images), masks=[masks])
            except:
                continue
            transformed_image = transformed['image']
            transformed_masks = transformed['masks'][0]
            
            cv2.imwrite(new_images_name, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            Image.fromarray(np.array(transformed_masks)).save(new_labels_name)
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
    show_labels(IMAGE_PATH, LABEL_PATH)
    # show_labels(AUG_IMAGE_PATH, AUG_LABEL_PATH)
    
    # data_aug()