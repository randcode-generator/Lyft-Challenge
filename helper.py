import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2

#parameters
resized_height = 192
resized_width = 480
cropped_height = 320
cropped_width = 800

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def verify(arr_rgb, arr_seg):
    gt_bg = np.array(arr_seg)
    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    mask = np.dot(gt_bg, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    image = scipy.misc.toimage(arr_rgb)
    image.paste(mask, box=None, mask=mask)
    return image

def filterImage(image, filter = [7, 0, 0], isCar = False):
    background_color = np.array(filter)
    gt_bg = np.all(image == background_color, axis=2).astype('float32')
    if(isCar == True):
        gt_bg[296:] = 0.
    return gt_bg

def gen_batch_function(data_folder):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        rgb_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        random.shuffle(rgb_paths)
        for batch_i in range(0, len(rgb_paths), batch_size):
            rgb_images = []
            seg_images = []
            for rgb_image_file in rgb_paths[batch_i:batch_i+batch_size]:
                filename_w_ext = os.path.basename(rgb_image_file)
                filename, _ = os.path.splitext(filename_w_ext)
                seg_image_file = os.path.join(data_folder, 'CameraSeg', filename+".png")

                rgb_image = scipy.misc.imread(rgb_image_file)
                seg_image = scipy.misc.imread(seg_image_file)

                #CROP IMAGE
                rgb_image = rgb_image[200:200+cropped_height, 0:0+cropped_width, :]
                seg_image = seg_image[200:200+cropped_height, 0:0+cropped_width, :]

                #FILTER SEG_IMAGES
                arr_seg_car = filterImage(seg_image, filter = [10, 0, 0], isCar = True)
                arr_seg_road = filterImage(seg_image, filter = [7, 0, 0])
                arr_seg_lanelines = filterImage(seg_image, filter = [6, 0, 0])
                arr_seg_road = np.logical_or(arr_seg_road, arr_seg_lanelines).astype('uint8')

                #RESIZE
                arr_rgb = cv2.resize(rgb_image, (resized_width, resized_height))
                arr_seg_car = cv2.resize(arr_seg_car, (resized_width, resized_height))
                arr_seg_road = cv2.resize(arr_seg_road, (resized_width, resized_height))

                #ANYTHING NOT CAR OR ROAD
                arr_everything_else = np.invert(np.logical_or(arr_seg_car, arr_seg_road))

                #BREAK SEG_IMAGES INTO PARTS
                h1 = np.array(arr_seg_car).flatten()
                h2 = np.array(arr_seg_road).flatten()
                h3 = np.array(arr_everything_else).flatten()
                arr_seg = np.vstack((h1, h2, h3)).T
                arr_seg = arr_seg.reshape((resized_height, resized_width, 3))

                rgb_images.append(arr_rgb)
                seg_images.append(arr_seg)

            print("lol ", np.array(rgb_images).shape)
            print("lol ", np.array(seg_images).shape)

            a = verify(np.array(rgb_images[0]), np.array(seg_images[0][:,:,1]))
            scipy.misc.imsave(filename+"output_road.png", a)
            scipy.misc.imsave(filename+"real_output_road.png", a.resize((800, 320)))

            a = verify(np.array(rgb_images[0]), np.array(seg_images[0][:,:,0]))
            scipy.misc.imsave(filename+"output_car.png", a)            
            scipy.misc.imsave(filename+"real_output_car.png", a.resize((800, 320)))

            yield np.array(rgb_images), np.array(seg_images)
    return get_batches_fn