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
    a = []
    counter = 0
    for i in range(0, len(arr_rgb)):
        b = []
        for j in range(0, len(arr_rgb[i])):
            gt_bg = np.array(arr_seg[i][j])
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            mask = np.dot(gt_bg, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            image = scipy.misc.toimage(arr_rgb[i][j])
            image.paste(mask, box=None, mask=mask)
            b.append(np.array(image))
            counter += 1
        a.append(b)
    return a

def windowImage(image, startx, starty, width, height, 
            isFilter=False, filter = [7, 0, 0], isCar = False):
    if(isFilter):
        background_color = np.array(filter)
        gt_bg = np.all(image == background_color, axis=2).astype('uint8')
        if(isCar):
            gt_bg[495:] = False
            print("called")
        image = gt_bg

    a=[]
    for i in range(0, 5):
        b=[]
        for j in range(0, 4):
            left = startx + (width * i)
            top = starty + (height * j)
            right = left + width
            bottom = top + height
            arr_img = image[top:bottom, left:right]
            b.append(arr_img)
        a.append(b)
    return a

def gen_batch_function(data_folder, image_shape):
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
            startx = 0
            starty = 264
            width = image_shape[1]
            height = image_shape[0]
            for rgb_image_file in rgb_paths[batch_i:batch_i+batch_size]:
                filename_w_ext = os.path.basename(rgb_image_file)
                filename, _ = os.path.splitext(filename_w_ext)
                seg_image_file = os.path.join(data_folder, 'CameraSeg', filename+".png")

                rgb_image = scipy.misc.imread(rgb_image_file)
                seg_image = scipy.misc.imread(seg_image_file)

                arr_rgb = windowImage(rgb_image, startx, starty, width, height)
                arr_seg = windowImage(seg_image, startx, starty, width, height, 
                    isFilter = True, filter = [7, 0, 0], isCar = False)

            yield np.array(arr_rgb), np.array(arr_seg)
    return get_batches_fn
