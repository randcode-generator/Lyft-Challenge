import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import scipy.misc
import cv2

#parameters
cropped_height = 320
cropped_width = 800
resized_height = 128
resized_width = 320
block_width = 160
block_height = 64
width_blocks = 2
height_blocks = 2
total_blocks = width_blocks * height_blocks
epoch = 15
car_prob_thres = 0.1
road_prob_thres = 0.96

file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	_, buffer = cv2.imencode('.png', array)
	return base64.b64encode(buffer).decode("utf-8")

answer_key = {}

# Frame numbering starts at 1
frame = 1

meta_graph = tf.train.import_meta_graph("./model/vehicles.meta")
background_color = np.array([0, 0, 0, 0])

def verify(arr_rgb, arr_seg):
    print(np.array(arr_rgb).shape, np.array(arr_seg).shape)
    c = 0
    a = []
    for _ in range(0, width_blocks):
        b = []
        for _ in range(0, height_blocks):
            gt_bg = np.array(arr_seg[c])
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            mask = np.dot(gt_bg, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            image = scipy.misc.toimage(arr_rgb[c])
            image.paste(mask, box=None, mask=mask)
            b.append(np.array(image))
            c += 1
        a.append(b)
    return a

def padding(arr_seg):
    c = 0
    a = []
    for _ in range(0, width_blocks):
        b = []
        for _ in range(0, height_blocks):
            gt_bg = np.array(arr_seg[c])
            b.append(gt_bg)
            c += 1
        a.append(b)
    h=[]
    for i in a:
        h.append(np.vstack(i))
    f = np.hstack(h)
    print(np.array(f).shape)
    f = cv2.resize(f,(cropped_width, cropped_height))

    g1 = np.zeros((264, 800)).astype('uint8')
    g2 = f.astype('uint8')
    g3 = np.zeros((80, 800)).astype('uint8')
    d = np.vstack((g1, g2, g3)).astype('uint8')
    return d

def windowImage(image, startx, starty, width, height, 
            isFilter=False, filter = [7, 0, 0], isCar = False):
    a=[]
    for i in range(0, width_blocks):
        for j in range(0, height_blocks):
            left = startx + (width * i)
            top = starty + (height * j)
            right = left + width
            bottom = top + height
            arr_img = image[top:bottom, left:right]
            a.append(arr_img)
    return a

def postProcessing(arr_rgb, im_softmax_org):
    print(np.array(im_softmax_org).shape)
    print(np.array(im_softmax_org[:, :, :, 0]).shape)
    im_soft_max_car = np.array(im_softmax_org[:, :, :, 0])
    im_soft_max_car = (im_soft_max_car > car_prob_thres).astype('uint8')
    pCar = padding(im_soft_max_car)

    im_soft_max_road = np.array(im_softmax_org[:, :, :, 1])
    im_soft_max_road = (im_soft_max_road > road_prob_thres).astype('uint8')
    pRoad = padding(im_soft_max_road)

    return (pCar, pRoad)

video = ["1.png", "1.png"]
images = []
images_org = []
seg_org = []
with tf.Session() as sess:
    meta_graph.restore(sess, tf.train.latest_checkpoint('./model'))
    graph = sess.graph
    logits = graph.get_tensor_by_name('fcn_logits:0')
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    for filename in video:
        rgb_frame=scipy.misc.imread(filename)
        images_org.append(rgb_frame)
        rgb_frame = rgb_frame[200:200+cropped_height, 0:0+cropped_width, :]
        rgb_frame = cv2.resize(rgb_frame, (resized_width, resized_height))

        arr_rgb = windowImage(rgb_frame, 0, 0, block_width, block_height)
        arr_rgb = np.array(arr_rgb).reshape(total_blocks * block_height, block_width, 3)
        images.append(arr_rgb)

        if(len(images) == epoch):
            im_softmax_org = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 0.001, input_image: images})
            print(np.array(im_softmax_org).shape)
            im_softmax_org = np.array(im_softmax_org).reshape(len(images), total_blocks, block_height, block_width, 3)
            for x in range(0,len(images)):
                arrs = postProcessing(images[x], im_softmax_org[x])
                answer_key[frame] = [encode(arrs[0]), encode(arrs[1])]
                frame+=1

            images.clear()
        im_softmax_org = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 0.001, input_image: images})
        print(np.array(im_softmax_org).shape)
        im_softmax_org = np.array(im_softmax_org).reshape(len(images), total_blocks, block_height, block_width, 3)
        for x in range(0,len(images)):
            arrs = postProcessing(images[x], im_softmax_org[x])
            seg_org.append(arrs)
            answer_key[frame] = [encode(arrs[0]), encode(arrs[1])]
            frame+=1

gt_bg = np.array(seg_org[0][0])
gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
mask = np.dot(gt_bg, np.array([[0, 255, 0, 127]]))
mask = scipy.misc.toimage(mask, mode="RGBA")
image = scipy.misc.toimage(np.array(images_org[0]))
image.paste(mask, box=None, mask=mask)
scipy.misc.imsave("final_car_2.png", np.array(image))

gt_bg = np.array(seg_org[0][1])
gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
mask = np.dot(gt_bg, np.array([[0, 255, 0, 127]]))
mask = scipy.misc.toimage(mask, mode="RGBA")
image = scipy.misc.toimage(np.array(images_org[0]))
image.paste(mask, box=None, mask=mask)
scipy.misc.imsave("final_road_2.png", np.array(image))