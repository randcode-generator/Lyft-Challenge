import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import scipy.misc

file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

answer_key = {}

# Frame numbering starts at 1
frame = 1

meta_graph = tf.train.import_meta_graph("./model/vehicles.meta")
image_shape = (64, 160)
total_size = image_shape[0]*image_shape[1]

background_color = np.array([0, 0, 0, 0])

def verify(arr_rgb, arr_seg):
    print(np.array(arr_rgb).shape, np.array(arr_seg).shape)
    c = 0
    a = []
    for _ in range(0, 5):
        b = []
        for _ in range(0, 4):
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

def windowImage(image, startx, starty, width, height, 
            isFilter=False, filter = [7, 0, 0], isCar = False):
    if(isFilter):
        background_color = np.array(filter)
        gt_bg = np.all(image == background_color, axis=2).astype('uint8')
        if(isCar):
            gt_bg[495:] = False
        image = gt_bg

    a=[]
    for i in range(0, 5):
        for j in range(0, 4):
            left = startx + (width * i)
            top = starty + (height * j)
            right = left + width
            bottom = top + height
            arr_img = image[top:bottom, left:right]
            a.append(arr_img)
    return a

filename = "0.png"
with tf.Session() as sess:
    meta_graph.restore(sess, tf.train.latest_checkpoint('./model'))
    graph = sess.graph
    logits = graph.get_tensor_by_name('fcn_logits:0')
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    rgb_image=scipy.misc.imread(filename)
    startx = 0
    starty = 264
    width = image_shape[1]
    height = image_shape[0]
    arr_rgb = windowImage(rgb_image, startx, starty, width, height)

    print(np.array(arr_rgb).shape)
    arr_rgb1 = np.array(arr_rgb).reshape((20*64, 160, 3))

    im_softmax_org = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 0.001, input_image: [arr_rgb1]})

    print(np.array(im_softmax_org).shape)

    im_soft_max1 = np.array(im_softmax_org[0][:, 0]).reshape(20, 64, 160)

    v = verify(arr_rgb, (im_soft_max1 > 0.3).astype('uint8'))
    h=[]
    for i in v:
        h.append(np.vstack(i))
    scipy.misc.imsave("output_car.png", np.hstack(h))

    im_soft_max1 = np.array(im_softmax_org[0][:, 1]).reshape(20, 64, 160)

    v = verify(arr_rgb, (im_soft_max1 > 0.3).astype('uint8'))
    h=[]
    for i in v:
        h.append(np.vstack(i))
    scipy.misc.imsave("output_road.png", np.hstack(h))

    # segmentation = (arr_seg > .5).astype('uint8')
    # print(np.unique(np.array(segmentation)))

    # goin = arr_seg[:,:,:,0]
    # goin = (goin > .5).astype('uint8')
    # a1 = verify(arr_rgb, goin)
    # h=[]
    # for i in a1:
    #     h.append(np.vstack(i))
    # scipy.misc.imsave("output_car.png", np.hstack(h))

    # goin = arr_seg[:,:,:,1]
    # goin = (goin > .9).astype('uint8')
    # a1 = verify(arr_rgb, goin)
    # h=[]
    # for i in a1:
    #     h.append(np.vstack(i))
    # scipy.misc.imsave("output_road.png", np.hstack(h))

    # goin = arr_seg[:,:,:,2]
    # goin = (goin > .5).astype('uint8')
    # a1 = verify(arr_rgb, goin)
    # h=[]
    # for i in a1:
    #     h.append(np.vstack(i))
    # scipy.misc.imsave("output_inverted.png", np.hstack(h))
