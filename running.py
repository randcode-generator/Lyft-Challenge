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

def padding(arr_seg):
    c = 0
    a = []
    for _ in range(0, 5):
        b = []
        for _ in range(0, 4):
            gt_bg = np.array(arr_seg[c])
            b.append(gt_bg)
            c += 1
        a.append(b)
    h=[]
    for i in a:
        h.append(np.vstack(i))
    f = np.hstack(h)
    print(np.array(f).shape)
    g1 = np.zeros((264, 800)).astype('uint8')
    g2 = f.astype('uint8')
    g3 = np.zeros((80, 800)).astype('uint8')
    d = np.vstack((g1, g2, g3)).astype('uint8')
    return d

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

def postProcessing(arr_rgb, im_softmax_org, image_shape):
    print(np.array(im_softmax_org).shape)
    print(np.array(im_softmax_org[:, :, :, 0]).shape)
    im_soft_max_car = np.array(im_softmax_org[:, :, :, 0])
    im_soft_max_car = (im_soft_max_car > 0.5).astype('uint8')
    pCar = padding(im_soft_max_car)

    im_soft_max_road = np.array(im_softmax_org[:, :, :, 1])
    im_soft_max_road = (im_soft_max_road > 0.5).astype('uint8')
    pRoad = padding(im_soft_max_road)

    return (pCar, pRoad)

video = ["0.png", "1.png"]
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
        startx = 0
        starty = 264
        width = image_shape[1]
        height = image_shape[0]
        arr_rgb = windowImage(rgb_frame, startx, starty, width, height)
        arr_rgb = np.array(arr_rgb).reshape((20*64, 160, 3))
        images.append(arr_rgb)

        if(len(images) == 10):
            im_softmax_org = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 0.001, input_image: images})
            print(np.array(im_softmax_org).shape)
            im_softmax_org = np.array(im_softmax_org).reshape(len(images), 20, 64, 160, 3)
            for x in range(0,len(images)):
                arrs = postProcessing(images[x], im_softmax_org[x], image_shape)
                answer_key[frame] = [encode(arrs[0]), encode(arrs[1])]
                frame+=1

            images.clear()
        im_softmax_org = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 0.001, input_image: images})
        print(np.array(im_softmax_org).shape)
        im_softmax_org = np.array(im_softmax_org).reshape(len(images), 20, 64, 160, 3)
        for x in range(0,len(images)):
            arrs = postProcessing(images[x], im_softmax_org[x], image_shape)
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