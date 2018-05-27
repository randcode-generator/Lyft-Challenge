import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import scipy.misc

file = sys.argv[-1]

if file == 'running.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

meta_graph = tf.train.import_meta_graph("/tmp/model/vehicles.meta")
image_shape = (96, 128)
total_size = image_shape[0]*image_shape[1]

background_color = np.array([0, 0, 0, 0])

def postProcessing(im_softmax_org, image_shape):
    im_softmax = im_softmax_org[0][:, 0].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax >= 1.0).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.imresize(mask, (600,800))
    t_f_vehicle_array = np.invert(np.all(street_im == background_color, axis=2)).astype('uint8')
    t_f_vehicle_array[496:] = 0
    
    im_softmax = im_softmax_org[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax >= 1.0).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.imresize(mask, (600,800))
    t_f_road_array = np.invert(np.all(street_im == background_color, axis=2)).astype('uint8')
    
    return (t_f_vehicle_array, t_f_road_array)

images = []
with tf.Session() as sess:
    meta_graph.restore(sess, tf.train.latest_checkpoint('/tmp/model'))
    graph = sess.graph
    logits = graph.get_tensor_by_name('fcn_logits:0')
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    for rgb_frame in video:
        image = scipy.misc.imresize(rgb_frame, image_shape)
        images.append(image)

        if(len(images) == 25):
            im_softmax_org = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 0.001, input_image: images})

            im_softmax_org = np.array(im_softmax_org).reshape(len(images), total_size, 3)
            for x in range(0,len(images)):
                arrs = postProcessing([im_softmax_org[x]], image_shape)
                answer_key[frame] = [encode(arrs[0]), encode(arrs[1])]
                frame+=1

            images.clear()
    
    if(len(images) > 0):
        im_softmax_org = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 0.001, input_image: images})
        im_softmax_org = np.array(im_softmax_org).reshape(len(images), total_size, 3)
        for x in range(0,len(images)):
            arrs = postProcessing([im_softmax_org[x]], image_shape)
            answer_key[frame] = [encode(arrs[0]), encode(arrs[1])]
            frame+=1

print (json.dumps(answer_key))