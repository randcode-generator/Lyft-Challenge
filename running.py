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
image_shape = (160, 576)

background_color = np.array([0, 0, 0, 0])

with tf.Session() as sess:
    meta_graph.restore(sess, tf.train.latest_checkpoint('/tmp/model'))
    graph = sess.graph
    logits = graph.get_tensor_by_name('fcn_logits:0')
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    for rgb_frame in video:
        image = scipy.misc.imresize(rgb_frame, image_shape)
        im_softmax_org = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 0.001, input_image: [image]})
    
        im_softmax = im_softmax_org[0][:, 0].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.3).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.imresize(mask, (600,800))
        t_f_vehicle_array = np.invert(np.all(street_im == background_color, axis=2)).astype('uint8')
        t_f_vehicle_array[496:] = 0

        im_softmax = im_softmax_org[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.3).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.imresize(mask, (600,800))
        t_f_road_array = np.invert(np.all(street_im == background_color, axis=2)).astype('uint8')

        answer_key[frame] = [encode(t_f_vehicle_array), encode(t_f_road_array)]
        frame+=1

print (json.dumps(answer_key))

# for rgb_frame in video:
	
#     # Grab red channel	
# 	red = rgb_frame[:,:,0]    
#     # Look for red cars :)
# 	binary_car_result = np.where(red>250,1,0).astype('uint8')
    
#     # Look for road :)
# 	binary_road_result = binary_car_result = np.where(red<20,1,0).astype('uint8')

# 	answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
#     # Increment frame
# 	frame+=1

# Print output in proper json format
#print (json.dumps(answer_key))