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
image_shape = (160, 576)

background_color = np.array([0, 0, 0, 0])

def verify(true_false, rgb_image):
    print(np.unique(true_false))
    mask = true_false.reshape(*true_false.shape, 1)
    print(mask.shape)
    mask = np.dot(mask, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(rgb_image)
    street_im.paste(mask, box=None, mask=mask)
    return street_im

with tf.Session() as sess:
    meta_graph.restore(sess, tf.train.latest_checkpoint('./model'))
    graph = sess.graph
    logits = graph.get_tensor_by_name('fcn_logits:0')
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    image_org=scipy.misc.imread("246.png")
    image = scipy.misc.imresize(image_org, image_shape)
    im_softmax_org = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 0.001, input_image: [image]})

    im_softmax = im_softmax_org[0][:, 0].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.imresize(mask, (600,800))
    scipy.misc.imsave("mask.png", np.array(street_im))
    t_f_vehicle_array = np.invert(np.all(street_im == background_color, axis=2)).astype('uint8')

    t_f_vehicle_array[496:] = 0
    c = verify(t_f_vehicle_array, image_org)
    scipy.misc.imsave("final.png", np.array(c))

    im_softmax = im_softmax_org[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.3).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.imresize(mask, (600,800))
    t_f_road_array = np.invert(np.all(street_im == background_color, axis=2)).astype('uint8')

    c = verify(t_f_road_array, image_org)
    scipy.misc.imsave("final_2.png", np.array(c))
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