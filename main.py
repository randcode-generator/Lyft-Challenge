import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    return graph.get_tensor_by_name(vgg_input_tensor_name), graph.get_tensor_by_name(vgg_keep_prob_tensor_name), graph.get_tensor_by_name(vgg_layer3_out_tensor_name), graph.get_tensor_by_name(vgg_layer4_out_tensor_name), graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    reg_value = 1e-3
    # TODO: Implement function
    fcn8 = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_value), name="fcn8")

    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=vgg_layer4_out.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_value), name="fcn9")

    fcn9_skip_connected = tf.add(fcn9, vgg_layer4_out, name="fcn9_plus_vgg_layer4")

    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=vgg_layer3_out.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_value), name="fcn10_conv2d")

    fcn10_skip_connected = tf.add(fcn10, vgg_layer3_out, name="fcn10_plus_vgg_layer3")

    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes, kernel_size=16, strides=(8, 8), padding='SAME', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_value), name="fcn11")

    return fcn11

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, name="fcn_train_op")

    return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    print("before global")
    sess.run(tf.global_variables_initializer())

    print("global run")
    for e in range(epochs):
        total_loss = 0
        for image, seg in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op],
                feed_dict = {
                    input_image: image,
                    correct_label: seg,
                    keep_prob: 0.001,
                    learning_rate: 0.0001
                })
            total_loss += loss
        print("EPOCH {}...".format(e))
        print("Loss = {:.3f}".format(total_loss))

def run():
    print("starting")
    num_classes = 3
    image_shape = (64, 160)
    data_dir = '/tmp/data'
    runs_dir = './runs'

    # # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'Train'), image_shape)

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        print("vgg loaded")
        last_layer = layers(layer3, layer4, layer7, num_classes)
        print("last layer")

        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(last_layer,correct_label, learning_rate, num_classes )
        print("optimize")
        
        # TODO: Train NN using the train_nn function
        epochs = 1
        batch_size = 16
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        print("train_nn")

        saver = tf.train.Saver()
        saver.save(sess, './vehicles')
        print("model saved")

if __name__ == '__main__':
    run()
