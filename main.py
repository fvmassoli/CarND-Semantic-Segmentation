import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path, print_size):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the graph from file
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # save the grah into a variable
    graph = tf.get_default_graph()

    # get the model layers
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    if print_size:
        tf.Print(keep, [tf.shape(keep)[:]])
        tf.Print(w1, [tf.shape(w1)[:]])
        tf.Print(w3, [tf.shape(w3)[:]])
        tf.Print(w4, [tf.shape(w3)[:]])
        tf.Print(w7, [tf.shape(w7)[:]])

    return w1, keep, w3, w4, w7
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, print_size):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    with tf.name_scope('layer_7'):
        layer7_conv_out = tf.layers.conv2d(vgg_layer7_out,
                                           num_classes,
                                           kernel_size=1,
                                           padding='same',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                           name='layer_7_conv1x1')
        tf.summary.histogram('layer_7_conv1x1_kernel',
                             [v for v in tf.trainable_variables() if v.name == 'layer_7_conv1x1/kernel:0'][0])
        tf.summary.histogram('layer_7_conv1x1_bias',
                             [v for v in tf.trainable_variables() if v.name == 'layer_7_conv1x1/bias:0'][0])

        # upsample
        layer7_upsampling = tf.layers.conv2d_transpose(layer7_conv_out,
                                                       num_classes,
                                                       kernel_size=4,
                                                       padding='same',
                                                       strides=2,
                                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                       name='layer_7_conv1x1_upsampling')
        tf.summary.histogram('layer_7_conv1x1_upsampling_kernel',
                             [v for v in tf.trainable_variables() if v.name == 'layer_7_conv1x1_upsampling/kernel:0'][0])
        tf.summary.histogram('layer_7_conv1x1_upsampling_bias',
                             [v for v in tf.trainable_variables() if v.name == 'layer_7_conv1x1_upsampling/bias:0'][0])

    with tf.name_scope('layer_4'):
        # make the layers having the same size
        layer4_in_b = tf.layers.conv2d(vgg_layer4_out,
                                       num_classes,
                                       kernel_size=1,
                                       padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='layer_4_conv1x1')
        tf.summary.histogram('layer_4_conv1x1_kernel',
                             [v for v in tf.trainable_variables() if v.name == 'layer_4_conv1x1/kernel:0'][0])
        tf.summary.histogram('layer_4_conv1x1_bias',
                             [v for v in tf.trainable_variables() if v.name == 'layer_4_conv1x1/bias:0'][0])

        # skip layer
        layer4_out = tf.add(layer7_upsampling, layer4_in_b, name='skip_layers_7_4')

        # upsample
        layer4_out_upsampling = tf.layers.conv2d_transpose(layer4_out,
                                                           num_classes,
                                                           kernel_size=4,
                                                           padding='same',
                                                           strides=2,
                                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                           name='layer_4_conv1x1_upsampling')
        tf.summary.histogram('layer_4_conv1x1_upsampling_kernel',
                             [v for v in tf.trainable_variables() if v.name == 'layer_4_conv1x1_upsampling/kernel:0'][0])
        tf.summary.histogram('layer_4_conv1x1_upsampling_bias',
                             [v for v in tf.trainable_variables() if v.name == 'layer_4_conv1x1_upsampling/bias:0'][0])

    with tf.name_scope('layer_3'):
        # make the layers having the same size
        layer3_in_b = tf.layers.conv2d(vgg_layer3_out,
                                       num_classes,
                                       kernel_size=1,
                                       padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='layer_3_conv1x1')
        tf.summary.histogram('layer_3_conv1x1_kernel',
                             [v for v in tf.trainable_variables() if v.name == 'layer_3_conv1x1/kernel:0'][0])
        tf.summary.histogram('layer_3_conv1x1_bias',
                             [v for v in tf.trainable_variables() if v.name == 'layer_3_conv1x1/bias:0'][0])

        # skip layer
        layer3_output = tf.add(layer4_out_upsampling, layer3_in_b, name='skip_layers_4_3')

    with tf.name_scope('output'):
        nn_last_layer = tf.layers.conv2d_transpose(layer3_output,
                                                   num_classes,
                                                   kernel_size=16,
                                                   strides=(8, 8),
                                                   padding='same',
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                   name='output')
        tf.summary.histogram('output_kernel',
                             [v for v in tf.trainable_variables() if v.name == 'output/kernel:0'][0])
        tf.summary.histogram('output_bias',
                             [v for v in tf.trainable_variables() if v.name == 'output/bias:0'][0])

    if print_size:
        tf.Print(layer7_conv_out, [tf.shape(layer7_conv_out)[:]])
        tf.Print(layer7_upsampling, [tf.shape(layer7_upsampling)[:]])
        tf.Print(layer4_in_b, [tf.shape(layer4_in_b)[:]])
        tf.Print(layer4_out_upsampling, [tf.shape(layer4_out_upsampling)[:]])
        tf.Print(layer3_in_b, [tf.shape(layer3_in_b)[:]])
        tf.Print(nn_last_layer, [tf.shape(nn_last_layer)[:]])

    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # reshape the output layer from 4D to 2D
    # each row(column) of the output indicates pixel value(class)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    with tf.name_scope('cost'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
        cross_entropy_loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_operation = optimizer.minimize(cross_entropy_loss)
        tf.summary.scalar('cost', cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn,
             train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    print('Training...')

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # creates a single target in order to write all
    # summaries at once
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tensorboard')
    writer.add_graph(sess.graph)

    i = 0
    l_rate = 1e-2
    for epoch in range(epochs):

        print('EPOCH {} ...'.format(epoch+1))

        if epoch % 3 == 0 and epoch != 0:
            l_rate = l_rate*0.8

        for image, label in get_batches_fn(batch_size):
            _, loss, summary = sess.run([train_op, cross_entropy_loss, merged_summary],
                                        feed_dict={input_image: image,
                                                   correct_label: label,
                                                   keep_prob: 0.5,
                                                   learning_rate: 1e-3})
            if i % 5 == 0:
                i += 1
                writer.add_summary(summary, i)

            print('Loss: {:.3f}'.format(loss))

        print()

    saver.save(sess, "model_cpkt/model.ckpt")

#tests.test_train_nn(train_nn)


def run():

    # we'll be doing binary classification.
    # The goal is to recognize road from not-road pixels
    # looking at the training image, they can support up to 3 classes, in order
    # to train on 3 classes you need to modify the helper function that label the training images
    # currently background is False and everything else is True, with 3 use 0, 1 and 2 for
    # red, black and pink pixels
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        # https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        epochs = 15
        batch_size = 10
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, layer_3_out, layer_4_out, layer_7_out = load_vgg(sess=sess,
                                                                                 vgg_path=vgg_path,
                                                                                 print_size=True)

        nn_last_layer = layers(vgg_layer3_out=layer_3_out,
                               vgg_layer4_out=layer_4_out,
                               vgg_layer7_out=layer_7_out,
                               num_classes=num_classes,
                               print_size=True)

        logits, training_operation, cross_entropy_loss = optimize(nn_last_layer=nn_last_layer,
                                                                  correct_label=correct_label,
                                                                  learning_rate=learning_rate,
                                                                  num_classes=num_classes)

        # train FCN
        train_nn(sess=sess,
                 epochs=epochs,
                 batch_size=batch_size,
                 get_batches_fn=get_batches_fn,
                 train_op=training_operation,
                 cross_entropy_loss=cross_entropy_loss,
                 input_image=input_image,
                 correct_label=correct_label,
                 keep_prob=keep_prob,
                 learning_rate=learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


'''

************************************************
************************************************
********* LIST OF TRAINABLE PARAMETERS *********
************************************************
************************************************

<tf.Variable 'conv1_1/filter:0' shape=(3, 3, 3, 64) dtype=float32_ref>
<tf.Variable 'conv1_1/biases:0' shape=(64,) dtype=float32_ref>

<tf.Variable 'conv1_2/filter:0' shape=(3, 3, 64, 64) dtype=float32_ref>
<tf.Variable 'conv1_2/biases:0' shape=(64,) dtype=float32_ref>

<tf.Variable 'conv2_1/filter:0' shape=(3, 3, 64, 128) dtype=float32_ref>
<tf.Variable 'conv2_1/biases:0' shape=(128,) dtype=float32_ref>

<tf.Variable 'conv2_2/filter:0' shape=(3, 3, 128, 128) dtype=float32_ref>
<tf.Variable 'conv2_2/biases:0' shape=(128,) dtype=float32_ref>

<tf.Variable 'conv3_1/filter:0' shape=(3, 3, 128, 256) dtype=float32_ref>
<tf.Variable 'conv3_1/biases:0' shape=(256,) dtype=float32_ref>

<tf.Variable 'conv3_2/filter:0' shape=(3, 3, 256, 256) dtype=float32_ref>
<tf.Variable 'conv3_2/biases:0' shape=(256,) dtype=float32_ref>

<tf.Variable 'conv3_3/filter:0' shape=(3, 3, 256, 256) dtype=float32_ref>
<tf.Variable 'conv3_3/biases:0' shape=(256,) dtype=float32_ref>

<tf.Variable 'conv4_1/filter:0' shape=(3, 3, 256, 512) dtype=float32_ref>
<tf.Variable 'conv4_1/biases:0' shape=(512,) dtype=float32_ref>

<tf.Variable 'conv4_2/filter:0' shape=(3, 3, 512, 512) dtype=float32_ref>
<tf.Variable 'conv4_2/biases:0' shape=(512,) dtype=float32_ref>

<tf.Variable 'conv4_3/filter:0' shape=(3, 3, 512, 512) dtype=float32_ref>
<tf.Variable 'conv4_3/biases:0' shape=(512,) dtype=float32_ref>

<tf.Variable 'conv5_1/filter:0' shape=(3, 3, 512, 512) dtype=float32_ref>
<tf.Variable 'conv5_1/biases:0' shape=(512,) dtype=float32_ref>

<tf.Variable 'conv5_2/filter:0' shape=(3, 3, 512, 512) dtype=float32_ref>
<tf.Variable 'conv5_2/biases:0' shape=(512,) dtype=float32_ref>

<tf.Variable 'conv5_3/filter:0' shape=(3, 3, 512, 512) dtype=float32_ref>
<tf.Variable 'conv5_3/biases:0' shape=(512,) dtype=float32_ref>

<tf.Variable 'fc6/weights:0' shape=(7, 7, 512, 4096) dtype=float32_ref>
<tf.Variable 'fc6/biases:0' shape=(4096,) dtype=float32_ref>

<tf.Variable 'fc7/weights:0' shape=(1, 1, 4096, 4096) dtype=float32_ref>
<tf.Variable 'fc7/biases:0' shape=(4096,) dtype=float32_ref>

<tf.Variable 'layer_7_conv1x1/kernel:0' shape=(1, 1, 4096, 2) dtype=float32_ref>
<tf.Variable 'layer_7_conv1x1/bias:0' shape=(2,) dtype=float32_ref>

<tf.Variable 'layer_7_conv1x1_upsampling/kernel:0' shape=(4, 4, 2, 2) dtype=float32_ref>
<tf.Variable 'layer_7_conv1x1_upsampling/bias:0' shape=(2,) dtype=float32_ref>

<tf.Variable 'layer_4_conv1x1/kernel:0' shape=(1, 1, 512, 2) dtype=float32_ref>
<tf.Variable 'layer_4_conv1x1/bias:0' shape=(2,) dtype=float32_ref>

<tf.Variable 'layer_4_conv1x1_upsampling/kernel:0' shape=(4, 4, 2, 2) dtype=float32_ref>
<tf.Variable 'layer_4_conv1x1_upsampling/bias:0' shape=(2,) dtype=float32_ref>

<tf.Variable 'layer_3_conv1x1/kernel:0' shape=(1, 1, 256, 2) dtype=float32_ref>
<tf.Variable 'layer_3_conv1x1/bias:0' shape=(2,) dtype=float32_ref>

<tf.Variable 'output/kernel:0' shape=(16, 16, 2, 2) dtype=float32_ref>
<tf.Variable 'output/bias:0' shape=(2,) dtype=float32_ref>

'''

if __name__ == '__main__':
    run()
