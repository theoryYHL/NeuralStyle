# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.io
import scipy.misc
import numpy as np

tf.app.flags.DEFINE_string("vgg_path", "./imagenet-vgg-verydeep-19.mat", "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("content_path", "./images/content.jpg", "content 이미지 위치")
tf.app.flags.DEFINE_string("style_path", "./images/style.jpg", "style 이미지 위치")
tf.app.flags.DEFINE_string("mix_path", "./images/mix.png", "새로운 이미지 저장 위치")
tf.app.flags.DEFINE_boolean("max_pool", True, "pooling을 max로 사용할지. False인 경우 average pooling을 사용")
CONTENT_LAYERS = ('conv3_1')
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')

FLAGS = tf.app.flags.FLAGS



def conv_namescope(input,weights_conv,bias_conv,namescope):
    with tf.variable_scope(namescope) as scope:
        # matconvnet: weights are [width, height, in_channels, out_channels]
        # tensorflow: weights are [height, width, in_channels, out_channels]
        kernels = tf.constant(np.transpose(weights_conv,(1,0,2,3)),dtype=None)
        bias = tf.constant(bias_conv)
        layer = tf.add(tf.nn.conv2d(input,kernels,strides=[1,1,1,1],padding='SAME'),bias)
        return layer

def relu_namescope(input,namescope):
    with tf.variable_scope(namescope) as scope:
        layer = tf.nn.relu(input)
        return layer

def pool_namescope(input,namescope):
    with tf.variable_scope(namescope) as scope:
        if FLAGS.max_pool:
            layer = tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        else:
            layer = tf.nn.avg_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return layer

def net(current_layer):
    VGG19_LAYERS = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    layers={}
    current = current_layer
    data = scipy.io.loadmat(FLAGS.vgg_path)
    """
    weights = data['layers'][0]
    kernels = weights[i][0][0][0][0][0]
    bias = weights[i][0][0][0][0][1]
    """
    weights = data['layers'][0]
    """why mean is necessary?"""
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    for i, name in enumerate(VGG19_LAYERS):
        if(name[:4]=="conv"):
            current = conv_namescope(current,weights[i][0][0][0][0][0],weights[i][0][0][0][0][1],name)
        elif(name[:4]=="relu"):
            current = relu_namescope(current, name)
        else:
            current = pool_namescope(current, name)
        layers[name] = current
    assert len(VGG19_LAYERS) == len(layers)
    return layers,mean_pixel



def main(_):

    content_image = scipy.misc.imread(FLAGS.content_path)
    content_size = (1,)+content_image.shape
    style_image = scipy.misc.imread(FLAGS.style_path)
    style_size = (1,) + style_image.shape

    image_input = tf.placeholder('float')
    current = image_input
    layers,mean_pixel = net(current)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(init)
        # content info save
        content_input = np.reshape((content_image - mean_pixel), content_size)
        content_save = layers[CONTENT_LAYERS].eval(feed_dict={image_input:content_input})
        # style info save
        style_input = np.reshape((style_image - mean_pixel), style_size)
        style_save = {}

        for i, name in enumerate(STYLE_LAYERS):
            feature = sess.run(layers[name], feed_dict={image_input: style_input})
            feature_vec = np.reshape(feature,(-1,feature.shape[3]))
            gram = np.matmul(feature_vec.T, feature_vec) / feature_vec.size
            style_save[name]=gram

    content_per_style = 1e-3
    mix_image = tf.Variable(content_input,dtype='float32')
    current = mix_image
    layers, mean_pixel = net(current)
    content_loss = content_per_style*tf.nn.l2_loss(layers[CONTENT_LAYERS]-content_save)
    style_losses = []
    for name in STYLE_LAYERS:
        mix_feature = layers[name]
        _,h,w,n = mix_feature.get_shape()
        size = h*w*n
        mix_feature_vec = tf.reshape(mix_feature,[-1,mix_feature.get_shape()[3].value])
        gram = tf.matmul(tf.transpose(mix_feature_vec),mix_feature_vec)/size.value
        style_losses.append(tf.nn.l2_loss(gram-style_save[name]))
    style_loss = tf.reduce_sum(style_losses)

    total_loss = content_loss+style_loss
    rate = 1.0
    optimizer = tf.train.AdamOptimizer(rate).minimize(total_loss)
    init = tf.global_variables_initializer()
    iteration = 1000
    with tf.Session() as sess:
        sess.run(init)
        for it in range(iteration):
            print it
            sess.run(optimizer)
            if it%100==1:
                mid = np.squeeze(mix_image.eval(), axis=0) + mean_pixel
                scipy.misc.imsave("./images/mix"+str(it)+".png", mid)
        final = np.squeeze(mix_image.eval(),axis=0)+mean_pixel
        scipy.misc.imsave(FLAGS.mix_path,final)


if __name__ == "__main__":
    tf.app.run()