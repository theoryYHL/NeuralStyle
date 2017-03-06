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
# 많은 예시들에서는 relu layer를 사용하는데 논문에서는 conv layer를 사용함
CONTENT_LAYERS = ('conv3_1') # 높으면 높을수록 원본 이미지가 깨진다
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')

FLAGS = tf.app.flags.FLAGS


# 혹시나 tensorboard가 보고 싶을까봐 namescope를 지정함
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

# 실제 net을 구성하는 함수
# current_layer를 넣어주면 그 위로 VGG19를 쌓아주는 함수이다
# output으로 layer를 내놓는데 이 값은 각 레이어의 출력 텐서가 저장된 딕셔너리다
# layer[conv1_1]에는 conv1_1의 output 텐서가 저장됨
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
    # 왜 있는지는 모르겠지만... 첫 레이어의 input을 대략 평균 값이 0가 되도록 시키기 위한 장치인듯
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
    # 사용할 이미지 받기
    content_image = scipy.misc.imread(FLAGS.content_path)
    content_size = (1,)+content_image.shape
    style_image = scipy.misc.imread(FLAGS.style_path)
    style_size = (1,) + style_image.shape

    # content feature와 style feature를 저장
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
            # 결국 gram은 i번째 커널과 j번째 커널의 값을 모든 위치에서 다 곱하고 평균내는 것이다
            feature = sess.run(layers[name], feed_dict={image_input: style_input})
            feature_vec = np.reshape(feature,(-1,feature.shape[3]))
            gram = np.matmul(feature_vec.T, feature_vec) / feature_vec.size
            style_save[name]=gram

    # 새로운 이미지를 만든다
    content_per_style = 1e-3 # 논문에 있던 content loss와 style loss의 적용 비율
    # 굳이 random에서 시작하지 말고 content 이미지에서 시작한다
    # 논문에서는 random에서 시작한다
    mix_image = tf.Variable(content_input, dtype='float32')
    current = mix_image
    layers, mean_pixel = net(current)
    content_loss = content_per_style*tf.nn.l2_loss(layers[CONTENT_LAYERS]-content_save)
    # style_loss는 여러 레이어에서 정해지므로 losses를 통해 전부 구하고 나중에 합친다.
    style_losses = []
    for name in STYLE_LAYERS:
        mix_feature = layers[name]
        _,h,w,n = mix_feature.get_shape()
        size = h*w*n
        mix_feature_vec = tf.reshape(mix_feature,[-1,mix_feature.get_shape()[3].value])
        gram = tf.matmul(tf.transpose(mix_feature_vec),mix_feature_vec)/size.value # 여기서 .value 안했다가 오류난거 고치느라 힘들었음
        style_losses.append(tf.nn.l2_loss(gram-style_save[name]))
    #원래 논문에서는 style_loss에 1/5를 곱해야 하지만... 어차피 content_per_style 값에 의해 의미가 없어진다
    style_loss = tf.reduce_sum(style_losses)
    total_loss = content_loss+style_loss

    # 실행
    rate = 1.0
    optimizer = tf.train.AdamOptimizer(rate).minimize(total_loss)
    init = tf.global_variables_initializer()
    iteration = 1000
    with tf.Session() as sess:
        sess.run(init)
        for it in range(iteration):
            print it
            sess.run(optimizer)
            """
            중간 과정도 보고싶다면 주석을 풀자
            if it%100==1:
                mid = np.squeeze(mix_image.eval(), axis=0) + mean_pixel
                scipy.misc.imsave("./images/mix"+str(it)+".png", mid)
            """
        final = np.squeeze(mix_image.eval(),axis=0)+mean_pixel
        scipy.misc.imsave(FLAGS.mix_path,final)


if __name__ == "__main__":
    tf.app.run()