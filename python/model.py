
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
from tensorflow.contrib import slim
import scipy.misc
from dataset import load_renders, generator

renders = load_renders('/home/sheen/renders2')
gen = generator(renders)

parser = argparse.ArgumentParser()
parser.add_argument("--ngf", type=int, default=8, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=8, help="number of discriminator filters in first conv layer")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.9, help="momentum term of adam")


a = parser.parse_args()
a.model_name = '12th'
checkpoint_dir = 'checkpoints/'
continue_training=False
BS = 3
number_of_types=6

input_images = tf.placeholder('float32',[None,256,256,3])
type_annealing = tf.placeholder('float32',[])
input_types = tf.placeholder('float32',[None,number_of_types]) #one hot types
random_type = tf.placeholder('float32',[None,number_of_types]) #one hot; different than input_types
# tiled_types = tf.contrib.keras.backend.repeat_elements(input_types,number_of_types, axis=0)

def normalize(images):
    return ([image/127.5-1 for image in images])

def unnormalize(images):
    return np.uint8((images+1)*127.5)
    
def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def lrelu(x, a=.2):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)



def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)



def create_generator(generator_inputs):
    print('gen:',generator_inputs.shape)
    if False:
        return tf.stack([generator_inputs for x in range(6)],axis=1)
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            # output = slim.layer_norm(convolved)
            output = instance_norm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = lrelu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = slim.layer_norm(output)
            '''remove this during testing?'''
            if dropout > 0.0:
                # if a.mode=='train':
                if True:
                    print("using dropout")
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
    generator_outputs_channels = 3*number_of_types
    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = lrelu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)
    print(layers[-1].shape)
    output = tf.reshape(layers[-1],[-1, number_of_types, 256,256,3])
    return output




def create_type_discriminator(discrim_inputs):
    print('type disc:',discrim_inputs.shape)
    if False:
        return tf.ones_like(input_types)
    dim = 32
    net = tf.pad(discrim_inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    net = lrelu(slim.conv2d(net, dim, [4,4], stride=2))
    print(net.shape)
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    net = lrelu(tf.nn.dropout(slim.layer_norm(slim.conv2d(net,  dim*2,[4,4], stride=2)),.5))
    print(net.shape)
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    net = lrelu( tf.nn.dropout(slim.layer_norm(slim.conv2d(net, dim*4, [4,4], stride=2)),.5))
    print(net.shape)
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    net = lrelu( tf.nn.dropout(slim.layer_norm(slim.conv2d(net, dim*8, [4,4], stride=2)),.5))
    print(net.shape)
    #net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    net = (slim.conv2d(net, number_of_types, [1,1], stride=1)) #BS, 20, 20, 6
    print(net.shape)
    logits = tf.reduce_mean(net,axis=[1,2])
    return logits
    

def create_discriminator(discrim_inputs):
    print('quality disc:',discrim_inputs.shape)
    if False:
        return tf.reduce_mean(discrim_inputs)
    
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    # input = tf.concat([discrim_inputs, discrim_targets], axis=3)
    input = discrim_inputs
    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = discrim_conv(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = a.ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            # normalized = batchnorm(convolved)
            normalized = slim.layer_norm(convolved)
            rectified = lrelu(normalized, 0.2)
            print(rectified.shape)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]




with tf.name_scope("gen1"):
    with tf.variable_scope("generator"):
        fake_spectrum = create_generator(input_images) #BS, types, 256, 256, 3


with tf.name_scope("real_quality_discriminator"):
    with tf.variable_scope("quality_discriminator"):
        predict_quality_real = create_discriminator(input_images)

random_fakes = tf.einsum('abcde,ab->acde',fake_spectrum, random_type)


with tf.name_scope("fake_quality_discriminator"):
    with tf.variable_scope("quality_discriminator", reuse=True):
        predict_quality_fake = create_discriminator(random_fakes)


EPS = 1e-4
quality_discrim_loss = tf.reduce_mean(-(tf.log(predict_quality_real + EPS) + tf.log(1 - predict_quality_fake + EPS))) * 1
gen_quality_loss = tf.reduce_mean(-tf.log(predict_quality_fake + EPS)) * 1




batched_fake_spectrum = tf.reshape(fake_spectrum, [-1, 256,256,3]) # #BS, types, 256, 256, 3 -> #BS*types, 256, 256, 3

with tf.name_scope("fake_type_discriminator"):
    with tf.variable_scope("type_discriminator"):
        predict_type_fake = create_type_discriminator(batched_fake_spectrum)
print('ptf out:',predict_type_fake.shape)
predict_type_fake = tf.reshape(predict_type_fake, [-1,6,6])

with tf.name_scope("real_type_discriminator"):
    with tf.variable_scope("type_discriminator", reuse=True):
        predict_type_real= create_type_discriminator(input_images)


spectrum_types = tf.constant(np.diag([1,1,1,1,1,1]),dtype='float32')
reshaped_spectrum_types = predict_type_fake * 0 + tf.expand_dims(spectrum_types,0)
type_gen_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_type_fake, labels = reshaped_spectrum_types))*type_annealing
type_discrim_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict_type_real, labels = input_types))*type_annealing

type_discrim_loss = (type_discrim_loss_real) - (type_gen_loss_fake)



with tf.name_scope("gen2"):
    with tf.variable_scope("generator", reuse=True):
        fake_output = create_generator(batched_fake_spectrum) #BS*types, types, 256, 256, 3


fake_output_restacked = tf.reshape(fake_output, [-1, number_of_types, number_of_types, 256,256,3]) #BS, types, types, 256, 256, 3



# fake_output_of_the_original_type = tf.reduce_sum(fake_output_restacked * tf.reshape(input_types,[-1,1,6,1,1,1]), axis=2) #zero out types other than the original type

fake_output_of_the_original_type2 = tf.einsum('abcdef,ac->abdef',fake_output_restacked, input_types)

reconstruction_loss = tf.reduce_mean(tf.square(tf.expand_dims(input_images,1) - fake_output_of_the_original_type2)) * 0
gen_loss = reconstruction_loss + type_gen_loss_fake + gen_quality_loss


gen_loss_summary = tf.summary.scalar("gen_loss", gen_loss)
reconstruction_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
type_gen_loss_fake_summary = tf.summary.scalar("type_gen_loss_fake", type_gen_loss_fake)
gen_quality_loss_summary = tf.summary.scalar("gen_quality_loss", gen_quality_loss)


type_discrim_loss_summary = tf.summary.scalar("type_discrim_loss", type_discrim_loss)
quality_discrim_loss_summary = tf.summary.scalar("quality_discrim_loss", quality_discrim_loss)

summary_op = tf.summary.merge_all()


global_step = tf.train.get_or_create_global_step()


with tf.name_scope("discriminator_train"):
    quality_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("quality_discriminator")]
    quality_discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
    '''testing with grad descent'''
    # discrim_optim = tf.train.GradientDescentOptimizer(a.lr)
    quality_discrim_grads_and_vars = quality_discrim_optim.compute_gradients(quality_discrim_loss, var_list=quality_discrim_tvars)
    quality_discrim_train = quality_discrim_optim.apply_gradients(quality_discrim_grads_and_vars)
    
    type_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("type_discriminator")]
    type_discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
    '''testing with grad descent'''
    # discrim_optim = tf.train.GradientDescentOptimizer(a.lr)
    type_discrim_grads_and_vars = type_discrim_optim.compute_gradients(type_discrim_loss, var_list=type_discrim_tvars)
    type_discrim_train = type_discrim_optim.apply_gradients(type_discrim_grads_and_vars)



with tf.name_scope("generator_train"):
    with tf.control_dependencies([type_discrim_train]):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars, global_step=global_step)



sess = tf.Session()
sess.run(tf.global_variables_initializer())



def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def save_images(src,spectrum,batch_number):
    imsave(np.concatenate((src,spectrum),axis=0), [1+number_of_types,1],'train/{}/{:06d}.png'.format(a.model_name, batch_number))

os.makedirs('train/'+a.model_name, exist_ok=True)

writer = tf.summary.FileWriter("./logs/"+a.model_name, sess.graph)
saver = tf.train.Saver()

if continue_training:
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, checkpoint)
    print('reloaded')


save_freq = 100
update_freq=10

usable_types = ['Normal', 'Water', 'Grass', 'Fire', 'Electric', 'Psychic']
random_data = {x:np.random.random([256,256,3]) for x in usable_types}

def prepare_data(data):
    images = []
    types = list(np.diag(np.ones(number_of_types)))
    for i,type in enumerate(usable_types):
        images.append(data[type])
    return images, types

gs=0
for i in range(50001):
    if gs < 5000: an = 0.0
    else: an = min((gs-5000)/8000,1); print(an)
    ii = []
    it = []
    for b in range(BS):
        im,ty = prepare_data(next(gen))
        ii+=im
        it+=ty
    
    rt = []
    for t in it:
        rand = np.random.random([number_of_types]) - t
        out = np.zeros(number_of_types)
        out[rand.argmax()] = 1
        rt.append(out)
    # print(it)
    # print(rt)
    fd = {input_types:it, input_images: normalize(ii), random_type:rt, type_annealing:an}
    
    
    fetches = {'g':gen_train, 'td': type_discrim_train, 'qd': quality_discrim_train, 'gs':global_step}
    if i % update_freq == 0:
        fetches['summary_str'] = summary_op
    if i % save_freq == 0:
        fetches['spectrum'] = fake_spectrum
    results = sess.run(fetches,fd)
    gs = results['gs']
    
    if i % update_freq == 0:
        writer.add_summary(results['summary_str'], gs)
        print(gs)
    if i % save_freq == 0:
        ind = np.random.randint(6)
        spectrum = unnormalize(results['spectrum'][ind])
        src = np.expand_dims(ii[ind],0)
        save_images(src,spectrum,gs)
        saver.save(sess, os.path.join(checkpoint_dir, a.model_name),global_step=gs)
    
    
