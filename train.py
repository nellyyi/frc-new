# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import dnnlib.tflib.tfutil as tfutil
import dnnlib.util as util

import config

from util import save_image, save_snapshot
from validation import ValidationSet
from dataset import create_dataset


from math import floor
import numpy as np
import imageio
import scipy.integrate


from tensorflow.python.ops import math_ops


def radial_mask(r, cx=128, cy=128,sx=np.arange(0, 256),sy=np.arange(0, 256), delta=1):
    
    ind = (sx[np.newaxis,:]-cx)**2 + (sy[:,np.newaxis]-cy)**2
    ind1 = ind <= ((r[0] + delta)**2) # one liner for this and below?
    ind2 = ind > (r[0]**2)
    return ind1*ind2

@tf.function
def trapezoidal_integral_approx(t, y):
    return math_ops.reduce_sum(
            math_ops.multiply(t[:-1] - t[1:],
                              (y[:-1] + y[1:]) / 2.),
            name='trapezoidal_integral_approx')


@tf.function
def fourier_ring_correlation(image1, image2, rn, spatial_freq):
        image1 = tf.compat.v1.cast(image1, tf.complex64)
        image2 = tf.compat.v1.cast(image2, tf.complex64)
       
        fft_image1 = tf.signal.fftshift(tf.signal.fft2d(image1), axes=[2,3])
        fft_image2 = tf.signal.fftshift(tf.signal.fft2d(image2), axes=[2,3])
       
        #fft_image1 = image1 #FT CODE
        #fft_image2 = image2 #FT CODE

        t1 = tf.multiply(fft_image1, rn)
        t2 = tf.multiply(fft_image2, rn)
        print("T1", t1.get_shape()) # Shoudl be (43, ?, 3, 256, 256)

        c1 = tf.math.real(tf.reduce_sum(tf.multiply(t1 , tf.math.conj(t2)), [1,2,3,4]))
        c2 = tf.reduce_sum(tf.math.abs(t1) ** 2, [1,2,3,4])
        c3 = tf.reduce_sum(tf.math.abs(t2) ** 2, [1,2,3,4])
     
        print("c", c1.get_shape(), c2.get_shape(), c3.get_shape())
        # Shape should be (43,)
        
        frc = tf.math.divide(tf.math.abs(c1) , tf.math.sqrt(tf.math.multiply(c2 , c3)))
        frc = tf.where(tf.compat.v1.is_inf(frc), tf.zeros_like(frc), frc) # inf
        frc = tf.where(tf.compat.v1.is_nan(frc), tf.zeros_like(frc), frc) # nan
        
        frc = tf.reshape(frc, [ frc.get_shape()[0],1])

        t = spatial_freq
        y = frc
        riemann_sum = tf.reduce_sum(tf.multiply(t[:-1] - t[1:], (y[:-1] + y[1:]) / 2.))
        print("RS", riemann_sum)
        #riemann_sum = trapezoidal_integral_approx(spatial_freq / max(spatial_freq), frc)
        return riemann_sum



class AugmentGaussian:
    def __init__(self, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range

    def add_train_noise_tf(self, x): # this is where the noise is added
        (minval,maxval) = self.train_stddev_range
        shape = tf.shape(x)
        #print("DEBUG add_train_noise_tf:", minval/255, maxval/255)
        rng_stddev = tf.compat.v1.random_uniform(shape=[1, 1, 1], minval=minval/255.0, maxval=maxval/255.0)
        return x + tf.compat.v1.random_normal(shape) * rng_stddev
    
    def add_validation_noise_np(self, x): # need to make sure this is also added !!! 
        return x + np.random.normal(size=x.shape)*(self.validation_stddev/255.0)    

class AugmentPoisson:
    def __init__(self, lam_max):
        self.lam_max = lam_max

    def add_train_noise_tf(self, x):
        chi_rng = tf.compat.v1.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=self.lam_max)
        return tf.compat.v1.random_poisson(chi_rng*(x+0.5), shape=[])/chi_rng - 0.5

    def add_validation_noise_np(self, x):
        chi = 30.0
        return np.random.poisson(chi*(x+0.5))/chi - 0.5

def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5+np.cos(t * np.pi)/2)**2
        return learning_rate * smooth
    return learning_rate

def train(
    submit_config: dnnlib.SubmitConfig,
    iteration_count: int,
    eval_interval: int,
    minibatch_size: int,
    learning_rate: float,
    ramp_down_perc: float,
    noise: dict,
    validation_config: dict,
    train_tfrecords: str,
    noise2noise: bool):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**validation_config)

    # Create a run context (hides low level details, exposes simple API to manage the run)
    ctx = dnnlib.RunContext(submit_config, config)

    # Initialize TensorFlow graph and session using good default settings
    tfutil.init_tf(config.tf_config)

    dataset_iter = create_dataset(train_tfrecords, minibatch_size, noise_augmenter.add_train_noise_tf)
    # Construct the network using the Network helper class and a function defined in config.net_config
    with tf.device("/gpu:0"):
        net = tflib.Network(**config.net_config)

    # Optionally print layer information
    net.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device("/cpu:0"):
        lrate_in        = tf.compat.v1.placeholder(tf.float32, name='lrate_in', shape=[])

        #print("DEBUG train:", "dataset iter got called")
        noisy_input, noisy_target, clean_target = dataset_iter.get_next()
        noisy_input_split = tf.split(noisy_input, submit_config.num_gpus)
        noisy_target_split = tf.split(noisy_target, submit_config.num_gpus)
        print(len(noisy_input_split), noisy_input_split)
        clean_target_split = tf.split(clean_target, submit_config.num_gpus)
        # Split [?, 3, 256, 256] across num_gpus over axis 0 (i.e. the batch) 

    # Define the loss function using the Optimizer helper class, this will take care of multi GPU
    opt = tflib.Optimizer(learning_rate=lrate_in, **config.optimizer_config)
    radii = np.arange(128).reshape(128,1) #image size 256, binning = 3
    radial_masks = np.apply_along_axis(radial_mask, 1, radii, 128, 128, np.arange(0, 256), np.arange(0, 256), 20)
    print("RN SHAPE!!!!!!!!!!:", radial_masks.shape)
    radial_masks = np.expand_dims(radial_masks, 1) # (128, 1, 256, 256)
    #radial_masks = np.squeeze(np.stack((radial_masks,) * 3, -1)) # 43, 3, 256, 256
    #radial_masks = radial_masks.transpose([0, 3, 1, 2])
    radial_masks = radial_masks.astype(np.complex64)
    radial_masks = tf.expand_dims(radial_masks, 1)
    
    rn          = tf.compat.v1.placeholder_with_default(radial_masks, [128, None, 1, 256, 256])
    rn_split    = tf.split(rn, submit_config.num_gpus, axis=1)
    freq_nyq = int(np.floor(int(256) / 2.0))

    spatial_freq = radii.astype(np.float32) / freq_nyq 
    spatial_freq = spatial_freq / max(spatial_freq)
    
    for gpu in range(submit_config.num_gpus):
        with tf.device("/gpu:%d" % gpu):
            net_gpu = net if gpu == 0 else net.clone()

            denoised_1 = net_gpu.get_output_for(noisy_input_split[gpu])
            denoised_2 = net_gpu.get_output_for(noisy_target_split[gpu])
            print(noisy_input_split[gpu].get_shape(), rn_split[gpu].get_shape())
            if noise2noise:
                meansq_error  = fourier_ring_correlation(noisy_target_split[gpu], denoised_1, rn_split[gpu], spatial_freq) - fourier_ring_correlation(noisy_target_split[gpu] - denoised_2, noisy_input_split[gpu] - denoised_1,rn_split[gpu], spatial_freq)
            else:
                meansq_error = tf.reduce_mean(tf.square(clean_target_split[gpu] - denoised))
            # Create an autosummary that will average over all GPUs
            #tf.summary.histogram(name, var)
            with tf.control_dependencies([autosummary("Loss", meansq_error)]):
                opt.register_gradients(meansq_error, net_gpu.trainables)

    train_step = opt.apply_updates()

    # Create a log file for Tensorboard
    summary_log = tf.compat.v1.summary.FileWriter(submit_config.run_dir)
    summary_log.add_graph(tf.compat.v1.get_default_graph())

    print('Training...')
    time_maintenance = ctx.get_time_since_last_update()
    ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=0, max_epoch=iteration_count)

    # The actual training loop
    for i in range(iteration_count):
        # Whether to stop the training or not should be asked from the context
        if ctx.should_stop():
            break
        # Dump training status
        if i % eval_interval == 0:

            time_train = ctx.get_time_since_last_update()
            time_total = ctx.get_time_since_start()
            print("DEBUG TRAIN!", noisy_input.dtype, noisy_input[0][0].dtype)
            # Evaluate 'x' to draw a batch of inputs
            [source_mb, target_mb] = tfutil.run([noisy_input, clean_target])
            denoised = net.run(source_mb)
            save_image(submit_config, denoised[0], "img_{0}_y_pred.tif".format(i))
            save_image(submit_config, target_mb[0], "img_{0}_y.tif".format(i))
            save_image(submit_config, source_mb[0], "img_{0}_x_aug.tif".format(i))

            validation_set.evaluate(net, i, noise_augmenter.add_validation_noise_np)

            print('iter %-10d time %-12s sec/eval %-7.1f sec/iter %-7.2f maintenance %-6.1f' % (
                autosummary('Timing/iter', i),
                dnnlib.util.format_time(autosummary('Timing/total_sec', time_total)),
                autosummary('Timing/sec_per_eval', time_train),
                autosummary('Timing/sec_per_iter', time_train / eval_interval),
                autosummary('Timing/maintenance_sec', time_maintenance)))

            dnnlib.tflib.autosummary.save_summaries(summary_log, i)
            ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=i, max_epoch=iteration_count)
            time_maintenance = ctx.get_last_update_interval() - time_train

            save_snapshot(submit_config, net, str(i))
        lrate =  compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate)
        tfutil.run([train_step], {lrate_in: lrate})

    print("Elapsed time: {0}".format(util.format_time(ctx.get_time_since_start())))
    save_snapshot(submit_config, net, 'final')

    # Summary log and context should be closed at the end
    summary_log.close()
    ctx.close()
