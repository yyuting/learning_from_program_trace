

from __future__ import division

import gpu_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_util.pick_gpu_lowest_memory())

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
plt = pyplot

import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy
import numpy.random
import random
import argparse_util
import pickle
from tensorflow.python.client import timeline
import copy
import sys; sys.path += ['apps']
import importlib
import importlib.util
import subprocess
import shutil
from tf_util import *
import json
import glob

import warnings
import skimage
import skimage.io
import skimage.transform
import scipy.ndimage

import copy
import gc

allowed_dtypes = ['float64', 'float32', 'uint8']
no_L1_reg_other_layers = True

width = 500
height = 400

allow_nonzero = False

identity_output_layer = True

less_aggresive_ini = False

conv_padding = "SAME"
padding_offset = 32

analyze_per_ch = 2

shaders_pool = [
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('mandelbulb', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})
    ],
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('mandelbulb_slim', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})
    ],
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True, 'specified_ind': 'subsample_taylor_exp_vals.npy2.npy'}), 
        ('mandelbulb_slim', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True, 'specified_ind': 'subsample_taylor_exp_vals.npy4.npy'}), 
        ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})
    ],
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot_with_bg', {'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('mandelbulb_slim', 'none', 'datas_mandelbulb_with_bg', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_new_extrapolation_subsample_2', {'every_nth': 2, 'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('primitives_wheel_only', 'none', 'datas_primitives_correct_test_range', {'fov': 'small'})
    ],
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot_with_bg', {'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True, 'specified_ind': 'subsample_taylor_exp_vals.npy2.npy'}), 
        ('mandelbulb_slim', 'none', 'datas_mandelbulb_with_bg', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_new_extrapolation_subsample_2', {'every_nth': 2, 'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True, 'specified_ind': 'subsample_taylor_exp_vals.npy4.npy'}), 
        ('primitives_wheel_only', 'none', 'datas_primitives_correct_test_range', {'fov': 'small'})
    ]
]

all_shaders = shaders_pool[0]

shaders_aux_pool = [
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small'}), 
        ('mandelbulb', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small'}), 
        ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})
    ],
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small'}), 
        ('mandelbulb_slim', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small'}), 
        ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})
    ],
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot_with_bg', {'fov': 'small'}), 
        ('mandelbulb_slim', 'none', 'datas_mandelbulb_with_bg', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_new_extrapolation_subsample_2', {'every_nth': 2, 'fov': 'small'}), 
        ('primitives_wheel_only', 'none', 'datas_primitives_correct_test_range', {'fov': 'small'})
    ]
]

all_shaders_aux = shaders_aux_pool[0]

def smart_mkdir(dir):
    if os.path.isdir(dir):
        print('dir already exists', dir)
        return
    
    parent, _ = os.path.split(dir)
    parent_stack = []
    
    while not os.path.isdir(parent):
        parent_stack.append(parent)
        parent, _ = os.path.split(parent)
        
    for pa in parent_stack[::-1]:
        os.mkdir(pa)
   
    os.mkdir(dir)


def get_tensors(dataroot, name, camera_pos, shader_time, output_type='remove_constant', nsamples=1, shader_name='zigzag', geometry='plane', color_inds=[], manual_features_only=False, h_start=0, h_offset=height, w_start=0, w_offset=width, samples=None, fov='regular', t_sigma=1/60.0, first_last_only=False, last_only=False, subsample_loops=-1, last_n=-1, first_n=-1, first_n_no_last=-1, mean_var_only=False, zero_samples=False, render_fix_spatial_sample=False, render_zero_spatial_sample=False, spatial_samples=None, every_nth=-1, every_nth_stratified=False, additional_features=True, ignore_last_n_scale=0, include_noise_feature=False, no_noise_feature=False, relax_clipping=False, render_sigma=None, same_sample_all_pix=False, automatic_subsample=False, automate_raymarching_def=False, log_only_return_def_raymarching=True, debug=[], SELECT_FEATURE_THRE=200, compiler_problem_idx=-1, feature_normalize_lo_pct=20, specified_ind=None, write_file=True, alt_dir=''):

    manual_features_only = manual_features_only

    if output_type not in ['rgb', 'bgr']:
            
        hi_pct = 100 - feature_normalize_lo_pct
        feature_scale_file = os.path.join(dataroot, 'feature_scale_%d_%d.npy' % (feature_normalize_lo_pct, hi_pct))
        feature_bias_file = os.path.join(dataroot, 'feature_bias_%d_%d.npy' % (feature_normalize_lo_pct, hi_pct))

        if os.path.exists(feature_scale_file) and os.path.exists(feature_bias_file):
            feature_scale = np.load(feature_scale_file)
            feature_bias = np.load(feature_bias_file)
        else:
            print('-----------------------------------------------')
            print('WARNING: featue_scale and feature_bias with corresponding lo/hi pct label not found, using default files instead')
            feature_scale = np.load(os.path.join(dataroot, 'feature_scale.npy'))
            feature_bias = np.load(os.path.join(dataroot, 'feature_bias.npy'))


        tolerance = 2.0

    if compiler_problem_idx < 0:
        compiler_problem_full_name = os.path.abspath(os.path.join(name, 'compiler_problem.py'))
    else:
        compiler_problem_full_name = os.path.abspath(os.path.join(name, 'compiler_problem%d.py' % compiler_problem_idx))
    if not os.path.exists(compiler_problem_full_name):
        print('Error! No compiler_problem.py found in target directory: %s' % name)
        print('Because the compiler source code needs significant clean-up, it is not released yet.')
        print('The current released pipeline is therefore unable to translate a DSL program to TF program.')
        print('Please specify the target directory to be the ones containing trained models, or copy compiler_problem.py in those directories to the new target directory')
        raise
        
    spec = importlib.util.spec_from_file_location("module.name", compiler_problem_full_name)
    compiler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler_module)
        
    feature_pl = []

    features, vec_output, manual_features = get_render(camera_pos, shader_time, nsamples=nsamples, shader_name=shader_name, geometry=geometry, return_vec_output=True, compiler_module=compiler_module, manual_features_only=manual_features_only, h_start=h_start, h_offset=h_offset, w_start=w_start, w_offset=w_offset, samples=samples, fov=fov, t_sigma=t_sigma, zero_samples=zero_samples, render_fix_spatial_sample=render_fix_spatial_sample, render_zero_spatial_sample=render_zero_spatial_sample, spatial_samples=spatial_samples, additional_features=additional_features, include_noise_feature=include_noise_feature, no_noise_feature=no_noise_feature, render_sigma=render_sigma, same_sample_all_pix=same_sample_all_pix, debug=debug)
    
    
    # workaround if for some feature sparsification setup, RGB channels are not logged
    # also prevent aux feature from not being logged
    features = features + vec_output + manual_features
        

    color_features = vec_output
    valid_features = []
    with tf.control_dependencies(color_features):
        with tf.variable_scope("auxiliary"):
            valid_inds = []
            feature_ind = 0
            if output_type in ['rgb', 'bgr']:
                valid_features = vec_output
                valid_inds = [0, 1, 2]
            else:
                for i in range(len(features)):
                    if isinstance(features[i], (float, int, numpy.bool_)):
                        continue
                    else:
                        if features[i] in valid_features:
                            continue
                    feature_ind += 1
                    valid_inds.append(i)
                    valid_features.append(features[i])

            for i in range(len(valid_features)):
                if valid_features[i].dtype != dtype:
                    valid_features[i] = tf.cast(valid_features[i], dtype)
            if write_file:
                numpy.save('%s/valid_inds.npy' % name, valid_inds)
            elif alt_dir is not None:
                numpy.save('%s/valid_inds.npy' % alt_dir, valid_inds)

            if manual_features_only:
                manual_inds = []
                manual_features_valid = []
                additional_bias = []
                additional_scale = []
                for k in range(len(manual_features)):
                    feature = manual_features[k]
                    if not isinstance(feature, (float, int)):
                        try:
                            raw_ind = valid_features.index(feature)
                            manual_inds.append(raw_ind)
                            manual_features_valid.append(feature)
                        except:
                            raise
                out_features = manual_features_valid

                feature_bias = feature_bias[manual_inds]
                feature_scale = feature_scale[manual_inds]

            else:
                out_features = valid_features
                
            

            if ignore_last_n_scale > 0 and output_type not in ['rgb', 'bgr']:
                new_inds = numpy.arange(feature_bias.shape[0] - ignore_last_n_scale)
                if include_noise_feature:
                    new_inds = numpy.concatenate((new_inds, [feature_bias.shape[0]-2, feature_bias.shape[0]-1]))
                feature_bias = feature_bias[new_inds]
                feature_scale = feature_scale[new_inds]


            if specified_ind is not None:
                specified_ind_vals = specified_ind
                
                new_features = []
                for ind in specified_ind_vals:
                    new_features.append(out_features[ind])
                out_features = new_features
                
                feature_bias = feature_bias[specified_ind_vals]
                feature_scale = feature_scale[specified_ind_vals]
            
            for vec in vec_output:
                actual_ind = out_features.index(vec)
                color_inds.append(actual_ind)

            if output_type not in ['rgb', 'bgr']:
                for ind in color_inds:
                    feature_bias[ind] = 0.0
                    feature_scale[ind] = 1.0
            
            if len(feature_pl) > 0:
                for var in feature_pl:
                    if var in out_features:
                        idx = out_features.index(var)

            
            if output_type == 'remove_constant':
                features = tf.parallel_stack(out_features)

                features = tf.transpose(features, [1, 2, 3, 0])


            elif output_type == 'all':
                features = tf.cast(tf.stack(features, axis=-1), tf.float32)
            elif output_type in ['rgb', 'bgr']:
                features = tf.cast(tf.stack(vec_output, axis=-1), tf.float32)
                if output_type == 'bgr':
                    features = features[..., ::-1]
            else:
                raise
            

            if (output_type not in ['rgb', 'bgr']):
                features += feature_bias
                features *= feature_scale

                # sanity check for manual features
                if manual_features_only and False:
                    manual_inds = []
                    manual_features_valid = []
                    for feature in manual_features:
                        if not isinstance(feature, (float, int)):
                            raw_ind = valid_features.index(feature)
                            manual_inds.append(raw_ind)
                            manual_features_valid.append(feature)
                    manual_features_valid = tf.parallel_stack(manual_features_valid)
                    manual_features_valid = tf.transpose(manual_features_valid, [1, 2, 3, 0])
                    manual_features_valid += feature_bias[manual_inds]
                    manual_features_valid *= feature_scale[manual_inds]

                    camera_pos_val = numpy.load(os.path.join(dataroot, 'train.npy'))
                    feed_dict = {camera_pos: camera_pos_val[0], shader_time:[0]}
                    sess = tf.Session()
                    manual_features_val, features_val = sess.run([manual_features_valid, features], feed_dict=feed_dict)
                    for k in range(len(manual_inds)):
                        print(numpy.max(numpy.abs(manual_features_val[:, :, :, k] - features_val[:, :, :, manual_inds[k]])))

            elif output_type in ['rgb', 'bgr']: # workaround because clip_by_value will make all nans to be the higher value
                features = tf.where(tf.is_nan(features), tf.zeros_like(features), features)

            if (not relax_clipping) or (output_type in ['rgb', 'bgr']):
                features_clipped = tf.clip_by_value(features, 0.0, 1.0)
                features = features_clipped
            else:
                features -= 0.5
                features *= 2
                features = tf.clip_by_value(features, -2.0, 2.0)

            features = tf.where(tf.is_nan(features), tf.zeros_like(features), features)

    
    return features

    #numpy.save('valid_inds.npy', valid_inds)
    #return features

def get_render(camera_pos, shader_time, samples=None, nsamples=1, shader_name='zigzag', color_inds=None, return_vec_output=False, render_size=None, render_sigma=None, compiler_module=None, geometry='plane', zero_samples=False, debug=[], extra_args=[None], render_g=False, manual_features_only=False, fov='regular', h_start=0, h_offset=height, w_start=0, w_offset=width, t_sigma=1/60.0, render_fix_spatial_sample=False, render_zero_spatial_sample=False, spatial_samples=None, additional_features=True, include_noise_feature=False, no_noise_feature=False, same_sample_all_pix=False):

    assert compiler_module is not None

    if additional_features:
        if geometry not in ['none']:
            features_len_add = 7
        else:
            features_len_add = 2
        if no_noise_feature:
            features_len_add -= 2
    else:
        if include_noise_feature:
            features_len_add = 2
        else:
            features_len_add = 0

    features_len = compiler_module.f_log_intermediate_len + features_len_add

    vec_output_len = compiler_module.vec_output_len

    manual_features_len = compiler_module.f_log_intermediate_subset_len
    manual_depth_offset = 0
    if geometry not in ['none']:
        manual_features_len += 1
        manual_depth_offset = 1

    f_log_intermediate_subset = [None] * manual_features_len

        
    if render_size is not None:
        global width
        global height
        width = render_size[0]
        height = render_size[1]
        
    f_log_intermediate = [None] * features_len
    vec_output = [None] * vec_output_len

    xv, yv = tf.meshgrid(tf.range(w_offset, dtype=dtype), tf.range(h_offset, dtype=dtype), indexing='ij')
    xv = tf.transpose(xv)
    yv = tf.transpose(yv)
    xv = tf.expand_dims(xv, 0)
    yv = tf.expand_dims(yv, 0)
    xv = tf.tile(xv, [nsamples, 1, 1])
    yv = tf.tile(yv, [nsamples, 1, 1])
    xv_orig = xv
    yv_orig = yv
    xv += tf.expand_dims(tf.expand_dims(w_start, axis=1), axis=2)
    yv += tf.expand_dims(tf.expand_dims(h_start, axis=1), axis=2)
    tensor_x0 = xv
    tensor_x1 = yv
    tensor_x2 = tf.expand_dims(tf.expand_dims(shader_time, axis=1), axis=2) * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)

    if samples is None:
        if not same_sample_all_pix:
            sample1 = tf.random_normal(tf.shape(xv), dtype=dtype)
            sample2 = tf.random_normal(tf.shape(xv), dtype=dtype)
            sample3 = tf.random_normal(tf.shape(xv), dtype=dtype)
        else:
            sample1 = tf.fill(tf.shape(xv), tf.random_normal((), dtype=dtype))
            sample2 = tf.fill(tf.shape(xv), tf.random_normal((), dtype=dtype))
            sample3 = tf.fill(tf.shape(xv), tf.random_normal((), dtype=dtype))
    else:
        sample3 = 0.0
        # the assumption here is batch_size = 1
        if isinstance(samples[0], numpy.ndarray) and isinstance(samples[1], numpy.ndarray):
            sample1 = tf.constant(samples[0], dtype=dtype)
            sample2 = tf.constant(samples[1], dtype=dtype)
            if samples[0].shape[1] == height + padding_offset and samples[0].shape[2] == width + padding_offset:
                start_slice = [0, tf.cast(h_start[0], tf.int32) + padding_offset // 2, tf.cast(w_start[0], tf.int32) + padding_offset // 2]
                size_slice = [nsamples, int(h_offset), int(w_offset)]
                sample1 = tf.slice(sample1, start_slice, size_slice)
                sample2 = tf.slice(sample2, start_slice, size_slice)
            else:
                assert samples[0].shape[1] == h_offset and samples[1].shape[2] == w_offset
        else:
            assert isinstance(samples[0], tf.Tensor) and isinstance(samples[1], tf.Tensor)
            sample1 = samples[0]
            sample2 = samples[1]
            start_slice = [0, tf.cast(h_start[0], tf.int32) + padding_offset // 2, tf.cast(w_start[0], tf.int32) + padding_offset // 2]
            size_slice = [nsamples, int(h_offset), int(w_offset)]
            sample1 = tf.slice(sample1, start_slice, size_slice)
            sample2 = tf.slice(sample2, start_slice, size_slice)
        #if dtype == tf.float64:
        #    sample1 = samples[0].astype(np.float64)
        #    sample2 = samples[1].astype(np.float64)

    if render_sigma is None:
        render_sigma = [0.5, 0.5, t_sigma]
    print('render_sigma:', render_sigma)

    if not zero_samples:
        #print("using random samples")

        if (render_fix_spatial_sample or render_zero_spatial_sample) and spatial_samples is not None:
            #print("fix spatial samples")
            sample1 = tf.constant(spatial_samples[0], dtype=dtype)
            sample2 = tf.constant(spatial_samples[1], dtype=dtype)

        vector3 = [tensor_x0 + render_sigma[0] * sample1, tensor_x1 + render_sigma[1] * sample2, tensor_x2]

    else:
        vector3 = [tensor_x0, tensor_x1, tensor_x2]
        sample1 = tf.zeros_like(sample1)
        sample2 = tf.zeros_like(sample2)

    f_log_intermediate[0] = shader_time
    f_log_intermediate[1] = camera_pos

    get_shader(vector3, f_log_intermediate, f_log_intermediate_subset, camera_pos, features_len, manual_features_len, shader_name=shader_name, color_inds=color_inds, vec_output=vec_output, compiler_module=compiler_module, geometry=geometry, debug=debug, extra_args=extra_args, render_g=render_g, manual_features_only=manual_features_only, fov=fov, features_len_add=features_len_add, manual_depth_offset=manual_depth_offset, additional_features=additional_features)


    if (not no_noise_feature):
        if (additional_features or include_noise_feature):
            f_log_intermediate[features_len-2] = sample1
            f_log_intermediate[features_len-1] = sample2

    if return_vec_output:
        return f_log_intermediate, vec_output, f_log_intermediate_subset
    else:
        return f_log_intermediate

def get_shader(x, f_log_intermediate, f_log_intermediate_subset, camera_pos, features_len, manual_features_len, shader_name='zigzag', color_inds=None, vec_output=None, compiler_module=None, geometry='plane', debug=[], extra_args=[None], render_g=False, manual_features_only=False, fov='regular', features_len_add=7, manual_depth_offset=1, additional_features=True):
    assert compiler_module is not None
    features_dt = []
    
    input_pl_to_features = []
    
    features = get_features(x, camera_pos, geometry=geometry, debug=debug, extra_args=extra_args, fov=fov, features_dt=features_dt)
    
    if vec_output is None:
        vec_output = [None] * 3

    # adding depth
    if geometry == 'plane':
        f_log_intermediate_subset[-1] = features[7]
    elif geometry in ['hyperboloid1', 'paraboloid']:
        f_log_intermediate_subset[-1] = extra_args[0]
    elif geometry not in ['none']:
        raise

    with tf.variable_scope("auxiliary"):

        if geometry not in ['none'] and additional_features:
            if not render_g:
                h = 1e-4
            else:
                h = 1e-8
            if geometry == 'plane':
                u_ind = 1
                v_ind = 2
            elif geometry in ['hyperboloid1', 'sphere', 'paraboloid']:
                u_ind = 8
                v_ind = 9
            else:
                raise

            new_x = x[:]
            new_x[0] = x[0] - h
            features_neg_x = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
            new_x[0] = x[0] + h
            features_pos_x = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
            f_log_intermediate[features_len-features_len_add] = (features_pos_x[u_ind] - features_neg_x[u_ind]) / (2 * h)
            f_log_intermediate[features_len-features_len_add+1] = (features_pos_x[v_ind] - features_neg_x[v_ind]) / (2 * h)

            new_x = x[:]
            new_x[1] = x[1] - h
            features_neg_y = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
            new_x[1] = x[1] + h
            features_pos_y = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
            f_log_intermediate[features_len-features_len_add+2] = (features_pos_y[u_ind] - features_neg_y[u_ind]) / (2 * h)
            f_log_intermediate[features_len-features_len_add+3] = (features_pos_y[v_ind] - features_neg_y[v_ind]) / (2 * h)

            f_log_intermediate[features_len-features_len_add+4] = f_log_intermediate[features_len-features_len_add] * f_log_intermediate[features_len-features_len_add+3] - f_log_intermediate[features_len-features_len_add+1] * f_log_intermediate[features_len-features_len_add+2]

            
    if len(debug) > 0:
        vec_output[0] = debug[0]
    if not render_g:
        compiler_module.f(features, f_log_intermediate, vec_output, f_log_intermediate_subset)
    else:
        assert geometry not in ['none']
        sigma = [None] * len(features)
        for i in range(len(features)):
            variance = (tf.square((features_pos_x[i] - features[i]) / h) * 0.25 + tf.square(numpy.array(features_pos_y[i] - features[i]) / h) * 0.25)
            sigma[i] = tf.sqrt(variance)
            sigma[i] = tf.where(tf.is_nan(sigma[i]), tf.zeros_like(sigma[i]), sigma[i])
        # workaround for bug in shader compiler
        if geometry == 'plane':
            sigma[7] = 0.0
        global dtype
        dtype = tf.float32
        for i in range(len(features)):
            if isinstance(features[i], tf.Tensor):
                features[i] = tf.cast(features[i], dtype)
            if isinstance(sigma[i], tf.Tensor):
                sigma[i] = tf.cast(sigma[i], dtype)

        compiler_module.g(features, vec_output, sigma)

    return

def get_features(x, camera_pos, geometry='plane', debug=[], extra_args=[None], fov='regular', features_dt=[]):
    
    if fov.startswith('regular'):
        ray_dir = [x[0] - width / 2, x[1] + 1, width / 2]
    elif fov.startswith('small'):
        ray_dir = [x[0] - width / 2, x[1] - height / 2, 1.73 * width / 2]
        #print("use small fov (60 degrees horizontally)")
    else:
        raise
        
    ray_origin = [camera_pos[0], camera_pos[1], camera_pos[2]]
    ang1 = camera_pos[3]
    ang2 = camera_pos[4]
    ang3 = camera_pos[5]


    for i in range(len(ray_origin)):
        ray_origin[i] = tf.expand_dims(tf.expand_dims(ray_origin[i], axis=1), axis=2)
    ang1 = tf.expand_dims(tf.expand_dims(ang1, axis=1), axis=2)
    ang2 = tf.expand_dims(tf.expand_dims(ang2, axis=1), axis=2)
    ang3 = tf.expand_dims(tf.expand_dims(ang3, axis=1), axis=2)

    ray_dir_norm = tf.sqrt(ray_dir[0] **2 + ray_dir[1] ** 2 + ray_dir[2] ** 2)
    ray_dir[0] /= ray_dir_norm
    ray_dir[1] /= ray_dir_norm
    ray_dir[2] /= ray_dir_norm

    sin1 = tf.sin(ang1)
    cos1 = tf.cos(ang1)
    sin2 = tf.sin(ang2)
    cos2 = tf.cos(ang2)
    sin3 = tf.sin(ang3)
    cos3 = tf.cos(ang3)

    if 'seperable' in fov:
        ray_dir_p = [(sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir[0] + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir[1] + cos2 * cos3 * ray_dir[2],
                     (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir[1] + cos2 * sin3 * ray_dir[2],
                     cos1 * cos2 * ray_dir[0] + sin1 * cos2 * ray_dir[1] + -sin2 * ray_dir[2]]
    else:
        ray_dir_p = [cos2 * cos3 * ray_dir[0] + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir[1] + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir[2],
                     cos2 * sin3 * ray_dir[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir[1] + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir[2],
                     -sin2 * ray_dir[0] + sin1 * cos2 * ray_dir[1] + cos1 * cos2 * ray_dir[2]]

    N = [0, 0, 1.0]

    if geometry == 'plane':
        features = [None] * 8
        
        t_ray = -ray_origin[2] / (ray_dir_p[2])
        features[0] = x[2]

        features[1] = ray_origin[0] + t_ray * ray_dir_p[0]
        features[2] = ray_origin[1] + t_ray * ray_dir_p[1]
        features[3] = ray_origin[2] + t_ray * ray_dir_p[2]
        features[4] = -ray_dir_p[0]
        features[5] = -ray_dir_p[1]
        features[6] = -ray_dir_p[2]
        features[7] = t_ray
    elif geometry == 'none':
        features = [None] * 7
        features[0] = x[2]
        features[1] = ray_dir_p[0]
        features[2] = ray_dir_p[1]
        features[3] = ray_dir_p[2]
        features[4] = ray_origin[0] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[5] = ray_origin[1] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[6] = ray_origin[2] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)

    return features

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer(in_channels=[], allow_map_to_less=False, ndims=2):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if not allow_nonzero:
            #print('initializing all zero')
            array = np.zeros(shape, dtype=float)
        else:
            x = np.sqrt(6.0 / (shape[ndims] + shape[ndims+1])) / 1.5
            array = numpy.random.uniform(-x, x, size=shape)
            #print('initializing xavier')
            #return tf.constant(array, dtype=dtype)
        cx = shape[0] // 2
        if ndims > 1:
            cy = shape[1] // 2
        if len(in_channels) > 0:
            input_inds = in_channels
            output_inds = range(len(in_channels))
        elif allow_map_to_less:
            input_inds = range(min(shape[ndims], shape[ndims+1]))
            output_inds = input_inds
        else:
            input_inds = range(shape[ndims])
            output_inds = input_inds
        for i in range(len(input_inds)):
            if ndims == 2:
                if less_aggresive_ini:
                    array[cx, cy, input_inds[i], output_inds[i]] *= 10.0
                else:
                    array[cx, cy, input_inds[i], output_inds[i]] = 1.0
            elif ndims == 1:
                if less_aggresive_ini:
                    array[cx, input_inds[i], output_inds[i]] *= 10.0
                else:
                    array[cx, input_inds[i], output_inds[i]] = 1.0
        return tf.constant(array, dtype=dtype)
    return _initializer

nm = None

conv_channel = 24
actual_conv_channel = conv_channel

dilation_remove_large = False
dilation_clamp_large = False
dilation_remove_layer = False
dilation_threshold = 8

def build_vgg(input, output_nc=3, channels=[]):
    # 2 modifications
    # 1. use avg_pool instead of max_pool (for smooth gradient)
    # 2. use lrelu instead of relu (consistent with other models)
    net = input
    if len(channels) == 0:
        out_channels = [64, 128, 256, 512, 512]
    else:
        out_channels = channels
    nconvs = [2, 2, 3, 3, 3]
    for i in range(5):
        if i > 0:
            net = slim.avg_pool2d(net, 2, scope='pool_%d' % i)
        for j in range(nconvs[i]):
            net = slim.conv2d(net, out_channels[i], [3, 3], activation_fn=lrelu, scope='lrelu_%d_%d' % (i, j), padding=conv_padding)
    
    net = slim.conv2d(net, 1, [1, 1], activation_fn=lrelu, scope='out')
    return net
        
def build(input, ini_id=True, regularizer_scale=0.0, final_layer_channels=-1, identity_initialize=False, output_nc=3):
    regularizer = None
    if not no_L1_reg_other_layers and regularizer_scale > 0.0:
        regularizer = slim.l1_regularizer(regularizer_scale)
    if ini_id or identity_initialize:
        net=slim.conv2d(input,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(allow_map_to_less=True),scope='g_conv1',weights_regularizer=regularizer, padding=conv_padding)
    else:
        net=slim.conv2d(input,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,scope='g_conv1',weights_regularizer=regularizer, padding=conv_padding)

    dilation_schedule = [2, 4, 8, 16, 32, 64]
    for ind in range(len(dilation_schedule)):
        dilation_rate = dilation_schedule[ind]
        conv_ind = ind + 2
        if dilation_rate > dilation_threshold:
            if dilation_remove_large:
                dilation_rate = 1
            elif dilation_clamp_large:
                dilation_rate = dilation_threshold
            elif dilation_remove_layer:
                continue
        #print('rate is', dilation_rate)
        net=slim.conv2d(net,actual_conv_channel,[3,3],rate=dilation_rate,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv'+str(conv_ind),weights_regularizer=regularizer, padding=conv_padding)


    net=slim.conv2d(net,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9',weights_regularizer=regularizer, padding=conv_padding)
    if final_layer_channels > 0:
        if actual_conv_channel > final_layer_channels and (not identity_initialize):
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, scope='final_0', weights_regularizer=regularizer, padding=conv_padding)
            nlayers = [1, 2]
        else:
            nlayers = [0, 1, 2]
        for nlayer in nlayers:
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(allow_map_to_less=True), scope='final_'+str(nlayer),weights_regularizer=regularizer, padding=conv_padding)

    #print('identity last layer?', identity_initialize and identity_output_layer)
    net=slim.conv2d(net,output_nc,[1,1],rate=1,activation_fn=None,scope='g_conv_last',weights_regularizer=regularizer, weights_initializer=identity_initializer(allow_map_to_less=True) if (identity_initialize and identity_output_layer) else tf.contrib.layers.xavier_initializer(), padding=conv_padding)
    return net

def prepare_data_root(dataroot, additional_input=False):
    output_names=[]
    val_img_names=[]
    map_names = []
    val_map_names = []
    grad_names = []
    val_grad_names = []
    add_names = []
    val_add_names = []
    
    validate_img_names = []

    train_output_dir = os.path.join(dataroot, 'train_img')
    test_output_dir = os.path.join(dataroot, 'test_img')
    
    validate_output_dir = os.path.join(dataroot, 'validate_img')

    for file in sorted(os.listdir(train_output_dir)):
        output_names.append(os.path.join(train_output_dir, file))
    for file in sorted(os.listdir(test_output_dir)):
        val_img_names.append(os.path.join(test_output_dir, file))
        
    if os.path.isdir(validate_output_dir):
        for file in sorted(os.listdir(validate_output_dir)):
            validate_img_names.append(os.path.join(validate_output_dir, file))

    if additional_input:
        train_add_dir = os.path.join(dataroot, 'train_add')
        test_add_dir = os.path.join(dataroot, 'test_add')
        for file in sorted(os.listdir(train_add_dir)):
            add_names.append(os.path.join(train_add_dir, file))
        for file in sorted(os.listdir(test_add_dir)):
            val_add_names.append(os.path.join(test_add_dir, file))

    return output_names, val_img_names, map_names, val_map_names, grad_names, val_grad_names, add_names, val_add_names, validate_img_names

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def generate_parser():
    parser = argparse_util.ArgumentParser(description='FastImageProcessing')
    parser.add_argument('--name', dest='name', default='', help='name of task')
    parser.add_argument('--is_train', dest='is_train', action='store_true', help='state whether this is training or testing')
    parser.add_argument('--use_batch', dest='use_batch', action='store_true', help='whether to use batches in training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='size of batches')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of epochs to train, seperated by comma')
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true', help='debug mode')
    parser.add_argument('--no_preload', dest='preload', action='store_false', help='whether to preload data')
    parser.add_argument('--test_training', dest='test_training', action='store_true', help='use training data for testing purpose')
    parser.add_argument('--input_w', dest='input_w', type=int, default=960, help='supplemental information needed when using queue to read binary file')
    parser.add_argument('--input_h', dest='input_h', type=int, default=640, help='supplemental information needed when using queue to read binary file')
    parser.add_argument('--which_epoch', dest='which_epoch', type=int, default=0, help='decide which epoch to read the checkpoint')
    parser.add_argument('--add_initial_layers', dest='add_initial_layers', action='store_true', help='add initial conv layers without dilation')
    parser.add_argument('--initial_layer_channels', dest='initial_layer_channels', type=int, default=-1, help='number of channels in initial layers')
    parser.add_argument('--conv_channel_multiplier', dest='conv_channel_multiplier', type=int, default=1, help='multiplier for conv channel')
    parser.add_argument('--add_final_layers', dest='add_final_layers', action='store_true', help='add final conv layers without dilation')
    parser.add_argument('--final_layer_channels', dest='final_layer_channels', type=int, default=-1, help='number of channels in final layers')
    parser.add_argument('--dilation_remove_large', dest='dilation_remove_large', action='store_true', help='when specified, use ordinary conv layer instead of dilated conv layer with large dilation rate')
    parser.add_argument('--dilation_clamp_large', dest='dilation_clamp_large', action='store_true', help='when specified, clamp large dilation rate to a give threshold')
    parser.add_argument('--dilation_threshold', dest='dilation_threshold', type=int, default=8, help='threshold used to remove or clamp dilation')
    parser.add_argument('--dilation_remove_layer', dest='dilation_remove_layer', action='store_true', help='when specified, use less dilated conv layers')
    parser.add_argument('--conv_channel_no', dest='conv_channel_no', type=int, default=-1, help='directly specify number of channels for dilated conv layers')
    parser.add_argument('--mean_estimator', dest='mean_estimator', action='store_true', help='if true, use mean estimator instead of neural network')
    parser.add_argument('--estimator_samples', dest='estimator_samples', type=int, default=1, help='number of samples used in mean estimator')
    parser.add_argument('--accurate_timing', dest='accurate_timing', action='store_true', help='if true, do not calculate loss for more accurate timing')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='learning rate for adam optimizer')
    parser.add_argument('--identity_initialize', dest='identity_initialize', action='store_true', help='if specified, initialize weights such that output is 1 sample RGB')
    parser.add_argument('--nonzero_ini', dest='allow_nonzero', action='store_true', help='if specified, use xavier for all those supposed to be 0 entries in identity_initializer')
    parser.add_argument('--no_identity_output_layer', dest='identity_output_layer', action='store_false', help='if specified, do not use identity mapping for output layer')
    parser.add_argument('--less_aggresive_ini', dest='less_aggresive_ini', action='store_true', help='if specified, use a less aggresive way to initialize RGB weights (multiples of the original xavier weights)')
    parser.add_argument('--render_only', dest='render_only', action='store_true', help='if specified, render using given camera pos, does not calculate loss')
    parser.add_argument('--render_camera_pos', dest='render_camera_pos', default='camera_pos.npy', help='used to render result')
    parser.add_argument('--render_t', dest='render_t', default='render_t.npy', help='used to render output')
    parser.add_argument('--train_res', dest='train_res', action='store_true', help='if specified, out_img = in_noisy_img + out_network')
    parser.add_argument('--RGB_norm', dest='RGB_norm', type=int, default=2, help='specify which p-norm to use for RGB loss')
    parser.add_argument('--mean_estimator_memory_efficient', dest='mean_estimator_memory_efficient', action='store_true', help='if specified, use a memory efficient way to calculate mean estimator, but may not be accurate in time')
    parser.add_argument('--manual_features_only', dest='manual_features_only', action='store_true', help='if specified, use only manual features already specified in each shader program')
    parser.add_argument('--tiled_training', dest='tiled_training', action='store_true', help='if specified, use tiled training')
    parser.add_argument('--tiled_w', dest='tiled_w', type=int, default=240, help='default width for tiles if using tiled training')
    parser.add_argument('--tiled_h', dest='tiled_h', type=int, default=320, help='default height for tiles if using tiled training')
    parser.add_argument('--fov', dest='fov', default='regular', help='specified the camera field of view')
    parser.add_argument('--first_last_only', dest='first_last_only', action='store_true', help='if specified, log only 1st and last iteration, do not log mean and std')
    parser.add_argument('--last_only', dest='last_only', action='store_true', help='log last iteration only')
    parser.add_argument('--subsample_loops', dest='subsample_loops', type=int, default=-1, help='log every n iter')
    parser.add_argument('--last_n', dest='last_n', type=int, default=-1, help='log last n iterations')
    parser.add_argument('--first_n', dest='first_n', type=int, default=-1, help='log first n iterations and last one')
    parser.add_argument('--first_n_no_last', dest='first_n_no_last', type=int, default=-1, help='log first n iterations')
    parser.add_argument('--mean_var_only', dest='mean_var_only', action='store_true', help='if flagged, use only mean and variance as loop statistic')
    parser.add_argument('--render_fix_spatial_sample', dest='render_fix_spatial_sample', action='store_true', help='if specified, fix spatial sample at rendering')
    parser.add_argument('--render_zero_spatial_sample', dest='render_zero_spatial_sample', action='store_true', help='if specified, use zero spatial sample')
    parser.add_argument('--render_fov', dest='render_fov', default='', help='if specified, can overwrite fov at render time')
    parser.add_argument('--every_nth', dest='every_nth', type=int, default=-1, help='log every nth var')
    parser.add_argument('--every_nth_stratified', dest='every_nth_stratified', action='store_true', help='if specified, do stratified sampling for every nth traces')
    parser.add_argument('--aux_plus_manual_features', dest='aux_plus_manual_features', action='store_true', help='if specified, use RGB+aux+manual features')
    parser.add_argument('--no_additional_features', dest='additional_features', action='store_false', help='if specified, do not use additional features during training')
    parser.add_argument('--ignore_last_n_scale', dest='ignore_last_n_scale', type=int, default=0, help='if nonzero, ignore the last n entries of stored feature_bias and feature_scale')
    parser.add_argument('--include_noise_feature', dest='include_noise_feature', action='store_true', help='if specified, include noise as additional features during trianing')
    parser.add_argument('--no_noise_feature', dest='no_noise_feature', action='store_true', help='if specified, do not include noise as additional features during training, will override include_noise_feature')
    parser.add_argument('--relax_clipping', dest='relax_clipping', action='store_true', help='if specified relax the condition of clipping features from 0-1 to -2-2')
    parser.add_argument('--train_with_zero_samples', dest='train_with_zero_samples', action='store_true', help='if specified, only use center of pixel for training')
    parser.add_argument('--tile_only', dest='tile_only', action='store_true', help='if specified, render only tiles (part of an entire image) according to tile_start')
    parser.add_argument('--no_summary', dest='write_summary', action='store_false', help='if specified, do not write train result to summary')
    parser.add_argument('--lpips_loss', dest='lpips_loss', action='store_true', help='if specified, use perceptual loss from Richard Zhang paepr')
    parser.add_argument('--lpips_loss_scale', dest='lpips_loss_scale', type=float, default=1.0, help='specifies the scale of lpips loss')
    parser.add_argument('--no_l2_loss', dest='l2_loss', action='store_false', help='if specified, do not use l2 loss')
    parser.add_argument('--lpips_net', dest='lpips_net', default='alex', help='specifies which network to use for lpips loss')
    parser.add_argument('--render_sigma', dest='render_sigma', type=float, default=0.5, help='specifies the sigma used for rendering')
    parser.add_argument('--same_sample_all_pix', dest='same_sample_all_pix', action='store_true', help='if specified, generate scalar random noise for all pixels instead of a matrix')
    parser.add_argument('--analyze_channel', dest='analyze_channel', action='store_true', help='in this mode, analyze and visualize contribution of each channel')
    parser.add_argument('--bad_example_base_dir', dest='bad_example_base_dir', default='', help='base dir for bad examples(RGB+Aux) in analyze_channel mode')
    parser.add_argument('--analyze_current_only', dest='analyze_current_only', action='store_true', help='in this mode, analyze only g_current (mostly because at lower res to resolve OOM)')
    parser.add_argument('--additional_input', dest='additional_input', action='store_true', help='if true, find additional input features from train/test_add')
    parser.add_argument('--save_frequency', dest='save_frequency', type=int, default=100, help='specifies the frequency to save a checkpoint')
    parser.add_argument('--camera_pos_file', dest='camera_pos_file', default='', help='if specified, use for no_dataroot mode')
    parser.add_argument('--feature_size_only', dest='feature_size_only', action='store_true', help='if specified, do not further create neural network, return after collecting the feature size')
    parser.add_argument('--automatic_subsample', dest='automatic_subsample', action='store_true', help='if specified, automatically decide program subsample rate (and raymarching and function def)')
    parser.add_argument('--automate_raymarching_def', dest='automate_raymarching_def', action='store_true', help='if specified, automatically choose schedule for raymarching and function def (but not subsampling rate')
    parser.add_argument('--inference_seq_len', dest='inference_seq_len', type=int, default=8, help='sequence length for inference')
    parser.add_argument('--SELECT_FEATURE_THRE', dest='SELECT_FEATURE_THRE', type=int, default=200, help='when automatically decide subsample rate, this will decide the trace budget')
    parser.add_argument('--repeat_timing', dest='repeat_timing', type=int, default=1, help='if > 1, repeat inference multiple times to get stable timing')
    parser.add_argument('--compiler_problem_idx', dest='compiler_problem_idx', type=int, default=-1, help='if nonnegative, use this idx to find appropriate compiler problem')
    parser.add_argument('--render_no_video', dest='render_no_video', action='store_true', help='in this mode, render images only, do not generate video')
    parser.add_argument('--render_dirname', dest='render_dirname', default='render', help='directory used to store render result')
    parser.add_argument('--render_tile_start', dest='render_tile_start', default='', help='specifies the tile start for each rendering if render in test_training mode')
    parser.add_argument('--feature_reduction_ch', dest='feature_reduction_ch', type=int, default=-1, help='specifies dimensionality after feature reduction channel. By default it should be the same as following initial layer or dilation layers, but we might want to change the dimensionality larger for fair RGBx comparison')
    parser.add_argument('--collect_validate_loss', dest='collect_validate_loss', action='store_true', help='if true, collect validation loss (and training score) and write to tensorboard')
    parser.add_argument('--read_from_best_validation', dest='read_from_best_validation', action='store_true', help='if true, read from the best validation checkpoint')
    parser.add_argument('--feature_normalize_lo_pct', dest='feature_normalize_lo_pct', type=int, default=25, help='used to find feature_bias file')
    parser.add_argument('--specified_ind', dest='specified_ind', default='', help='if specified, using the specified ind to define a subset of the trace for learning')
    parser.add_argument('--test_output_dir', dest='test_output_dir', default='', help='if specified, write output to this directory instead')
    parser.add_argument('--no_overwrite_option_file', dest='overwrite_option_file', action='store_false', help='if specified, do not overwrite option file even if the old one is outdated')
    parser.add_argument('--dataroot_parent', dest='dataroot_parent', default='', help='specifies the parent directory for all dataroot dirs')
    parser.add_argument('--epoch_per_shader', dest='epoch_per_shader', type=int, default=1, help='number of epochs run per shader')
    parser.add_argument('--multiple_feature_reduction_ch', dest='multiple_feature_reduction_ch', default='', help='specifies different feature reduction ch for different shader')
    parser.add_argument('--choose_shaders', dest='choose_shaders', type=int, default=0, help='specifies which set of shaders to use')
    parser.add_argument('--alt_dataroot', dest='alt_dataroot', default='', help='specifies alternate dataroot used to replace hard-coded dataroot')
    parser.add_argument('--analyze_encoder_to_output_contribution_only', dest='analyze_encoder_to_output_contribution_only', action='store_true', help='if specified, only analyze encoder channel contribution to output')
    
    parser.set_defaults(is_train=False)
    parser.set_defaults(use_batch=False)
    parser.set_defaults(debug_mode=False)
    parser.set_defaults(preload=True)
    parser.set_defaults(test_training=False)
    parser.set_defaults(add_initial_layers=False)
    parser.set_defaults(add_final_layers=False)
    parser.set_defaults(dilation_remove_large=False)
    parser.set_defaults(dilation_clamp_large=False)
    parser.set_defaults(dilation_remove_layer=False)
    parser.set_defaults(mean_estimator=False)
    parser.set_defaults(accurate_timing=False)
    parser.set_defaults(identity_initialize=False)
    parser.set_defaults(allow_nonzero=False)
    parser.set_defaults(identity_output_layer=True)
    parser.set_defaults(less_aggresive_ini=False)
    parser.set_defaults(train_res=False)
    parser.set_defaults(mean_estimator_memory_efficient=False)
    parser.set_defaults(tiled_training=False)
    parser.set_defaults(first_last_only=False)
    parser.set_defaults(last_only=False)
    parser.set_defaults(render_fix_spatial_sample=False)
    parser.set_defaults(render_zero_spatial_sample=False)
    parser.set_defaults(mean_var_only=False)
    parser.set_defaults(every_nth_stratified=False)
    parser.set_defaults(aux_plus_manual_features=False)
    parser.set_defaults(additional_features=True)
    parser.set_defaults(include_noise_feature=False)
    parser.set_defaults(no_noise_feature=False)
    parser.set_defaults(perceptual_loss=False)
    parser.set_defaults(relax_clipping=False)
    parser.set_defaults(train_with_zero_samples=False)
    parser.set_defaults(tile_only=False)
    parser.set_defaults(write_summary=True)
    parser.set_defaults(lpips_loss=False)
    parser.set_defaults(l2_loss=True)
    parser.set_defaults(same_sample_all_pix=False)
    parser.set_defaults(analyze_channel=False)
    parser.set_defaults(analyze_current_only=False)
    parser.set_defaults(additional_input=False)
    parser.set_defaults(feature_size_only=False)
    parser.set_defaults(automatic_subsample=False)
    parser.set_defaults(automate_raymarching_def=False)
    parser.set_defaults(log_only_return_def_raymarching=True)
    parser.set_defaults(render_no_video=False)
    parser.set_defaults(collect_validate_loss=False)
    parser.set_defaults(read_from_best_validation=False)
    parser.set_defaults(overwrite_option_file=True)
    parser.set_defaults(analyze_encoder_to_output_contribution_only=False)
    
    return parser

def main():
    parser = generate_parser()
    
    args = parser.parse_args()

    main_network(args)

def main_network(args):


    # batch_size can work with 2, but OOM with 4

    if args.name == '':
        args.name = ''.join(random.choice(string.digits) for _ in range(5))

    if not os.path.isdir(args.name):
        os.makedirs(args.name)

    # for simplicity, only allow batch_size=1 for inference
    # TODO: can come back to relax this contrain.
    inference_entire_img_valid = False
    if not args.is_train:
        args.use_batch = False
        args.batch_size = 1
        # at test time, always inference an entire image, rahter than tiles
        if not args.test_training:
            inference_entire_img_valid = True

    global actual_conv_channel
    actual_conv_channel *= args.conv_channel_multiplier
    if actual_conv_channel == 0:
        actual_conv_channel = args.conv_channel_no
    if args.initial_layer_channels < 0:
        args.initial_layer_channels = actual_conv_channel
    if args.final_layer_channels < 0:
        args.final_layer_channels = actual_conv_channel
        
    global dilation_threshold
    dilation_threshold = args.dilation_threshold
    global padding_offset
    padding_offset = 4 * args.dilation_threshold

    assert (not args.dilation_clamp_large) or (not args.dilation_remove_large) or (not args.dilation_remove_layer)
    global dilation_clamp_large
    dilation_clamp_large = args.dilation_clamp_large
    global dilation_remove_large
    dilation_remove_large = args.dilation_remove_large
    global dilation_remove_layer
    dilation_remove_layer = args.dilation_remove_layer

    if not args.add_final_layers:
        args.final_layer_channels = -1

    global width
    width = args.input_w
    global height
    height = args.input_h

    global allow_nonzero
    allow_nonzero = args.allow_nonzero

    global identity_output_layer
    identity_output_layer = args.identity_output_layer

    global less_aggresive_ini
    less_aggresive_ini = args.less_aggresive_ini

    if args.render_only:
        args.is_train = False
        if args.render_fov != '':
            args.fov = args.render_fov


    if args.tiled_training or args.tile_only:
        global conv_padding
        conv_padding = "VALID"
        
    if args.tiled_training:
        assert width % args.tiled_w == 0
        assert height % args.tiled_h == 0
        ntiles_w = width / args.tiled_w
        ntiles_h = height / args.tiled_h
    else:
        ntiles_w = 1
        ntiles_h = 1

    render_sigma = [args.render_sigma, args.render_sigma, 0]
    
    if (args.tiled_training or args.tile_only) and (not inference_entire_img_valid):
        output_pl_w = args.tiled_w
        output_pl_h = args.tiled_h
    else:
        output_pl_w = args.input_w
        output_pl_h = args.input_h
           
    
            
    
    
    
    
    avg_loss = 0
    tf.summary.scalar('avg_loss', avg_loss)

    avg_loss_l2 = 0
    tf.summary.scalar('avg_loss_l2', avg_loss_l2)


    avg_training_loss = 0
    tf.summary.scalar('avg_training_loss', avg_training_loss)

    avg_test_close = 0
    tf.summary.scalar('avg_test_close', avg_test_close)
    avg_test_far = 0
    tf.summary.scalar('avg_test_far', avg_test_far)
    avg_test_middle = 0
    tf.summary.scalar('avg_test_middle', avg_test_middle)
    avg_test_all = 0
    tf.summary.scalar('avg_test_all', avg_test_all)
    reg_loss = 0
    tf.summary.scalar('reg_loss', reg_loss)
    l2_loss = 0
    tf.summary.scalar('l2_loss', l2_loss)
    perceptual_loss = 0
    tf.summary.scalar('perceptual_loss', perceptual_loss)
    
    validate = 0
    tf.summary.scalar('validate', validate)

    
    merged = tf.summary.merge_all()
    
    orig_args = copy.copy(args)
    
    if args.is_train:
        global_epoch = args.epoch
    elif args.analyze_channel:
        if args.analyze_encoder_to_output_contribution_only:
            global_epoch = args.which_epoch + 1
        else:
            global_epoch = args.which_epoch + 2 + args.conv_channel_no // analyze_per_ch
    else:
        global_epoch = args.which_epoch + 1
        
    global all_shaders, shaders_pool, shaders_aux_pool
        
    if args.manual_features_only:
        shaders_pool = shaders_aux_pool
        
    all_shaders = shaders_pool[args.choose_shaders]
        
    all_train_writers = [None] * len(all_shaders)
    
    if args.multiple_feature_reduction_ch != '':
        multiple_feature_reduction_ch = [int(val) for val in args.multiple_feature_reduction_ch.split(',')]
        assert len(multiple_feature_reduction_ch) == len(all_shaders)
    else:
        multiple_feature_reduction_ch = None
        
    alt_dataroot = None
    if args.alt_dataroot != '':
        alt_dataroot = args.alt_dataroot.split(',')
        assert len(alt_dataroot) == len(all_shaders)
    
    T0 = time.time()
    
    for global_e in range(args.which_epoch + 1, global_epoch + 1):
        
        print(global_e)
        
        analyze_encoder_only = False
        if args.analyze_channel:
            if args.analyze_encoder_to_output_contribution_only:
                analyze_encoder_only = False
            elif global_e - args.which_epoch <= args.conv_channel_no // analyze_per_ch:
                analyze_encoder_only = True
    
        for shader_ind in range(len(all_shaders)):

            tf.reset_default_graph()

            sess = tf.Session()

            shader_name, geometry, dataroot, extra_args = all_shaders[shader_ind]
            
            if alt_dataroot is not None:
                dataroot = alt_dataroot[shader_ind]

            print('running shader %s' % shader_name)

            args = copy.copy(orig_args)

            args.shader_name = shader_name
            args.geometry = geometry
            args.dataroot = os.path.join(orig_args.dataroot_parent, dataroot)

            for key in extra_args.keys():
                if key == 'specified_ind':
                    setattr(args, key, os.path.join(orig_args.dataroot_parent, dataroot, extra_args[key]))
                else:
                    setattr(args, key, extra_args[key])
                
            if multiple_feature_reduction_ch is not None:
                args.feature_reduction_ch = multiple_feature_reduction_ch[shader_ind]

            output_names, val_img_names, map_names, val_map_names, grad_names, val_grad_names, add_names, val_add_names, validate_img_names = prepare_data_root(args.dataroot, additional_input=args.additional_input)
            if args.test_training:
                val_img_names = output_names
                val_map_names = map_names
                val_grad_names = grad_names
                val_add_names = add_names

            camera_pos = tf.placeholder(dtype, shape=[6, args.batch_size])
            shader_time = tf.placeholder(dtype, shape=args.batch_size)
            output_pl = tf.placeholder(tf.float32, shape=[None, output_pl_h, output_pl_w, 3])

            if args.is_train or args.test_training:
                camera_pos_vals = np.load(os.path.join(args.dataroot, 'train.npy'))
                time_vals = np.load(os.path.join(args.dataroot, 'train_time.npy'))
                if args.tile_only:
                    tile_start_vals = np.load(os.path.join(args.dataroot, 'train_start.npy'))
            else:
                camera_pos_vals = np.concatenate((
                                    np.load(os.path.join(args.dataroot, 'test_close.npy')),
                                    np.load(os.path.join(args.dataroot, 'test_far.npy')),
                                    np.load(os.path.join(args.dataroot, 'test_middle.npy'))
                                    ), axis=0)

                time_vals = np.load(os.path.join(args.dataroot, 'test_time.npy'))

            nexamples = time_vals.shape[0]
            
            if args.specified_ind != '':
                my_specified_ind_file = os.path.join(args.name, '%s_specified_ind.npy' % args.shader_name)
                specified_ind = np.load(args.specified_ind)
                if not os.path.exists(my_specified_ind_file):
                    shutil.copyfile(args.specified_ind, my_specified_ind_file)
                else:
                    my_ind = np.load(my_specified_ind_file)
                    assert np.allclose(my_ind, specified_ind)
            else:
                specified_ind = None
            
            def feature_reduction_layer(input_to_network, _replace_normalize_weights=None, shadername=''):
                with tf.variable_scope("feature_reduction" + shadername, reuse=tf.AUTO_REUSE):

                    actual_nfeatures = args.input_nc

                    if args.feature_reduction_ch > 0:
                        actual_feature_reduction_ch = args.feature_reduction_ch
                    else:
                        actual_feature_reduction_ch = args.initial_layer_channels

                    w_shape = [1, 1, actual_nfeatures, actual_feature_reduction_ch]
                    conv = tf.nn.conv2d
                    strides = [1, 1, 1, 1]

                    weights = tf.get_variable('w0', w_shape, initializer=tf.contrib.layers.xavier_initializer() if not args.identity_initialize else identity_initializer(color_inds, ndims=2))

                    weights_to_input = weights

                    reduced_feat = conv(input_to_network, weights_to_input, strides, "SAME")

                    if args.initial_layer_channels <= actual_conv_channel:
                        ini_id = True
                    else:
                        ini_id = False

                    if args.add_initial_layers:
                        for nlayer in range(3):
                            reduced_feat = slim.conv2d(reduced_feat, actual_initial_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(allow_map_to_less=True), scope='initial_'+str(nlayer), padding=conv_padding)          

                return reduced_feat

            with tf.variable_scope("shader"):
                output_type = 'remove_constant'

                if args.mean_estimator and not args.mean_estimator_memory_efficient:
                    shader_samples = args.estimator_samples
                else:
                    shader_samples = args.batch_size

                color_inds = []
                if args.tiled_training or args.tile_only:

                    h_start = tf.placeholder(dtype=dtype, shape=args.batch_size)
                    w_start = tf.placeholder(dtype=dtype, shape=args.batch_size)

                    if not inference_entire_img_valid:
                        h_offset = args.tiled_h + padding_offset
                        w_offset = args.tiled_w + padding_offset
                    else:
                        h_offset = args.input_h + padding_offset
                        w_offset = args.input_w + padding_offset

                    if args.is_train or args.tile_only:
                        feed_samples = None
                    else:
                        # for inference, need to ensure that noise samples used within an image is the same
                        if not args.mean_estimator:
                            feed_samples = [tf.placeholder(dtype=dtype, shape=[args.batch_size, height+padding_offset, width+padding_offset]), tf.placeholder(dtype=dtype, shape=[args.batch_size, height+padding_offset, width+padding_offset])]
                        else:
                            feed_samples = [tf.placeholder(dtype=dtype, shape=[args.estimator_samples, height+padding_offset, width+padding_offset]), tf.placeholder(dtype=dtype, shape=[args.estimator_samples, height+padding_offset, width+padding_offset])]


                else:
                    h_start = tf.constant([0.0])
                    w_start = tf.constant([0.0])
                    h_offset = height
                    w_offset = width
                    feed_samples = None
                zero_samples = False


                spatial_samples = None
                if args.render_fix_spatial_sample:
                    spatial_samples = [numpy.random.normal(size=(1, h_offset, w_offset)), numpy.random.normal(size=(1, h_offset, w_offset))]
                elif args.render_zero_spatial_sample:
                    spatial_samples = [numpy.zeros((1, h_offset, w_offset)), numpy.zeros((1, h_offset, w_offset))]

                if args.train_with_zero_samples:
                    zero_samples = True


                debug = []

                def generate_input_to_network_wrapper():
                    def func():

                        return get_tensors(args.dataroot, args.name, camera_pos, shader_time, output_type, shader_samples, shader_name=args.shader_name, geometry=args.geometry, color_inds=color_inds, manual_features_only=args.manual_features_only, h_start=h_start, h_offset=h_offset, w_start=w_start, w_offset=w_offset, samples=feed_samples, fov=args.fov, first_last_only=args.first_last_only, last_only=args.last_only, subsample_loops=args.subsample_loops, last_n=args.last_n, first_n=args.first_n, first_n_no_last=args.first_n_no_last, mean_var_only=args.mean_var_only, zero_samples=zero_samples, render_fix_spatial_sample=args.render_fix_spatial_sample, render_zero_spatial_sample=args.render_zero_spatial_sample, spatial_samples=spatial_samples, every_nth=args.every_nth, every_nth_stratified=args.every_nth_stratified, additional_features=args.additional_features, ignore_last_n_scale=args.ignore_last_n_scale, include_noise_feature=args.include_noise_feature, no_noise_feature=args.no_noise_feature, relax_clipping=args.relax_clipping, render_sigma=render_sigma, same_sample_all_pix=args.same_sample_all_pix, automatic_subsample=args.automatic_subsample, automate_raymarching_def=args.automate_raymarching_def, log_only_return_def_raymarching=args.log_only_return_def_raymarching, SELECT_FEATURE_THRE=args.SELECT_FEATURE_THRE, debug=debug, compiler_problem_idx=shader_ind, feature_normalize_lo_pct=args.feature_normalize_lo_pct, specified_ind=specified_ind, write_file=args.overwrite_option_file, alt_dir=args.test_output_dir)

                    return func

                generate_input_to_network = generate_input_to_network_wrapper()
                input_to_network = generate_input_to_network()

                if args.feature_size_only:
                    print('feature size: ', int(input_to_network.shape[-1]))
                    return

                output = output_pl

                if (args.tiled_training or args.tile_only) and args.mean_estimator:
                    input_to_network = tf.slice(input_to_network, [0, padding_offset // 2, padding_offset // 2, 0], [args.estimator_samples, output_pl_h, output_pl_w, 3])
                elif args.additional_input:
                    additional_input = tf.pad(additional_input_pl, [[0, 0], [padding_offset // 2, padding_offset // 2], [padding_offset // 2, padding_offset // 2], [0, 0]], "SYMMETRIC")
                    print('concatenating additional input')
                    input_to_network = tf.concat((input_to_network, additional_input), axis=3)


                if len(color_inds) == 3:
                    color_inds = color_inds[::-1]

                if input_to_network is not None:
                    args.input_nc = int(input_to_network.shape[-1])
                    debug_input = input_to_network


            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                if args.debug_mode and args.mean_estimator:
                    with tf.variable_scope("shader"):
                        network = tf.reduce_mean(input_to_network, axis=0, keep_dims=True)
                    sparsity_loss = 0
                else:
                    if args.input_nc <= actual_conv_channel:
                        ini_id = True
                    else:
                        ini_id = False
                    alpha = tf.placeholder(tf.float32)
                    alpha_val = 1.0

                    replace_normalize_weights = None
                    normalize_weights = None

                    sparsity_loss = tf.constant(0.0, dtype=dtype)

                    actual_initial_layer_channels = args.initial_layer_channels

                    feature_reduction_tensor = None


                    input_to_network = feature_reduction_layer(input_to_network, _replace_normalize_weights=replace_normalize_weights, shadername=args.shader_name)
                    feature_reduction_tensor = input_to_network
                    
                    reduced_dim_feature = input_to_network

                    if not analyze_encoder_only:

                        network=build(input_to_network, ini_id, final_layer_channels=args.final_layer_channels, identity_initialize=args.identity_initialize, output_nc=3)




            loss = 0.0
            loss_to_opt = 0.0
                        
            
            if not analyze_encoder_only:
            
                weight_map = tf.placeholder(tf.float32,shape=[None,None,None])

                if args.l2_loss:
                    if (not args.train_res) or (args.debug_mode and args.mean_estimator):
                        diff = network - output
                    else:
                        input_color = tf.stack([debug_input[..., ind] for ind in color_inds], axis=-1)
                        diff = network + input_color - output
                        network += input_color

                    if args.RGB_norm % 2 != 0:
                        diff = tf.abs(diff)
                    powered_diff = diff ** args.RGB_norm

                    loss_per_sample = tf.reduce_mean(powered_diff, (1, 2, 3))
                    loss = tf.reduce_mean(loss_per_sample)
                else:
                    loss = tf.constant(0.0, dtype=dtype)

                loss_l2 = loss
                loss_add_term = loss

                if args.lpips_loss:
                    sys.path += ['lpips-tensorflow']
                    import lpips_tf
                    loss_lpips = lpips_tf.lpips(network, output, model='net-lin', net=args.lpips_net)

                    perceptual_loss_add = args.lpips_loss_scale * loss_lpips
                    if args.batch_size > 1:
                        perceptual_loss_add = tf.reduce_mean(perceptual_loss_add)
                    loss += perceptual_loss_add
                else:
                    perceptual_loss_add = tf.constant(0)


                loss_to_opt = loss + sparsity_loss
                gen_loss_GAN = tf.constant(0.0)
                discrim_loss = tf.constant(0.0)

                with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):

                    adam_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
                    var_list = tf.trainable_variables()

                    adam_before = adam_optimizer

                    opt=adam_optimizer.minimize(loss_to_opt,var_list=var_list)

            all_vars = tf.trainable_variables()

            encoder_vars = [var for var in all_vars if 'feature_reduction' in var.name]

            generator_vars = [var for var in all_vars if 'feature_reduction' not in var.name]

            encoder_saver = tf.train.Saver(encoder_vars, max_to_keep=1000)
            

            if analyze_encoder_only:
                savers = [encoder_saver]
                save_names = ['%s_encoder' % args.shader_name]
            else:
                gen_saver = tf.train.Saver(generator_vars, max_to_keep=1000)
                savers = [encoder_saver, gen_saver]
                save_names = ['%s_encoder' % args.shader_name, 'model_gen']




            #print("initialize local vars")
            sess.run(tf.local_variables_initializer())
            #print("initialize global vars")
            sess.run(tf.global_variables_initializer())

            read_from_epoch = False

            if (not (args.debug_mode and args.mean_estimator)) and (not args.collect_validate_loss):

                ckpts = [None] * len(savers)

                if args.read_from_best_validation:
                    assert not args.is_train
                    for c_i in range(len(savers)):
                        ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, 'best_val', save_names[c_i]))
                    if None in ckpts:
                        print('No best validation result exist')
                        raise
                    read_from_epoch = False

                else:

                    if not args.is_train:
                        # in this version we do not save models to root directory anymore
                        assert args.which_epoch > 0
                        encoder_epoch = args.which_epoch
                        others_epoch = args.which_epoch
                        read_from_epoch = True
                    else:
                        encoder_epoch = global_e - 1

                        if shader_ind == 0:
                            others_epoch = global_e - 1
                        else:
                            others_epoch = global_e

                        read_from_epoch = False

                    encoder_saver_exist = True
                    other_saver_exist = True

                    for c_i in range(len(savers)):
                        if savers[c_i] == encoder_saver:
                            ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, "%04d"%int(encoder_epoch), save_names[c_i]))
                            if ckpts[c_i] is None:
                                encoder_saver_exist = False
                        else:
                            ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, "%04d"%int(others_epoch), save_names[c_i]))
                            if ckpts[c_i] is None:
                                other_saver_exist = False

                    if not other_saver_exist:
                        assert not read_from_epoch
                        assert global_e == 1 and shader_ind == 0

                    if not encoder_saver_exist:
                        assert not read_from_epoch
                        assert global_e == 1


                for c_i in range(len(ckpts)):
                    if ckpts[c_i] is not None:
                        ckpt = ckpts[c_i]
                        print('loaded '+ ckpt.model_checkpoint_path)
                        savers[c_i].restore(sess, ckpt.model_checkpoint_path)
                print('finished loading')




            num_epoch = args.epoch_per_shader



            def read_ind(img_arr, name_arr, id, is_npy):
                img_arr[id] = read_name(name_arr[id], is_npy)
                if img_arr[id] is None:
                    return False
                elif img_arr[id].shape[0] * img_arr[id].shape[1] > 2200000:
                    img_arr[id] = None
                    return False
                return True



            if args.preload and args.is_train:
                output_images = np.empty([camera_pos_vals.shape[0], output_pl_h, output_pl_w, 3])
                all_grads = [None] * camera_pos_vals.shape[0]
                all_adds = np.empty([camera_pos_vals.shape[0], output_pl_h, output_pl_w, 1])
                for id in range(camera_pos_vals.shape[0]):
                    output_images[id, :, :, :] = read_name(output_names[id], False)
                    print(id)
                    if args.additional_input:
                        all_adds[id, :, :, 0] = read_name(add_names[id], True)
                        
            def read_name(name, is_npy, is_bin=False):
                if not os.path.exists(name):
                    return None
                if not is_npy and not is_bin:
                    return np.float32(cv2.imread(name, -1)) / 255.0
                elif is_npy:
                    ans = np.load(name)
                    return ans
                else:
                    return np.fromfile(name, dtype=np.float32).reshape([640, 960, args.input_nc])

            if args.analyze_channel:
                
                if args.test_output_dir != '':
                    args.name = args.test_output_dir

                
                
                feed_dict = {}
                current_dir = 'train' if args.test_training else 'test'
                current_dir = os.path.join(args.name, current_dir)
                
                if not os.path.isdir(current_dir):
                    os.makedirs(current_dir)
                    
                valid_inds = np.load(os.path.join(args.name, 'valid_inds.npy'))
                
                if specified_ind is not None:
                    valid_inds = valid_inds[specified_ind]
                
                if args.tile_only:
                    if inference_entire_img_valid:
                        feed_dict[h_start] = np.array([- padding_offset // 2]).astype('i')
                        feed_dict[w_start] = np.array([- padding_offset // 2]).astype('i')
                        
                       
                if not analyze_encoder_only:
                    
                    if args.analyze_encoder_to_output_contribution_only:
                        total_loss_grad = tf.gradients(loss_l2, reduced_dim_feature, stop_gradients=tf.trainable_variables())[0]
                        total_loss_taylor = tf.reduce_mean(tf.abs(tf.reduce_mean(total_loss_grad * reduced_dim_feature, (1, 2))), 0)
                        total_loss_taylor_vals = np.zeros(args.conv_channel_no)
                    else:
                        total_loss_grad = tf.gradients(loss_l2, debug_input, stop_gradients=tf.trainable_variables())[0]
                        total_loss_taylor = tf.reduce_mean(tf.abs(tf.reduce_mean(total_loss_grad * debug_input, (1, 2))), 0)

                        total_loss_taylor_vals = np.zeros(args.input_nc)
                
                    for i in range(len(val_img_names)):
                        print(args.shader_name, i)
                        output_ground = np.expand_dims(read_name(val_img_names[i], False, False), 0)

                        camera_val = np.expand_dims(camera_pos_vals[i, :], axis=1)
                        feed_dict[camera_pos] = camera_val
                        feed_dict[shader_time] = time_vals[i:i+1]

                        if not inference_entire_img_valid:
                            feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset // 2
                            feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset // 2

                        feed_dict[output_pl] = output_ground

                        current_total_taylor = sess.run(total_loss_taylor, feed_dict=feed_dict)

                        total_loss_taylor_vals += current_total_taylor

                    total_loss_taylor_vals /= len(val_img_names)
                    numpy.save(os.path.join(current_dir, '%stotal_loss_taylor_vals_%s_%s.npy' % ('encoder_to_output_' if args.analyze_encoder_to_output_contribution_only else '', args.shader_name, 'train' if args.is_train else 'test')), total_loss_taylor_vals)
                    
                    continue
                
                
                encoder_channelwise_taylor_vals = np.zeros((analyze_per_ch, args.input_nc))
                
                encoder_channelwise_taylors = []
                channelwise_sum = tf.reduce_sum(feature_reduction_tensor, (0, 1, 2))
                
                start_ch = (global_e - args.which_epoch - 1) * analyze_per_ch

                for i in range(start_ch, start_ch + analyze_per_ch):
                    encoder_channelwise_grad = tf.gradients(channelwise_sum[i], debug_input, stop_gradients=tf.trainable_variables())[0]
                    encoder_channelwise_taylor = tf.reduce_mean(tf.abs(encoder_channelwise_grad * debug_input), (0, 1, 2))
                    encoder_channelwise_taylors.append(encoder_channelwise_taylor)
                    
                encoder_channelwise_taylors = tf.stack(encoder_channelwise_taylors, 0)
                

                
                for i in range(len(val_img_names)):
                    print(args.shader_name, start_ch, i)

                    output_ground = np.expand_dims(read_name(val_img_names[i], False, False), 0)

                    camera_val = np.expand_dims(camera_pos_vals[i, :], axis=1)
                    feed_dict[camera_pos] = camera_val
                    feed_dict[shader_time] = time_vals[i:i+1]

                    if not inference_entire_img_valid:
                        feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset // 2
                        feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset // 2

                    feed_dict[output_pl] = output_ground

                    current_channelwise_taylors = sess.run(encoder_channelwise_taylors, feed_dict=feed_dict)

                    encoder_channelwise_taylor_vals += current_channelwise_taylors
                
                encoder_channelwise_taylor_vals /= len(val_img_names)
                
                if start_ch == 0:
                    numpy.save(os.path.join(current_dir, 'encoder_channelwise_taylor_vals_%s.npy' % args.shader_name), encoder_channelwise_taylor_vals)
                else:
                    old_vals = np.load(os.path.join(current_dir, 'encoder_channelwise_taylor_vals_%s.npy' % args.shader_name))
                    assert old_vals.shape == (start_ch, args.input_nc)
                    new_vals = np.concatenate((old_vals, encoder_channelwise_taylor_vals), 0)
                    numpy.save(os.path.join(current_dir, 'encoder_channelwise_taylor_vals_%s.npy' % args.shader_name), new_vals)

                
                
                
                print('max channelwise ind')
                str_max_channelwise_ind = ''
                for i in range(encoder_channelwise_taylor_vals.shape[0]):
                    max_channelwise_ind = np.argsort(encoder_channelwise_taylor_vals[i])[::-1]
                    print(i + start_ch, ':, ', valid_inds[max_channelwise_ind[:5]])
                    str_max_channelwise_ind += 'encoder channel %d:\n' % (i + start_ch)
                    str_max_channelwise_ind += ', '.join([str(ind) for ind in valid_inds[max_channelwise_ind[:20]]])
                    str_max_channelwise_ind += '\n'
                
                if start_ch == 0:
                    access = 'w'
                else:
                    access = 'a'
                open(os.path.join(current_dir, 'encoder_max_channelwise_ind_%s.txt' % args.shader_name), access).write(str_max_channelwise_ind)        
                
                continue

            if args.is_train:
                if args.write_summary:
                    if all_train_writers[shader_ind] is None:
                        all_train_writers[shader_ind] = tf.summary.FileWriter(os.path.join(args.name, args.shader_name), sess.graph)


                rec_arr_len = time_vals.shape[0]


                all=np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_l2=np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_training_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_perceptual = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_gen_gan_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_discrim_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)

                min_avg_loss = 1e20
                old_val_loss = 1e20
                

                for epoch in range(1, num_epoch+1):

                    cnt=0

                    permutation = np.random.permutation(int(nexamples * ntiles_h * ntiles_w))
                    nupdates = permutation.shape[0] if not args.use_batch else int(np.ceil(float(permutation.shape[0]) / args.batch_size))
                    sub_epochs = 1

                    feed_dict={}


                    for i in range(nupdates):


                        st=time.time()
                        start_id = i * args.batch_size
                        end_id = min(permutation.shape[0], (i+1)*args.batch_size)

                        frame_idx = (permutation[start_id:end_id] / (ntiles_w * ntiles_h)).astype('i')
                        tile_idx = (permutation[start_id:end_id] % (ntiles_w * ntiles_h)).astype('i')
                        run_options = None
                        run_metadata = None


                        T_before = time.time()
                            
                        if not args.preload:
                            

                            output_arr = np.empty([args.batch_size, output_pl_h, output_pl_w, 3])

                            for img_idx in range(frame_idx.shape[0]):
                                output_arr[img_idx, :, :, :] = read_name(output_names[frame_idx[img_idx]], False)

                            if args.additional_input:
                                additional_arr = np.empty([args.batch_size, output_pl.shape[1].value, output_pl.shape[2].value, 1])
                                for img_idx in range(frame_idx.shape[0]):
                                    additional_arr[img_idx, :, :, 0] = read_name(add_names[frame_idx[img_idx]], True)
                                    
                            

                        else:
                            output_arr = output_images[frame_idx]
                            if args.additional_input:
                                additional_arr = all_adds[frame_idx]
                                
                        T_load = time.time() - T_before

                        if args.tiled_training:
                            assert args.batch_size == 1
                            tile_idx = tile_idx[0]
                            tile_h = tile_idx // ntiles_w
                            tile_w  = tile_idx % ntiles_w
                            output_patch = output_arr[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :]
                            if numpy.sum(output_patch > 0) < args.tiled_h * args.tiled_w * 3 / 100:
                                continue
                            output_arr = output_patch
                            for key, value in feed_dict.items():
                                if isinstance(value, numpy.ndarray) and len(value.shape) >= 3 and value.shape[1] == height and value.shape[2] == width:
                                    if len(value.shape) == 3:
                                        tiled_value = value[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w)]
                                    else:
                                        tiled_value = value[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :]
                                    feed_dict[key] = tiled_value
                            feed_dict[h_start] = numpy.array([tile_h * height / ntiles_h - padding_offset / 2])
                            feed_dict[w_start] = numpy.array([tile_w * width / ntiles_w - padding_offset / 2])

                        if args.tile_only:
                            feed_dict[h_start] = tile_start_vals[frame_idx, 0] - padding_offset / 2
                            feed_dict[w_start] = tile_start_vals[frame_idx, 1] - padding_offset / 2

                        feed_dict[output_pl] = output_arr
                        if args.additional_input:
                            feed_dict[additional_input_pl] = additional_arr

                        camera_val = camera_pos_vals[frame_idx, :].transpose()
                        feed_dict[camera_pos] = camera_val
                        feed_dict[shader_time] = time_vals[frame_idx]



                        st1 = time.time()

                        _,current, current_l2, current_training, current_perceptual, current_gen_loss_GAN, current_discrim_loss, =sess.run([opt,loss, loss_l2, loss_to_opt, perceptual_loss_add, gen_loss_GAN, discrim_loss],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

                        st2 = time.time()


                        if numpy.isnan(current):
                            print(frame_idx, tile_idx)
                            raise


                        current_slice = permutation[start_id:end_id]
                        all[current_slice]=current
                        all_l2[current_slice]=current_l2
                        all_training_loss[current_slice] = current_training
                        all_perceptual[current_slice] = current_perceptual
                        all_gen_gan_loss[current_slice] = current_gen_loss_GAN
                        all_discrim_loss[current_slice] = current_discrim_loss
                        cnt += args.batch_size if args.use_batch else 1
                        print("%d %d %.5f %.5f %.2f %.2f %.2f"%((global_e - 1) * num_epoch + epoch, cnt, current, np.mean(all[np.where(all)]), time.time()-st, st2-st1, T_load))

                    avg_loss = np.mean(all[np.where(all)])
                    avg_loss_l2 = np.mean(all_l2[np.where(all_l2)])
                    avg_training_loss = np.mean(all_training_loss)
                    avg_perceptual = np.mean(all_perceptual)
                    avg_gen_gan = np.mean(all_gen_gan_loss)
                    avg_discrim = np.mean(all_discrim_loss)

                    if min_avg_loss > avg_training_loss:
                        min_avg_loss = avg_training_loss

                    if args.write_summary:
                        summary = tf.Summary()
                        summary.value.add(tag='avg_loss', simple_value=avg_loss)
                        summary.value.add(tag='avg_loss_l2', simple_value=avg_loss_l2)
                        summary.value.add(tag='avg_training_loss', simple_value=avg_training_loss)
                        summary.value.add(tag='avg_perceptual', simple_value=avg_perceptual)
                        summary.value.add(tag='avg_gen_gan', simple_value=avg_gen_gan)
                        summary.value.add(tag='avg_discrim', simple_value=avg_discrim)
                        all_train_writers[shader_ind].add_summary(summary, (global_e - 1) * num_epoch + epoch)

                    smart_mkdir("%s/%04d/%04d/%s"%(args.name, global_e, epoch, args.shader_name))
                    target=open("%s/%04d/%04d/%s/score.txt"%(args.name, global_e, epoch, args.shader_name),'w')
                    target.write("%f"%np.mean(all[np.where(all)]))
                    target.close()

                if global_e % args.save_frequency == 0:
                    for s_i in range(len(savers)):
                        ckpt_dir = os.path.join("%s/%04d"%(args.name, global_e), save_names[s_i])
                        if not os.path.isdir(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        savers[s_i].save(sess,"%s/model.ckpt" % ckpt_dir)



            if not args.is_train:

                if args.test_output_dir != '':
                    args.name = args.test_output_dir

                if args.collect_validate_loss:
                    assert args.test_training
                    dirs = sorted(os.listdir(args.name))
                    camera_pos_vals = np.load(os.path.join(args.dataroot, 'validate.npy'))
                    time_vals = np.load(os.path.join(args.dataroot, 'validate_time.npy'))
                    if args.tile_only:
                        tile_start_vals = np.load(os.path.join(args.dataroot, 'validate_start.npy'))

                    validate_imgs = []

                    for name in validate_img_names:
                        validate_imgs.append(np.expand_dims(read_name(name, False, False), 0))

                    # stored in the order of
                    # epoch, current, current_l2, current_perceptual, current_gen, current_discrim
                    all_vals = []

                    all_example_vals = np.empty([len(validate_img_names), 6])

                    for dir in dirs:
                        success = False
                        try:
                            global_e = int(dir)
                            success = True
                        except:
                            pass

                        if not success:
                            continue

                        ckpts = [None] * len(savers)
                        for c_i in range(len(savers)):
                            ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, dir, save_names[c_i]))

                        if None not in ckpts:
                            for c_i in range(len(ckpts)):
                                ckpt = ckpts[c_i]
                                print('loaded '+ ckpt.model_checkpoint_path)
                                savers[c_i].restore(sess, ckpt.model_checkpoint_path)
                        else:
                            continue

                        for ind in range(len(validate_img_names)):
                            feed_dict = {camera_pos: np.expand_dims(camera_pos_vals[ind], 1),
                                         shader_time: time_vals[ind:ind+1],
                                         output_pl: validate_imgs[ind]}

                            if args.tile_only:
                                feed_dict[h_start] = tile_start_vals[ind:ind+1, 0] - padding_offset / 2
                                feed_dict[w_start] = tile_start_vals[ind:ind+1, 1] - padding_offset / 2
                            else:
                                feed_dict[h_start] = np.array([- padding_offset / 2])
                                feed_dict[w_start] = np.array([- padding_offset / 2])

                            current, current_l2, current_perceptual, current_gen_loss_GAN, current_discrim_loss = sess.run([loss, loss_l2, perceptual_loss_add, gen_loss_GAN, discrim_loss], feed_dict=feed_dict)
                            all_example_vals[ind] = np.array([global_e, current, current_l2, current_perceptual, current_gen_loss_GAN, current_discrim_loss])

                        all_vals.append(np.mean(all_example_vals, 0))

                    all_vals = np.array(all_vals)
                    np.save(os.path.join(args.name, '%s_validation.npy' % args.shader_name), all_vals)
                    open(os.path.join(args.name, 'validation.txt'), 'w').write('validation dataset: %s\n raw data stored in: %s\n' % (args.dataroot, os.uname().nodename))
                    
                    min_idx = np.argsort(all_vals[:, 1])
                    min_epoch = all_vals[min_idx[0], 0]
                    
                    open(os.path.join(args.name, '%s_best_val_epoch.txt' % args.shader_name), 'w').write(str(int(min_epoch)))
                    
                    figure = plt.figure(figsize=(20,10))

                    plt.subplot(5, 1, 1)
                    plt.plot(all_vals[:, 0], all_vals[:, 1])
                    plt.ylabel('all non GAN loss')

                    plt.subplot(5, 1, 2)
                    plt.plot(all_vals[:, 0], all_vals[:, 2])
                    plt.ylabel('l2 loss')

                    plt.subplot(5, 1, 3)
                    plt.plot(all_vals[:, 0], all_vals[:, 3])
                    plt.ylabel('perceptual loss')

                    plt.subplot(5, 1, 4)
                    plt.plot(all_vals[:, 0], all_vals[:, 4])
                    plt.ylabel('GAN generator loss')

                    plt.subplot(5, 1, 5)
                    plt.plot(all_vals[:, 0], all_vals[:, 5])
                    plt.ylabel('GAN discrim loss')

                    plt.savefig(os.path.join(args.name, '%s_validation.png' % args.shader_name))
                    plt.close(figure)

                else:

                    if args.render_only:
                        camera_pos_vals = np.load(args.render_camera_pos)
                        time_vals = np.load(args.render_t)
                        if not inference_entire_img_valid:
                            tile_start_vals = np.load(args.render_tile_start)

                    if args.render_only:
                        debug_dir = args.name + '/%s' % args.render_dirname
                    elif args.mean_estimator:
                        debug_dir = "%s/mean%d"%(args.name, args.estimator_samples)
                        debug_dir += '_test' if not args.test_training else '_train'
                        #debug_dir = "%s/mean%d"%('/localtmp/yuting', args.estimator_samples)
                    else:
                        #debug_dir = "%s/debug"%args.name
                        debug_dir = args.name + '/' + ('test' if not args.test_training else 'train')
                        if args.debug_mode:
                            debug_dir += '_debug'
                        #debug_dir = "%s/debug"%'/localtmp/yuting'



                    if read_from_epoch:
                        debug_dir += "_epoch_%04d"%args.which_epoch
                        
                    debug_dir = debug_dir + '_' + args.shader_name

                    if not os.path.isdir(debug_dir):
                        os.makedirs(debug_dir)

                    if args.render_only and os.path.exists(os.path.join(debug_dir, 'video.mp4')):
                        os.remove(os.path.join(debug_dir, 'video.mp4'))

                    if args.render_only:
                        shutil.copyfile(args.render_t, os.path.join(debug_dir, 'render_t.npy'))

                        shutil.copyfile(args.render_camera_pos, os.path.join(debug_dir, 'camera_pos.npy'))

                    nburns = 10

                    if args.repeat_timing > 1:
                        nburns = 20
                        time_stats = numpy.zeros(time_vals.shape[0] * args.repeat_timing)
                        time_count = 0

                    python_time = numpy.zeros(time_vals.shape[0])

                    run_options = None
                    run_metadata = None

                    feed_dict = {}


                    if args.render_only:
                        if h_start.op.type == 'Placeholder':
                            feed_dict[h_start] = np.array([- padding_offset / 2])
                        if w_start.op.type == 'Placeholder':
                            feed_dict[w_start] = np.array([- padding_offset / 2])
                        else:
                            nexamples = time_vals.shape[0]

                        for i in range(nexamples):
                            #feed_dict = {camera_pos: camera_pos_vals[i:i+1, :].transpose(), shader_time: time_vals[i:i+1]}

                            if not inference_entire_img_valid:
                                feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset / 2
                                feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset / 2


                            feed_dict[camera_pos] = camera_pos_vals[i:i+1, :].transpose()
                            feed_dict[shader_time] = time_vals[i:i+1]

                            if args.additional_input:
                                feed_dict[additional_input_pl] = np.expand_dims(np.expand_dims(read_name(val_add_names[i], True), axis=2), axis=0)

                            if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                nruns = args.estimator_samples
                                output_buffer = numpy.zeros((1, 640, 960, 3))
                            else:
                                nruns = 1

                            for _ in range(nruns):
                                output_image = sess.run(network, feed_dict=feed_dict)
                                if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                    output_buffer += output_image[:, :, :, ::-1]



                            if args.mean_estimator:
                                output_image = output_image[:, :, :, ::-1]
                            if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                output_buffer /= args.estimator_samples
                                output_image[:] = output_buffer[:]

                            output_image = np.clip(output_image,0.0,1.0)
                            output_image *= 255.0
                            cv2.imwrite("%s/%06d.png"%(debug_dir, i+1),np.uint8(output_image[0,:,:,:]))

                            print('finished', i)

                        if not args.render_no_video:
                            os.system('ffmpeg %s -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%('', os.path.join(debug_dir, '%06d.png'), os.path.join(debug_dir, 'video.mp4')))
                            open(os.path.join(debug_dir, 'index.html'), 'w+').write("""
            <html>
            <body>
            <br><video controls><source src="video.mp4" type="video/mp4"></video><br>
            </body>
            </html>""")
                        return
                    else:

                        nexamples = time_vals.shape[0]


                        all_test = np.zeros(nexamples, dtype=float)
                        all_grad = np.zeros(nexamples, dtype=float)
                        all_l2 = np.zeros(nexamples, dtype=float)
                        all_perceptual = np.zeros(nexamples, dtype=float)
                        python_time = numpy.zeros(nexamples)


                        for i in range(nexamples):
                            print(i)

                            camera_val = np.expand_dims(camera_pos_vals[i, :], axis=1)
                            #feed_dict = {camera_pos: camera_val, shader_time: time_vals[i:i+1]}
                            feed_dict[camera_pos] = camera_val
                            feed_dict[shader_time] = time_vals[i:i+1]


                            output_ground = np.expand_dims(read_name(val_img_names[i], False, False), 0)


                            if args.tile_only:
                                if not inference_entire_img_valid:
                                    feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset / 2
                                    feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset / 2
                                else:
                                    feed_dict[h_start] = np.array([- padding_offset / 2])
                                    feed_dict[w_start] = np.array([- padding_offset / 2])
                            if output_ground is not None:
                                feed_dict[output_pl] = output_ground
                            if args.additional_input:
                                feed_dict[additional_input_pl] = np.expand_dims(np.expand_dims(read_name(val_add_names[i], True), axis=2), axis=0)



                            output_buffer = np.zeros([1, args.input_h, args.input_w, 3])
                            

                            st = time.time()
                            if args.tiled_training:
                                st_sum = 0
                                timeline_sum = 0
                                l2_loss_val = 0
                                grad_loss_val = 0
                                perceptual_loss_val = 0
                                output_patch = numpy.zeros((1, int(height/ntiles_h), int(width/ntiles_w), 3))
                                if not args.mean_estimator:
                                    feed_dict[feed_samples[0]] = numpy.random.normal(size=(1, height+padding_offset, width+padding_offset))
                                    feed_dict[feed_samples[1]] = numpy.random.normal(size=(1, height+padding_offset, width+padding_offset))
                                else:
                                    feed_dict[feed_samples[0]] = numpy.random.normal(size=(args.estimator_samples, height+padding_offset, width+padding_offset))
                                    feed_dict[feed_samples[1]] = numpy.random.normal(size=(args.estimator_samples, height+padding_offset, width+padding_offset))
                                for tile_h in range(int(ntiles_h)):
                                    for tile_w in range(int(ntiles_w)):
                                        tiled_feed_dict = {}
                                        tiled_feed_dict[h_start] = np.array([tile_h * height / ntiles_h - padding_offset / 2])
                                        tiled_feed_dict[w_start] = np.array([tile_w * width / ntiles_w - padding_offset / 2])
                                        for key, value in feed_dict.items():
                                            if isinstance(value, numpy.ndarray) and len(value.shape) >= 3 and value.shape[1] == height and value.shape[2] == width:
                                                if len(value.shape) == 3:
                                                    tiled_value = value[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w)]
                                                else:
                                                    tiled_value = value[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :]
                                                tiled_feed_dict[key] = tiled_value
                                            else:
                                                tiled_feed_dict[key] = value
                                        st_before = time.time()
                                        if not args.accurate_timing:
                                            output_patch, l2_loss_patch, grad_loss_patch, perceptual_patch = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], feed_dict=tiled_feed_dict, options=run_options, run_metadata=run_metadata)
                                            st_after = time.time()
                                        else:
                                            sess.run([network], feed_dict=feed_dict)
                                            st_after = time.time()
                                            output_patch, l2_loss_patch, grad_loss_patch, perceptual_patch = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], feed_dict=tiled_feed_dict, options=run_options, run_metadata=run_metadata)
                                        st_sum += (st_after - st_before)
                                        print(st_after - st_before)
                                        output_buffer[0, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :] = output_patch[0, :, :, :]
                                        l2_loss_val += l2_loss_patch
                                        grad_loss_val += grad_loss_patch
                                        perceptual_loss_val += perceptual_patch
                                        
                                print("timeline estimate:", timeline_sum)
                                output_image = output_buffer
                                if args.mean_estimator:
                                    output_image = output_image[:, :, :, ::-1]
                                l2_loss_val /= (ntiles_w * ntiles_h)
                                grad_loss_val /= (ntiles_w * ntiles_h)
                                perceptual_loss_val /=  (ntiles_w * ntiles_h)
                                print("rough time estimate:", st_sum)

                            else:
                                if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                    nruns = args.estimator_samples
                                else:
                                    nruns = args.repeat_timing
                                st_sum = 0
                                timeline_sum = 0
                                for k in range(nruns):

                                    st_before = time.time()
                                    if not args.accurate_timing:
                                        output_image, l2_loss_val, grad_loss_val, perceptual_loss_val = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                        st_after = time.time()
                                    else:
                                        sess.run(network, feed_dict=feed_dict)
                                        st_after = time.time()
                                        output_image, l2_loss_val, grad_loss_val, perceptual_loss_val = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                    st_sum += (st_after - st_before)
                                    if args.repeat_timing > 1:
                                        time_stats[time_count] = st_after - st_before
                                        time_count += 1
                                    if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                        output_buffer += output_image[:, :, :, ::-1]

                                    output_images = [output_image]
                                st2 = time.time()

                                print("rough time estimate:", st_sum)

                                if args.mean_estimator:
                                    output_image = output_image[:, :, :, ::-1]
                                #print("output_image swap axis")
                                if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                    output_buffer /= args.estimator_samples
                                    output_image[:] = output_buffer[:]
                                
                            st2 = time.time()

                            loss_val = l2_loss_val
                            #print("loss", loss_val, l2_loss_val * 255.0 * 255.0)
                            all_test[i] = loss_val
                            all_l2[i] = l2_loss_val
                            all_grad[i] = grad_loss_val
                            all_perceptual[i] = perceptual_loss_val

                            if output_image.shape[3] == 3:
                                output_image=np.clip(output_image,0.0,1.0)
                                output_image *= 255.0
                                cv2.imwrite("%s/%06d.png"%(debug_dir, i+1),np.uint8(output_image[0,:,:,:]))
                            else:
                                raise

                            python_time[i] = st_sum


                        if args.repeat_timing > 1:
                            time_stats = time_stats[nburns:]
                            numpy.save(os.path.join(debug_dir, 'time_stats.npy'), time_stats)
                            open('%s/time_stats.txt' % debug_dir, 'w').write('%f, %f, %f, %f' % (np.median(time_stats), np.percentile(time_stats, 25), np.percentile(time_stats, 75), np.std(time_stats)))

                        open("%s/all_loss.txt"%debug_dir, 'w').write("%f, %f"%(np.mean(all_l2), np.mean(all_grad)))
                        numpy.save(os.path.join(debug_dir, 'python_time.npy'), python_time)
                        
                        open("%s/all_time.txt"%debug_dir, 'w').write("%f"%(np.median(python_time)))

                        print("all times saved")

                    test_dirname = debug_dir

                    target=open(os.path.join(test_dirname, 'score.txt'),'w')
                    target.write("%f"%np.mean(all_test[np.where(all_test)]))
                    target.close()
                    target=open(os.path.join(test_dirname, 'vgg.txt'),'w')
                    target.write("%f"%np.mean(all_perceptual[np.where(all_perceptual)]))
                    target.close()
                    target=open(os.path.join(test_dirname, 'vgg_same_scale.txt'),'w')
                    target.write("%f"%np.mean(all_perceptual[np.where(all_perceptual)]))
                    target.close()
                    if all_test.shape[0] == 30:
                        score_close = np.mean(all_test[:5])
                        score_far = np.mean(all_test[5:10])
                        score_middle = np.mean(all_test[10:])
                        target=open(os.path.join(test_dirname, 'score_breakdown.txt'),'w')
                        target.write("%f, %f, %f"%(score_close, score_far, score_middle))
                        target.close()
                        perceptual_close = np.mean(all_perceptual[:5])
                        perceptual_far = np.mean(all_perceptual[5:10])
                        perceptual_middle = np.mean(all_perceptual[10:])
                        target=open(os.path.join(test_dirname, 'vgg_breakdown.txt'),'w')
                        target.write("%f, %f, %f"%(perceptual_close, perceptual_far, perceptual_middle))
                        target.close()
                        target=open(os.path.join(test_dirname, 'vgg_breakdown_same_scale.txt'),'w')
                        target.write("%f, %f, %f"%(perceptual_close, perceptual_far, perceptual_middle))
                        target.close()

                    if args.test_training:
                        grounddir = os.path.join(args.dataroot, 'train_img')
                    else:
                        grounddir = os.path.join(args.dataroot, 'test_img')

            print('time difference with last', time.time() - T0)
            T0 = time.time()

            sess.close()
    
    if orig_args.collect_validate_loss:
        # need to aggregate validation across multiple shaders
        
        all_shader_validations = []
        
        for shader_ind in range(len(all_shaders)):
            shader_name = all_shaders[shader_ind][0]
            all_vals = np.load(os.path.join(orig_args.name, '%s_validation.npy' % shader_name))
            all_shader_validations.append(all_vals)
            
        all_shader_validations = np.array(all_shader_validations)
        
        # per shader min error across all epochs
        min_err = np.min(all_shader_validations, 1)
        min_err = np.expand_dims(min_err, 1)
        
        normalized_validations = all_shader_validations / min_err
        accumulated_validations = np.sum(all_shader_validations, 0)
        
        min_idx = np.argsort(accumulated_validations[:, 1])
        min_epoch = all_shader_validations[0, min_idx[0], 0]

        val_dir = os.path.join(args.name, 'best_val')
        if os.path.isdir(val_dir):
            shutil.rmtree(val_dir)
        
        shutil.copytree(os.path.join(args.name, '%04d' % int(min_epoch)), val_dir)

        open(os.path.join(args.name, 'avg_best_val_epoch.txt'), 'w').write(str(int(min_epoch)))

if __name__ == '__main__':
    main()
