import sys; sys.path += ['../']
import model

import warnings
import skimage

import argparse_util
import render_util

import numpy as np
import numpy

import tensorflow as tf

import skimage.io
import skimage.transform

import os

import importlib
import importlib.util
import subprocess

from tensorflow.python.client import timeline

import time

import json

def main():
    parser = argparse_util.ArgumentParser(description='global_opt_tf_parser')
    parser.add_argument('--ndims', dest='ndims', type=int, default=3, help='dummy')
    parser.add_argument('--check_g', dest='check_g', type=int, default=1, help='dummy')
    parser.add_argument('--samples', dest='samples', type=int, default=1, help='number of samples per pixel')
    parser.add_argument('--error', dest='error', type=int, default=1, help='dummy')
    parser.add_argument('--ground', dest='ground', type=int, default=1, help='dummy')
    parser.add_argument('--print_rho', dest='print_rho', type=int, default=1, help='dummy')
    parser.add_argument('--skip_save', dest='skip_save', type=int, default=0, help='if 1, skip saving contents')
    parser.add_argument('--load_adjust_var', dest='load_adjust_var', default='', help='dummy')
    parser.add_argument('--render', dest='render', default='480,320', help='render size')
    parser.add_argument('--render_sigma', dest='render_sigma', default='0.5,0.5,0', help='specifies sigma used for rendering')
    parser.add_argument('--g_samples', dest='g_samples', type=int, default=0, help='dummy')
    parser.add_argument('--outdir', dest='outdir', default='', help='dirctory to store output files')
    parser.add_argument('--min_time', dest='min_time', type=float, default=0.0, help='dummy')
    parser.add_argument('--max_time', dest='max_time', type=float, default=0.0, help='dummy')
    parser.add_argument('--allow_adjust_var_mismatch_error', dest='allow_adjust_var_mismatch_error', type=int, default=0, help='dummy')
    parser.add_argument('--shader_only', dest='shader_only', type=int, default=1, help='dummy')
    parser.add_argument('--geometry_name', dest='geometry_name', default='plane', help='specify geometry name')
    parser.add_argument('--nfeatures', dest='nfeatures', type=int, default=7, help='dummy')
    parser.add_argument('--camera_path', dest='camera_path', type=int, default=0, help='dummy')
    parser.add_argument('--gname', dest='gname', default='ground', help='name for g output')
    parser.add_argument('--is_gt', dest='is_gt', type=int, default=1, help='dummy')
    parser.add_argument('--our_id', dest='our_id', default='', help='id used to find unique compiler_problem and npy files')
    parser.add_argument('--zero_samples', dest='zero_samples', action='store_true', help='if specified, use all zero samples instead of random')
    parser.add_argument('--camera_pos_file', dest='camera_pos_file', default='', help='dummy')
    parser.add_argument('--generate_timeline', dest='generate_timeline', action='store_true', help='if specified, time the inference')
    parser.add_argument('--log_t_ray', dest='log_t_ray', action='store_true', help='if specified, log t_ray value to g_intermediates')
    parser.add_argument('--fov', dest='fov', default='regular', help='specifies the field of view of the camera')
    parser.add_argument('--collect_loop_statistic', dest='collect_loop_statistic', action='store_true', help='if true, vec_output also contains loop statistic, store them')
    parser.add_argument('--collect_loop_statistic_and_features', dest='collect_loop_statistic_and_features', action='store_true', help='if true, collect loop statstics and log_intermediates')
    parser.add_argument('--log_manual_features', dest='log_manual_features', action='store_true', help='if specified, log manual features only')
    parser.add_argument('--efficient_trace', dest='efficient_trace', action='store_true', help='if specified, do not log duplicate trace')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='specifies batch size when rendering with nsamples > 1')
    parser.add_argument('--tile_only', dest='tile_only', action='store_true', help='if specified, render only tiles (part of an entire image) according to tile_start')
    parser.add_argument('--rescale_tile_start', dest='rescale_tile_start', type=float, default=1, help='specifies whether we are rendering to a reslution different to when tile_start is created')
    parser.add_argument('--use_texture_maps', dest='use_texture_maps', action='store_true', help='if specified, this program uses texture maps')
    parser.add_argument('--partial_trace', dest='partial_trace', type=float, default=1, help='if less than 1, only record the first 100x percent of the trace wehre x = partial_trace')
    parser.add_argument('--temporal_texture_buffer', dest='temporal_texture_buffer', action='store_true', help='if specified, the output of previous frame becomes texture for next frame')
    parser.add_argument('--no_store_temporal_texture', dest='store_temporal_texture', action='store_false', help='if specified, do not store rendered temporal texture to disk')
    parser.add_argument('--store_temporal_texture', dest='store_temporal_texture', action='store_true', help='if specified, store rendered temporal texture to disk')
    parser.add_argument('--save_downsample_scale', dest='save_downsample_scale', type=int, default=1, help='if larger than 1, use the scale to downsample features before saving to g_intermediates')
    parser.add_argument('--n_boids', dest='n_boids', type=int, default=40, help='number of boids in boids app')
    parser.add_argument('--camera_sigma', dest='camera_sigma', default='', help='sigma used to sample camera position')
    parser.add_argument('--expand_boundary', dest='expand_boundary', type=int, default=0, help='pixel size expanded around boundary')
    parser.add_argument('--collect_feature_mean_only', dest='collect_feature_mean_only', action='store_true', help='if true, do not write g_intermediates to disk, collect its mean value instead')
    parser.add_argument('--feature_normalize_dir', dest='feature_normalize_dir', default='', help='directory that stores feature_bias and feature_scale for the current dataset')
    parser.add_argument('--feature_start_ind', dest='feature_start_ind', type=int, default=-1, help='index of feature that we start computing (used if the feature length is too large')
    parser.add_argument('--feature_end_ind', dest='feature_end_ind', type=int, default=-1, help='index of feature that we end computing (used if the feature length is too large')
    parser.add_argument('--reference_dir', dest='reference_dir', default='', help='directory for reference images')
    args = parser.parse_args()

    if args.samples == 0:
        args.samples = 1

    assert args.samples % args.batch_size == 0

    render_size = args.render.split(',')
    render_size = [int(x) for x in render_size]
    #print("render_size", render_size)

    render_sigma = args.render_sigma.split(',')
    render_sigma = [float(x) for x in render_sigma]
    print("render_sigma:", render_sigma)
    
    if args.camera_sigma != '':
        camera_sigma = np.array([float(val) for val in args.camera_sigma.split(',')])
    else:
        camera_sigma = None

    camera_pos_file, render_t_file, render_index_file, compiler_file, _, tile_start_file, texture_file = render_util.get_filenames(args.our_id)

    compiler_problem_full_name = os.path.abspath(compiler_file)
    spec = importlib.util.spec_from_file_location("module_name", compiler_problem_full_name)
    compiler_problem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler_problem)

    if args.geometry_name not in ['boids', 'boids_coarse']:
        camera_pos = np.load(camera_pos_file)
    if args.geometry_name != 'boids':
        render_t = np.load(render_t_file)
        

    render_index = np.load(render_index_file)

    if args.tile_only:
        tile_start = np.load(tile_start_file)
        if args.rescale_tile_start != 1:
            tile_start /= args.rescale_tile_start
        h_start = tf.placeholder(model.dtype, shape=1)
        w_start = tf.placeholder(model.dtype, shape=1)
        h_offset = render_size[0]
        w_offset = render_size[1]
        #camera_render_size = (960, 640)
        camera_render_size = (render_size[0] * 3, render_size[0] * 2)
    else:
        h_start = tf.constant([0.0])
        w_start = tf.constant([0.0])
        h_offset = render_size[1]
        w_offset = render_size[0]
        camera_render_size = render_size
        
    if args.expand_boundary > 0:
        real_h_offset = h_offset + 2 * args.expand_boundary
        real_w_offset = w_offset + 2 * args.expand_boundary
        real_h_start = h_start - args.expand_boundary
        real_w_start = w_start - args.expand_boundary
    else:
        real_h_offset = h_offset
        real_w_offset = w_offset
        real_h_start = h_start
        real_w_start = w_start

    if args.temporal_texture_buffer:
        texture_map_size = compiler_problem.vec_output_len
        texture_maps = []
        if args.geometry_name in ['boids', 'boids_coarse']:
            texture_maps = tf.placeholder(tf.float32, (args.batch_size, args.n_boids, texture_map_size))
        else:
            single_texture_size = [render_size[1], render_size[0]]
            for i in range(texture_map_size):
                texture_maps.append(tf.placeholder(tf.float32, single_texture_size))
    elif args.use_texture_maps:
        combined_texture_maps = np.load(texture_file)
        texture_maps = []
        for i in range(combined_texture_maps.shape[0]):
            texture_maps.append(tf.convert_to_tensor(combined_texture_maps[i], dtype=model.dtype))
        texture_map_size = len(texture_maps)
    else:
        texture_maps = []
        texture_map_size = 0


    extra_args = [None]


    with tf.variable_scope("shader"):
        if args.geometry_name not in ['boids', 'boids_coarse']:
            if args.geometry_name != 'texture_approximate_10f':
                camera_pos_pl = tf.placeholder(model.dtype, shape=[6, 1])
            else:
                camera_pos_pl = tf.placeholder(model.dtype, shape=[33, 1])
            shader_time = tf.placeholder(model.dtype, shape=1)

        
            feed_dict = {camera_pos_pl: np.expand_dims(camera_pos[0, :], 1), shader_time: render_t[0:1]}
            
            if camera_sigma is not None:
                camera_sample = tf.random.normal(tf.shape(camera_pos_pl), dtype=model.dtype)
                camera_sample *= np.expand_dims(camera_sigma, 1)
                camera_pos_sampled = camera_pos_pl + camera_sample
            else:
                camera_pos_sampled = camera_pos_pl
            
        else:
            feed_dict = {}
            camera_pos_pl = None
            shader_time = None
            if args.geometry_name == 'boids_coarse':
                shader_time = tf.placeholder(model.dtype, shape=1)
                feed_dict[shader_time] = render_t[0:1]
            camera_pos_sampled = camera_pos_pl
            
        if args.tile_only:
            feed_dict[h_start] = tile_start[0:1, 0]
            feed_dict[w_start] = tile_start[0:1, 1]
        update_texture = False
        if args.temporal_texture_buffer:
            if args.use_texture_maps:
                combined_texture_maps = np.load(texture_file)
                if args.geometry_name in ['boids', 'boids_coarse']:
                    feed_dict[texture_maps] = np.expand_dims(combined_texture_maps, 0)
                else:
                    if combined_texture_maps.dtype in [float, int]:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if not (combined_texture_maps.shape[1] == render_size[1] and combined_texture_maps.shape[2] == render_size[0]):
                                combined_texture_maps = skimage.transform.resize(combined_texture_maps, (combined_texture_maps.shape[0], render_size[1], render_size[0]))
                        for k in range(len(texture_maps)):
                            feed_dict[texture_maps[k]] = combined_texture_maps[k]
                    else:
                        update_texture_names = combined_texture_maps.tolist()
                        current_texture_maps = np.transpose(np.load(update_texture_names[0]), (2, 0, 1))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if not (current_texture_maps.shape[1] == render_size[1] and current_texture_maps.shape[2] == render_size[0]):
                                current_texture_maps = skimage.transform.resize(current_texture_maps, (current_texture_maps.shape[0], render_size[1], render_size[0]))
                        for k in range(len(texture_maps)):
                            feed_dict[texture_maps[k]] = current_texture_maps[k]
            else:
                for k in range(len(texture_maps)):
                    feed_dict[texture_maps[k]] = np.zeros([render_size[1], render_size[0]])
                
        features, vec_output, manual_features = model.get_render(camera_pos_sampled, shader_time, nsamples=args.batch_size, samples=None, shader_name='compiler_problem', return_vec_output=True, render_size=camera_render_size, compiler_module=compiler_problem, geometry=args.geometry_name, zero_samples=args.zero_samples, debug=[feed_dict], extra_args=extra_args, fov=args.fov, manual_features_only=args.log_manual_features, h_start=real_h_start, h_offset=real_h_offset, w_start=real_w_start, w_offset=real_w_offset, render_sigma=render_sigma, texture_maps=texture_maps, n_boids=args.n_boids)
        
        raw_features = features
        
    if args.partial_trace >= 1:
        # TODO: temporal texture buffer and mean_var loop statistic is conflicting, both are using positions in vec_output
        # but because in practice mean_var is inefficient and we won't choose this strategy, will prioritize temporal texture buffer
        #if len(vec_output) > 3 and args.collect_loop_statistic_and_features:
        #    loop_statistic = vec_output[3:]
        #    vec_output = vec_output[:3]
        #    if args.collect_loop_statistic_and_features:
        #        features = features + loop_statistic
        #else:
        #    loop_statistic = None
        # deprecate above code since loop_statistic is never collected now
        loop_statistic = None

        # workaround if for some feature sparsification setup, RGB channels are not logged
        features = features + vec_output + manual_features
        
        if args.temporal_texture_buffer:
            if isinstance(texture_maps, list):
                for k in range(len(texture_maps)):
                    features.append(tf.expand_dims(texture_maps[k], 0))
        
    else:
        
        len_raw_features = compiler_problem.f_log_intermediate_len
        features = features[:int(len_raw_features * args.partial_trace)] + features[len_raw_features:]

    if args.geometry_name in ['boids', 'boids_coarse']:
        out_img = tf.stack(vec_output, -1)
    else:
        out_img = tf.reduce_sum(tf.stack(vec_output, axis=3), axis=0)
        
    valid_inds = []
    log_intermediate = False
    base_features = 7 if args.geometry_name not in ['none', 'texture', 'texture_approximate_10f'] else 2

        
    if (len(features) > base_features + len(vec_output) + texture_map_size + len(manual_features)) or (args.log_manual_features and len(manual_features) > 0):
        if args.log_manual_features:
            features = manual_features
        all_features = []
        for i in range(len(features)):
            feature = features[i]
            if isinstance(feature, (float, int, numpy.bool_)):
                continue
            if args.collect_loop_statistic_and_features or args.efficient_trace:
                if feature in all_features:
                    continue
            valid_inds.append(i)
            all_features.append(feature)

        numpy.save(os.path.join(args.outdir, 'valid_inds.npy'), valid_inds)

        valid_features_len = len(valid_inds)
        print(valid_features_len)
        for i in range(len(all_features)):
            try:
                if all_features[i].dtype != model.dtype:
                    all_features[i] = tf.cast(all_features[i], model.dtype)
            except:
                raise
                
        
                
        if args.collect_feature_mean_only:
            
            feature_bias = np.load(os.path.join(args.feature_normalize_dir, 'feature_bias_20_80.npy'))
            feature_scale = np.load(os.path.join(args.feature_normalize_dir, 'feature_scale_20_80.npy'))
            
            color_inds = []
            for vec in vec_output:
                color_inds.append(all_features.index(vec))
                
            feature_bias[color_inds] = 0.0
            feature_scale[color_inds] = 1.0
                         
            np.save(os.path.join(args.outdir, 'color_inds.npy'), color_inds)
                
            if args.feature_end_ind >= 0:
                all_features = all_features[:args.feature_end_ind+1]
                feature_bias = feature_bias[:args.feature_end_ind+1]
                feature_scale = feature_scale[:args.feature_end_ind+1]
            if args.feature_start_ind > 0:
                all_features = all_features[args.feature_start_ind:]
                feature_bias = feature_bias[args.feature_start_ind:]
                feature_scale = feature_scale[args.feature_start_ind:]
                            
        
        all_features = tf.stack(all_features, axis=-1)
        
        # this might introduce a bug when collecting feature mean in venice, but let's wait till then to solve the bug
        if not isinstance(texture_maps, list):
            all_features = tf.concat((all_features, texture_maps), -1)
            valid_features_len += int(texture_maps.shape[-1])
        log_intermediate = True
        
        if args.collect_feature_mean_only:
            
            all_features += feature_bias
            all_features *= feature_scale
            
            all_features -= 0.5
            all_features *= 2
            
            all_features = tf.clip_by_value(all_features, -2, 2)

            out_img = tf.clip_by_value(out_img, 0, 1)
            out_img = tf.where(tf.is_nan(out_img), tf.zeros_like(out_img), out_img)
            
            reference_pl = tf.placeholder(model.dtype, [None, None, 3])
            
            reference_img_files = [os.path.join(args.reference_dir, file) for file in os.listdir(args.reference_dir) if file.endswith('.png')]
            reference_img_files = sorted(reference_img_files)
            
            assert len(reference_img_files) == render_index.shape[0]
            
            shader_error = tf.reduce_mean((out_img - reference_pl) ** 2, 2)
            shader_error = tf.expand_dims(tf.expand_dims(shader_error, 0), 3)
            
            if args.feature_end_ind < 0:
                # only log first and second order statistic for shader_error if it's the last piece of the entire trace
                all_features = tf.concat((all_features, shader_error), 3)
            
            all_features = tf.where(tf.is_nan(all_features), tf.zeros_like(all_features), all_features)
            
            all_features_first_order = tf.reduce_mean(all_features, (0, 1, 2))
            all_features_second_order = tf.reduce_mean(all_features ** 2, (0, 1, 2))
            all_features_cross = tf.reduce_mean(all_features * shader_error, (0, 1, 2))
            
            all_features = tf.stack((all_features_first_order, all_features_second_order, all_features_cross), 1)
            
            # +1 is for error channel
            feature_sum_val = np.zeros((int(all_features.shape[0]), 3))
            feature_sample_count = 0

    if args.collect_loop_statistic and loop_statistic is not None:
        for i in range(len(loop_statistic)):
            if isinstance(loop_statistic[i], (int, float)):
                loop_statistic[i] = loop_statistic[i] * tf.ones_like(vec_output[0])
            if loop_statistic[i] == 0.0:
                loop_statistic[i] = tf.zeros_like(vec_output[0])
        loop_statistic = tf.squeeze(tf.stack(loop_statistic, axis=3))

    

    sess = tf.Session()
    avg_t = numpy.empty(render_index.shape[0])
    if args.generate_timeline:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None

    all_time = 0
    
    feed_dict = {}
    
    xv, yv = numpy.meshgrid(numpy.arange(640), numpy.arange(960), indexing='ij')

    update_texture = False
    if args.temporal_texture_buffer:
        if args.use_texture_maps:
            combined_texture_maps = np.load(texture_file)
            if combined_texture_maps.dtype in [float, int]:
                if isinstance(texture_maps, list):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if not (combined_texture_maps.shape[1] == render_size[1] and combined_texture_maps.shape[2] == render_size[0]):
                            combined_texture_maps = skimage.transform.resize(combined_texture_maps, (combined_texture_maps.shape[0], render_size[1], render_size[0]))
                    for k in range(len(texture_maps)):
                        feed_dict[texture_maps[k]] = combined_texture_maps[k]
                else:
                    # TODO: currently in tf_parser only supports batch_size=1 for boids
                    if args.geometry_name.startswith('boids') and len(combined_texture_maps.shape) > 2:
                        update_texture = True
                        feed_dict[texture_maps] = combined_texture_maps[0]
                        update_texture_names = combined_texture_maps
                    else:
                        feed_dict[texture_maps] = np.expand_dims(combined_texture_maps, 0)
            else:
                update_texture = True
                update_texture_names = combined_texture_maps.tolist()
        else:
            for k in range(len(texture_maps)):
                feed_dict[texture_maps[k]] = np.zeros([render_size[1], render_size[0]])
                #feed_dict[texture_maps[i]] = xv
        #camera_pos = np.zeros([1800, 6])
        #render_t = numpy.linspace(0, 60, 1800)
        #render_index = numpy.arange(1800).astype('i')
        
    if args.geometry_name in ['boids', 'boids_coarse']:
        all_outputs = np.empty([render_index.shape[0], args.n_boids, len(vec_output)])
        if log_intermediate:
            all_intermediate = np.empty([render_index.shape[0], args.n_boids, valid_features_len])
                    
    
            
    for i in range(render_index.shape[0]):
        if args.geometry_name not in ['boids', 'boids_coarse']:
            feed_dict[camera_pos_pl] = camera_pos[i:i+1, :].transpose()
        if args.geometry_name != 'boids':
            feed_dict[shader_time] =  render_t[i:i+1]
        
        if args.tile_only:
            feed_dict[h_start] = tile_start[i:i+1, 0]
            feed_dict[w_start] = tile_start[i:i+1, 1]
        
        if update_texture:
            if not args.geometry_name.startswith('boids'):
                current_texture_maps = np.transpose(np.load(update_texture_names[i]), (2, 0, 1))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if not (current_texture_maps.shape[1] == render_size[1] and current_texture_maps.shape[2] == render_size[0]):
                        current_texture_maps = skimage.transform.resize(current_texture_maps, (current_texture_maps.shape[0], render_size[1], render_size[0]))
                for k in range(len(texture_maps)):
                    feed_dict[texture_maps[k]] = current_texture_maps[k]
            else:
                current_texture_maps = update_texture_names[i]
                feed_dict[texture_maps] = np.expand_dims(current_texture_maps, 0)
            
        if args.collect_feature_mean_only:
            feed_dict[reference_pl] = skimage.img_as_float(skimage.io.imread(reference_img_files[i]))
                
        if not args.skip_save:
            if args.geometry_name not in ['boids', 'boids_coarse']:
                if log_intermediate and not args.collect_feature_mean_only:
                    final_features = np.empty([real_h_offset, real_w_offset, args.samples * valid_features_len], dtype=numpy.float32)
                final_img = np.zeros([real_h_offset, real_w_offset, len(vec_output)])
            else:
                if log_intermediate and not args.collect_feature_mean_only:
                    final_features = np.empty([args.batch_size, args.n_boids, valid_features_len], dtype=numpy.float32)
                final_img = np.zeros([args.batch_size, args.n_boids, len(vec_output)])
            samples_count = args.samples
            for n in range(int(args.samples / args.batch_size)):

                if log_intermediate:
                    T1 = time.time()
                    val_features, val_img = sess.run([all_features, out_img], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    T2 = time.time()
                    assert args.samples == 1
                    
                    if not args.collect_feature_mean_only:
                        if args.geometry_name.startswith('boids'):
                            final_features = val_features
                        else:
                            final_features[:, :, n*valid_features_len:(n+1)*valid_features_len] = val_features[0, :, :, :]
                else:
                    T1 = time.time()
                    val_img = sess.run(out_img, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    T2 = time.time()
                
                all_time += T2 - T1
                print("time used:", T2 - T1)
                #if numpy.sum(numpy.isnan(val_img)) > 0:
                if False:
                    samples_count -= args.batch_size
                else:
                    final_img += val_img
                print(i, n, args.samples - samples_count)
                
            #with warnings.catch_warnings():
            #    warnings.simplefilter("ignore")
            #    for c in range(4):
            #        skimage.io.imsave(os.path.join(args.outdir, '%s_%d_%05d.png'%(args.gname, c, render_index[i])), numpy.clip(final_img[:, :, c+3], 0.0, 1.0))
                
            if log_intermediate:
                if args.geometry_name.startswith('boids'):
                    all_intermediate[i] = final_features
                else:
                    
                    if args.collect_feature_mean_only:
                        feature_sum_val += val_features
                        feature_sample_count += 1
                    else:
                        if args.save_downsample_scale > 1:
                            final_features = final_features[::args.save_downsample_scale, ::args.save_downsample_scale, :]
                        final_features = numpy.moveaxis(final_features, [0, 1, 2], [1, 2, 0])

                        numpy.save(os.path.join(args.outdir, 'g_intermediates%05d.npy'%(render_index[i])), final_features)
            
            final_img /= samples_count
            if args.geometry_name not in ['boids', 'boids_coarse']:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    skimage.io.imsave(os.path.join(args.outdir, '%s%05d.png'%(args.gname, render_index[i])), numpy.clip(final_img[:, :, :3], 0.0, 1.0))
                if args.temporal_texture_buffer and args.store_temporal_texture:
                    np.save(os.path.join(args.outdir, '%s%05d.npy'%(args.gname, render_index[i])), final_img)
            else:
                
                all_outputs[i] = final_img[:]
                
            if (not update_texture) and args.temporal_texture_buffer:
                if isinstance(texture_maps, list):
                    for k in range(len(texture_maps)):
                        feed_dict[texture_maps[k]] = final_img[:, :, k]
                else:
                    feed_dict[texture_maps] = final_img
            if args.generate_timeline:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(os.path.join(args.outdir, 'nn_%s%05d.json'%(args.gname, render_index[i])), 'w') as f:
                    f.write(chrome_trace)
        if args.geometry_name in ['hyperboloid1', 'paraboloid']:
            t_ray = extra_args[0]
        elif args.geometry_name == 'none' and log_intermediate and len(raw_features) == 3:
            t_ray = raw_features[0]
        else:
            t_ray = None
        if t_ray is not None:
            t_val = sess.run(t_ray, feed_dict=feed_dict)
            t_val[numpy.isnan(t_val)] = 0.0
            t_val[t_val < 0] = 0.0
            valid_t = t_val > 0
            avg_t[i] = numpy.sum(valid_t * t_val) / numpy.sum(valid_t)
            if args.log_t_ray:
                numpy.save(os.path.join(args.outdir, 'g_intermediates%05d.npy'%(render_index[i])), t_val)

        if args.collect_loop_statistic and loop_statistic is not None:
            loop_statistic_val = sess.run(loop_statistic, feed_dict=feed_dict)
            numpy.save(os.path.join(args.outdir, 'loop_statistic%05d.npy'%(render_index[i])), loop_statistic_val)
            
    if args.geometry_name in ['boids', 'boids_coarse']:
        np.save(os.path.join(args.outdir, '%s.npy' % args.gname), all_outputs)
        
    if log_intermediate and args.geometry_name.startswith('boids'):
        np.save(os.path.join(args.outdir, 'g_intermediates.npy'), all_intermediate)
        
    if args.collect_feature_mean_only:
        feature_mean = feature_sum_val / feature_sample_count
        if args.feature_start_ind == -1 and args.feature_end_ind == -1:
            np.save(os.path.join(args.outdir, 'feature_mean.npy'), feature_mean)
        else:
            np.save(os.path.join(args.outdir, 'feature_mean_%d_%d.npy' % (args.feature_start_ind, args.feature_end_ind)), feature_mean)

    print("avg time per frame: %f" % (all_time / render_index.shape[0]))

    if t_ray is not None:
        numpy.save(os.path.join(args.outdir, 'avg_t.npy'), avg_t)

def sort_t_val(arr):
    arr_sorted = numpy.sort(arr)
    ind = numpy.argsort(arr)
    close = ind[:5]
    far = ind[-5:]
    t_close = arr_sorted[close[-1]]
    t_far = arr_sorted[far[0]]

    for i in range(close[-1]+1, far[0]):
        if arr_sorted[i] >= 1.5 * t_close:
            start_ind = i
            break

    for i in range(far[0]-1, close[-1], -1):
        if arr_sorted[i] <= t_far / 1.5:
            end_ind = i
            break

    assert end_ind - start_ind >= 220
    random_ind = numpy.random.permutation(ind[start_ind:end_ind+1])

    train = random_ind[:200]
    middle = random_ind[-20:]

if __name__ == '__main__':
    main()
