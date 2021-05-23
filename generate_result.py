import os
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import colorsys
import skimage
import skimage.io
import shutil
import sys
from matplotlib.patches import BoxStyle
import skimage.exposure
import skimage.draw
import copy

# new inference time computed on node000 with 12 CPUs

hue_shaders = {
    'bricks': [0.1,1,1],
    'mandelbrot': [0.2, 1, 1],
    'mandelbulb': [0.3, 1, 1],
    'marble': [0.0, 0.6, 1],
    'oceanic': [0.5, 1, 1],
    'gear': [0.6, 0.9, 1],
    'trippy': [0.8, 0.9, 1],
    'venice': [0.9, 0.8, 1],
    'boids': [0.7, 0.8, 1]
}

model_parent_dir = ''
dataset_parent_dir = ''

tex_prefix = 'new_submission/'

score_pl = 0.0
img_pl = np.zeros((640, 960, 3))
allow_missing_MSAA = False
allow_missing_temporal = False
allow_missing_simulation = False
allow_missing_timing = True
use_gamma_corrected = False

app_shader_dir_200 = {
'denoising': {
    'bricks': {'dir': ['1x_1sample_bricks_with_bg_all/test',
                       '1x_1sample_bricks_with_bg_aux_largest_capacity/test',
                       '1x_1sample_bricks_with_bg_all/mean1_test'],
               'base_runtime_dir': '1x_1sample_bricks_with_bg_all/mean1_test',
               'gamma_correction': 0.775,
               'img_idx': 9,
               'img_zoom_bbox': [400, 480, -140, -80],
               'gt_dir': 'datas_bricks_with_bg/test_img',
               'msaa_sample': 1,
               'print': 'Bricks',
               'RGBx_ch': 105,
              },
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_with_bg_stratified_subsample_2/test',
                           '1x_1sample_mandelbrot_with_bg_aux_largest_capacity/test',
                           '1x_1sample_mandelbrot_with_bg_all/mean2_test'],
                   'base_runtime_dir': '1x_1sample_mandelbrot_with_bg_all/mean1_test',
                   'img_idx': 30,
                   'img_zoom_bbox': [250, 250+80, 570, 570+60],
                   'gt_dir': 'datas_mandelbrot_with_bg/test_img',
                   'msaa_sample': 2,
                   'gamma_correction': 0.7,
                   'print': 'Mandelbrot',
                   'RGBx_ch': 389,
                   'fov': 'small',
                  },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_with_bg_all/test',
                           '1x_1sample_mandelbulb_with_bg_aux_largest_capacity/test',
                           '1x_1sample_mandelbulb_with_bg_all/mean2_test'],
                   'base_runtime_dir': '1x_1sample_mandelbulb_with_bg_all/mean1_test',
                   'img_idx': 20,
                   'img_zoom_bbox': [250, 250+80, 325, 325+60],
                   'gt_dir': 'datas_mandelbulb_with_bg/test_img',
                   'msaa_sample': 2,
                   'print': 'Mandelbulb',
                   'RGBx_ch': 232,
                   'fov': 'small_seperable',
                   'geometry': 'none'
                  },
    'gear': {'dir': ['1x_1sample_primitives_all/test',
                           '1x_1sample_primitives_aux_largest_capacity/test',
                           '1x_1sample_primitives_all/mean1_test'],
             'base_runtime_dir': '1x_1sample_primitives_all/mean1_test',
                    'other_view': [
                        # gt frame
                        'datas_primitives_correct_test_range/test_extra_img/render_gt_path2_frame_150.png',
                        # ours extra frame, can be rendered using following command
                        # python model.py --name {MODEL_ROOT}/1x_1sample_primitives_all --dataroot {DATA_ROOT}/datas_primitives_correct_test_range/ --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --shader_name primitives_wheel_only --geometry none --fov small --data_from_gpu --identity_initialize --no_identity_output_layer --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --epoch 400 --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping --read_from_best_validation --render_only --render_camera_pos {DATA_ROOT}/extra_render_pos/primitives_camera_pos_path2.npy --render_t {DATA_ROOT}/extra_render_pos/primitives_render_t_path2.npy --render_dirname test_extra --render_fix_spatial_sample
                        '1x_1sample_primitives_all/test_extra/000150.png',
                        # RGBx extra frame, can be rendered using following command
                        # python model.py --name {MODEL_ROOT}/1x_1sample_primitives_aux_largest_capacity --dataroot {DATA_ROOT}/datas_primitives_correct_test_range/ --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --shader_name primitives_wheel_only --geometry none --fov small --data_from_gpu --identity_initialize --no_identity_output_layer --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --epoch 400 --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping --read_from_best_validation --render_only --render_camera_pos {DATA_ROOT}/extra_render_pos/primitives_camera_pos_path2.npy --render_t {DATA_ROOT}/extra_render_pos/primitives_render_t_path2.npy --render_dirname test_extra --render_fix_spatial_sample --manual_features_only --feature_reduction_ch 245
                        '1x_1sample_primitives_aux_largest_capacity/test_extra/000150.png',
                        # supersample extra frame, can be rendred using following command
                        # python model.py --name {MODEL_ROOT}/1x_1sample_primitives_all --dataroot {DATA_ROOT}/datas_primitives_correct_test_range/ --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --shader_name primitives_wheel_only --geometry none --fov small --data_from_gpu --identity_initialize --no_identity_output_layer --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --epoch 400 --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping --read_from_best_validation --render_only --render_camera_pos {DATA_ROOT}/extra_render_pos/primitives_camera_pos_path2.npy --render_t {DATA_ROOT}/extra_render_pos/primitives_render_t_path2.npy --render_dirname mean1_test_extra --render_fix_spatial_sample --mean_estimator --estimator_samples 1
                        '1x_1sample_primitives_all/mean1_test_extra/000150.png'],
                   'img_zoom_bbox': [250, 250+80, 460, 460+60],
                   'gt_dir': 'datas_primitives_correct_test_range/test_img',
                   'msaa_sample': 1,
                   'print': 'Gear',
                   'RGBx_ch': 245,
                   'fov': 'small',
                   'geometry': 'none'
             
                  },
    'trippy': {'dir': ['1x_1sample_trippy_stratified_subsample_8/test',
               '1x_1sample_trippy_aux_largest_capacity/test',
               '1x_1sample_trippy_subsample_2/mean4_test'],
               'base_runtime_dir': '1x_1sample_trippy_subsample_2/mean1_test',
               'img_idx': 30,
               'img_zoom_bbox': [550, 550+80, 65, 65+60],
               'gt_dir': 'datas_trippy_new_extrapolation_subsample_2/test_img',
               'msaa_sample': 4,
               'print': 'Trippy Heart',
               'RGBx_ch': 744,
               'fov': 'small',
               'every_nth': 2
              },
    'oceanic': {'dir': ['1x_1sample_oceanic_all/test',
                '1x_1sample_oceanic_aux_largest_capacity/test',
                '1x_1sample_oceanic_all/mean1_test'],
                'base_runtime_dir': '1x_1sample_oceanic_all/mean1_test',
                'img_idx': 11,
                'img_zoom_bbox': [475, 475+120, -120, -30],
                'gt_dir': 'datas_oceanic/test_img',
                'msaa_sample': 1,
                'print': 'Oceanic',
                'RGBx_ch': 211,
                'fov': 'small',
                'geometry': 'none'
               },
    'venice': {'dir': ['1x_1sample_venice_stratified_subsample_3/test',
               '1x_1sample_venice_aux_largest_capacity/test',
               '1x_1sample_venice_all/mean1_test'],
               'base_runtime_dir': '1x_1sample_venice_all/mean1_test',
               'img_idx': 3,
               'img_zoom_bbox': [170, 170+80, 40, 40+60],
               'gt_dir': 'datas_venice_new_extrapolation/test_img',
               'msaa_sample': 1,
               'print': 'Venice',
               'RGBx_ch': 517,
               'fov': 'small',
               'geometry': 'none'
              },
    },
'simplified': {
    'bricks': {'dir': ['1x_1sample_bricks_simplified_with_bg_all/test',
               '1x_1sample_bricks_simplified_with_bg_aux_largest_capacity/test',
               '1x_1sample_bricks_simplified_with_bg_all/mean1_test'],
               'base_runtime_dir': '1x_1sample_bricks_with_bg_all/mean1_test',
               'img_idx': 15,
               'img_zoom_bbox': [400, 480, 700, 760],
               'gt_dir': 'datas_bricks_simplified_with_bg/test_img',
               'print': 'Bricks',
               'RGBx_ch': 95
              },
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_simplified_with_bg_all/test',
                   '1x_1sample_mandelbrot_simplified_with_bg_aux_largest_capacity/test',
                   '1x_1sample_mandelbrot_simplified_with_bg_all/mean1_test'],
                   'base_runtime_dir': '1x_1sample_mandelbrot_with_bg_all/mean1_test',
                   'img_idx': 30,
                   'gamma_correction': 0.7,
                   'img_zoom_bbox': [260, 340, 750, 810],
                   'gt_dir': 'datas_mandelbrot_simplified_with_bg/test_img',
                   'print': 'Mandelbrot',
                   'fov': 'small',
                   'RGBx_ch': 226
                  },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_with_bg_siimplified_all/test',
                   '1x_1sample_mandelbulb_simplified_with_bg_aux_largest_capacity/test',
                   '1x_1sample_mandelbulb_with_bg_siimplified_all/mean1_test'],
                   'base_runtime_dir': '1x_1sample_mandelbulb_with_bg_all/mean1_test',
                   'img_idx': 12,
                   'img_zoom_bbox': [160, 240, 760, 820],
                   'gt_dir': 'datas_mandelbulb_simplified_with_bg/test_img',
                   'print': 'Mandelbulb',
                   'geometry': 'none',
                   'fov': 'small_seperable',
                   'RGBx_ch': 172
                  },
    'trippy': {'dir': ['1x_1sample_trippy_simplified_stratified_subsample_4/test',
               '1x_1sample_trippy_simplified_aux_largest_capacity/test',
               '1x_1sample_trippy_simplified_all/mean1_test'],
               'base_runtime_dir': '1x_1sample_trippy_subsample_2/mean1_test',
               'other_view': [
                   # gt frame
                   'datas_trippy_new_extrapolation_subsample_2/test_extra_img/render_gt_frame_90.png',
                   # ours extra frame, can be rendered using following command
                   # python model.py --name {MODEL_ROOT}/1x_1sample_trippy_simplified_stratified_subsample_4 --dataroot {DATAROOT}/datas_trippy_simplified_new_extrapolation --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --shader_name trippy_heart_simplified_proxy --data_from_gpu --identity_initialize --no_identity_output_layer --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --epoch 400 --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping --fov small --patch_gan_loss --gan_loss_scale 0.05 --epoch 800 --discrim_train_steps 8 --no_additional_features --ignore_last_n_scale 7 --include_noise_feature --specified_ind {MODEL_ROOT}/1x_1sample_trippy_simplified_stratified_subsample_4/specified_ind.npy --read_from_best_validation --render_only --render_camera_pos {DATA_ROOT}/extra_render_pos/trippy_camera_pos_long.npy --render_t {DATA_ROOT}/extra_render_pos/trippy_render_t_long.npy --render_dirname test_extra --render_fix_spatial_sample
                   '1x_1sample_trippy_simplified_stratified_subsample_4/test_extra/000090.png',
                   # RGBx extra frame, can be rendered using following command
                   # python model.py --name {MODEL_ROOT}/1x_1sample_trippy_simplified_aux_largest_capacity --dataroot {DATAROOT}/datas_trippy_simplified_new_extrapolation --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --shader_name trippy_heart_simplified_proxy --data_from_gpu --identity_initialize --no_identity_output_layer --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --epoch 400 --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping --fov small --patch_gan_loss --gan_loss_scale 0.05 --epoch 800 --discrim_train_steps 8 --read_from_best_validation --render_only --render_camera_pos {DATA_ROOT}/extra_render_pos/trippy_camera_pos_long.npy --render_t {DATA_ROOT}/extra_render_pos/trippy_render_t_long.npy --render_dirname test_extra --render_fix_spatial_sample --manual_features_only --feature_reduction_ch 760
                   '1x_1sample_trippy_simplified_aux_largest_capacity/test_extra/000090.png',
                   # input extra frame, can be rendered using following command
                   # python model.py --name {MODEL_ROOT}/1x_1sample_trippy_simplified_all --dataroot {DATAROOT}/datas_trippy_simplified_new_extrapolation --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --shader_name trippy_heart_simplified_proxy --data_from_gpu --identity_initialize --no_identity_output_layer --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --epoch 400 --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping --fov small --patch_gan_loss --gan_loss_scale 0.05 --epoch 800 --discrim_train_steps 8 --no_additional_features --ignore_last_n_scale 7 --include_noise_feature --read_from_best_validation --render_only --render_camera_pos {DATA_ROOT}/extra_render_pos/trippy_camera_pos_long.npy --render_t {DATA_ROOT}/extra_render_pos/trippy_render_t_long.npy --render_dirname mean1_test_extra --render_fix_spatial_sample --mean_estimator --estimator_samples 1
                   '1x_1sample_trippy_simplified_all/mean1_test_extra/000090.png'],
               'img_zoom_bbox': [50, 210, 80, 200],
               'gt_dir': 'datas_trippy_simplified_new_extrapolation/test_img',
               'print': 'Trippy Heart',
               'fov': 'small',
               'RGBx_ch': 760
              },
    'venice': {'dir': ['1x_1sample_venice_simplified_20_100_stratified_subsample_3/test',
                       '1x_1sample_venice_simplified_20_100_aux_largest_capacity/test',
                       '1x_1sample_venice_simplified_20_100_all/mean1_test'],
               'base_runtime_dir': '1x_1sample_venice_all/mean1_test',
               'img_idx': 18,
               'img_zoom_bbox': [160, 160+120, 150+165-5, 150+165+5+80],
               'gt_dir': 'datas_venice_simplified_20_100_new_extrapolation//test_img',
               'input_time_frag': 72,
               'print': 'Venice',
               'geometry': 'none',
               'fov': 'small',
               'RGBx_ch': 517
              },
    },
'temporal': {
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_temporal_with_bg_non_gan_onconst_stratified_subsample_2/test',
                   '1x_1sample_mandelbrot_temporal_with_bg_non_gan_on_const_aux_largest_capacity/test'],
                   'print': 'Mandelbrot',
                   'gamma_correction': 0.7,
                   'gt_dir': 'datas_mandelbrot_temporal_with_bg/test_img',
                   'img_idx': 30,
                   'fov': 'small',
                   'RGBx_ch': 389
                  },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_temporal_with_bg_non_gan_on_const_all/test',
                   '1x_1sample_mandelbulb_temporal_with_bg_non_gan_on_const_aux_largest_capacity/test'],
                   'print': 'Mandelbulb',
                   'img_idx': 12,
                   'gt_dir': 'datas_mandelbulb_temporal_with_bg/test_img',
                   'geometry': 'none',
                   'fov': 'small_seperable', 
                   'RGBx_ch': 232
                  },
    'trippy': {'dir': ['1x_1sample_trippy_temporal_stratified_subsample_8/test',
                       '1x_1sample_trippy_temporal_aux_largest_capacity/test'],
               'print': 'Trippy Heart',
               'img_idx': 30,
               'gt_dir': 'datas_trippy_temporal_new_extrapolation_subsample_2/test_img',
               'fov': 'small', 
               'RGBx_ch': 744,
               'every_nth': 2
                  },
    'mandelbrot_simplified': {'dir': ['1x_1sample_mandelbrot_simplified_temporal_with_bg_non_gan_on_const_all/test',
                              '1x_1sample_mandelbrot_simplified_temporal_with_bg_non_gan_on_const_aux_largest_capacity/test'],
                              'print': 'Simplified Mandelbrot',
                              'gamma_correction': 0.7,
                              'img_idx': 30,
                              'gt_dir': 'datas_mandelbrot_simplified_temporal_with_bg/test_img',
                              'fov': 'small', 
                              'RGBx_ch': 226
                             },
    'mandelbulb_simplified': {'dir': ['1x_1sample_mandelbulb_simplified_temporal_with_bg_non_gan_on_const_all/test',
                              '1x_1sample_mandelbulb_simplified_temporal_with_bg_non_gan_on_const_aux_largest_capacity/test'],
                              'print': 'Simplified Mandelbulb',
                              'img_idx': 12,
                              'gt_dir': 'datas_mandelbulb_simplified_temporal_with_bg/test_img',
                              'fov': 'small_seperable',
                              'geometry': 'none',
                              'RGBx_ch': 172
                             },
    
    'trippy_simplified': {'dir': ['1x_1sample_trippy_simplified_temporal_stratified_subsample_4/test',
                                  '1x_1sample_trippy_simplified_temporal_aux_largest_capacity/test'],
                          'print': 'Simplified Trippy Heart',
                          'img_idx': 30,
                          'gt_dir': 'datas_trippy_simplified_temporal_new_extrapolation/test_img',
                          'fov': 'small',
                          'RGBx_ch': 760
                         }
    },
'post_processing': {
    'mandelbulb_blur': {'dir': ['1x_1sample_mandelbulb_with_bg_defocus_blur/test',
                                '1x_1sample_mandelbulb_with_bg_defocus_blur_aux_largest_capacity/test'],
                        'base_runtime_dir': '1x_1sample_mandelbulb_with_bg_all/mean1_test',
                        'img_idx': 21,
                        'img_zoom_bbox': [320, 320+80, 450, 450+60],
                        'gt_dir': 'datas_mandelbulb_defocus_blur/test_img',
                        'gamma_correction': 0.55,
                        'crop_box': [76, 524, 207, -207],
                        'print': 'Mandelbulb Blur',
                        'geometry': 'none',
                        'fov': 'small_seperable',
                        'RGBx_ch': 232
                       },
    'trippy_sharpen': {'dir': ['1x_1sample_trippy_local_laplacian_stratified_subsample_8/test',
                          '1x_1sample_trippy_local_laplacian_aux_largest_capacity/test'],
                       'base_runtime_dir': '1x_1sample_trippy_subsample_2/mean1_test',
                       'img_idx': 30,
                       'img_zoom_bbox': [240, 240+80, 705, 705+60],
                       'gt_dir': 'datas_trippy_new_extrapolation_local_laplacian_subsample_2/test_img',
                       'print': 'Trippy Heart Sharpen',
                       'fov': 'small',
                       'RGBx_ch': 744,
                       'every_nth': 2
                      },
    'trippy_simplified_sharpen': {'dir': ['1x_1sample_trippy_simplified_local_laplacian_stratified_subsample_4/test',
                                     '1x_1sample_trippy_simplified_local_laplacian_aux_largest_capacity/test'],
                                  'base_runtime_dir': '1x_1sample_trippy_subsample_2/mean1_test',
                                  'img_idx': 30,
                                  'img_zoom_bbox': [440, 440+80, 550, 550+60],
                                  'gt_dir': 'datas_trippy_simplified_new_extrapolation_local_laplacian/test_img',
                                  'print': 'Simplified Trippy Heart Sharpen',
                                  'fov': 'small',
                                  'RGBx_ch': 760
                                 }
    },
'simulation': {
    'boids': {'dir': ['boids_all_ini_layers_no_relax_clipping/test',
              'boids_aux_largest_capacity_ini_layers/test'],
              'gt_dir': 'datas_boids',
              'RGBx_ch': 1173,
              'geometry': 'boids_coarse'}
    },
    'unified': {
    'mandelbrot': {'dir': ['1x_1sample_unified_mandelbrot_mandelbulb_trippy_primitives_with_bg_automatic_200/test_mandelbrot',
                           '1x_1sample_unified_mandelbrot_mandelbulb_trippy_primitives_with_bg_aux_largest_capacity/test_mandelbrot'],
                   'img_idx': 30,
                   'img_zoom_bbox': [250, 250+80, 570, 570+60],
                   'gt_dir': 'datas_mandelbrot_with_bg/test_img',
                   'msaa_sample': 5,
                   'print': 'Mandelbrot'
                  },
    'mandelbulb': {'dir': ['1x_1sample_unified_mandelbrot_mandelbulb_trippy_primitives_with_bg_automatic_200/test_mandelbulb_slim',
                           '1x_1sample_unified_mandelbrot_mandelbulb_trippy_primitives_with_bg_aux_largest_capacity/test_mandelbulb_slim'],
                   'img_idx': 20,
                   'img_zoom_bbox': [250, 250+80, 325, 325+60],
                   'gt_dir': 'datas_mandelbulb_with_bg/test_img',
                   'msaa_sample': 2,
                   'print': 'Mandelbulb'
                  },
    'trippy': {'dir': ['1x_1sample_unified_mandelbrot_mandelbulb_trippy_primitives_with_bg_automatic_200/test_trippy_heart',
                           '1x_1sample_unified_mandelbrot_mandelbulb_trippy_primitives_with_bg_aux_largest_capacity/test_trippy_heart'],
                   'img_idx': 30,
                   'img_zoom_bbox': [550, 550+80, 65, 65+60],
                   'gt_dir': 'datas_trippy_new_extrapolation_subsample_2/test_img',
                   'msaa_sample': 9,
                   'print': 'Trippy Heart',
                   'every_nth': 2
                  },
    'gear': {'dir': ['1x_1sample_unified_mandelbrot_mandelbulb_trippy_primitives_with_bg_automatic_200/test_primitives_wheel_only',
                           '1x_1sample_unified_mandelbrot_mandelbulb_trippy_primitives_with_bg_aux_largest_capacity/test_primitives_wheel_only'],
                   'img_idx': 15,
                   'img_zoom_bbox': [420, 420+80, 600, 600+60],
                   'gt_dir': 'datas_primitives_correct_test_range/test_img',
                   'msaa_sample': 1,
                   #'crop_box': [80, -180, 115, -275],
                   'print': 'Gear'
                  }
    }
}

app_names = ['denoising',
             'simplified',
             'post_processing',
             'temporal',
             'simulation',
             'unified']


diagonal_names = [('denoising', 'oceanic'),
                  ('denoising', 'gear'),
                  ('simplified', 'bricks'),
                  ('simplified', 'trippy')]


max_shader_per_fig = 5

def round_msaa(current_ratio):
    # keep 2 significant digits
    if current_ratio >= 100:
        ans = '%d' % (round(current_ratio / 10**(np.floor(numpy.log10(current_ratio)) - 1)) * 10**(np.floor(numpy.log10(current_ratio)) - 1))
    elif current_ratio >= 10:
        ans = '%d' % round(current_ratio)
    elif current_ratio >= 1:
        ans = '%.1f' % current_ratio
    else:
        ans = '%.2f' % current_ratio
    return ans

def get_score(dirs, app_name):

    neval = len(dirs)
    if app_name == 'denoising':
        assert neval == 3
    else:
        assert neval >= 2
        #neval = 2
        
    if app_name == 'unified':
        print('here')

    score = -np.ones((neval, 3))
    l2_score = -np.ones((neval, 3))
    ssim_score = -np.ones(neval)
    psnr_score = -np.ones(neval)
    
    runtime = []

    for i in range(neval):

        dir = dirs[i]
        
        base_dir, child_dir = os.path.split(dir)
        
        time_stats_file = os.path.join(model_parent_dir, dir, 'time_stats.txt')
        
        if os.path.exists(time_stats_file):
            time_vals = open(time_stats_file).read().split(',')
            median_runtime = float(time_vals[0])
            runtime.append(median_runtime)

        perceptual_breakdown_file = os.path.join(model_parent_dir, dir, 'perceptual_tf_breakdown.txt') 

        l2_single_file = os.path.join(model_parent_dir, dir, 'all_loss.txt')
        l2_raw_file = os.path.join(model_parent_dir, dir, 'all_l2.npy')

        if app_name in ['denoising', 'simplified', 'post_processing', 'temporal', 'unified']:
            
            if os.path.exists(perceptual_breakdown_file):
                perceptual_scores = open(perceptual_breakdown_file).read()
            else:
                assert allow_missing_MSAA and app_name in ['denoising', 'simplified'] and i == neval - 1, perceptual_breakdown_file
                perceptual_scores = '1,1,1'

            perceptual_scores.replace('\n', '')
            perceptual_scores.replace(' ', '')
            perceptual_scores = perceptual_scores.split(',')
            perceptual_scores = [float(score) for score in perceptual_scores]
            score[i][0] = perceptual_scores[2]
            score[i][1] = (perceptual_scores[0] + perceptual_scores[1]) / 2.0
            score[i][2] = (perceptual_scores[0] * 5 + perceptual_scores[1] * 5 + perceptual_scores[2] * 20) / 30
        elif app_name == 'simulation':
            
            l2_score_raw = np.load(l2_raw_file)
            score[i][0] = np.mean(l2_score_raw[19:64])
            score[i][1] = np.mean(l2_score_raw)
            score[i][2] = np.mean(l2_score_raw[19:64])
            
            #l2_scores = open(l2_single_file).read()
            #l2_scores.replace('\n', '')
            #l2_scores.replace(' ', '')
            #l2_scores = l2_scores.split(',')
            #l2_scores = [float(score) for score in l2_scores]
            #score[i][0] = l2_scores[0]
            #score[i][1] = l2_scores[0]
            #score[i][2] = l2_scores[0]
        else:
            raise

        if app_name not in ['simulation']:

            if app_name != 'temporal':
                l2_breakdown_file = os.path.join(model_parent_dir, dir, 'score_breakdown.txt')
            else:
                l2_breakdown_file = os.path.join(model_parent_dir, dir, 'score_breakdown.txt')

            if os.path.exists(l2_breakdown_file):
                l2_scores = open(l2_breakdown_file).read()
            else:
                assert allow_missing_MSAA and app_name in ['denoising', 'simplified'] and i == neval - 1
                l2_scores = '1,1,1'

            l2_scores.replace('\n', '')
            l2_scores.replace(' ', '')
            l2_scores = l2_scores.split(',')
            l2_scores = [float(score) for score in l2_scores]
            l2_score[i][0] = l2_scores[2]
            l2_score[i][1] = (l2_scores[0] + l2_scores[1]) / 2.0
            l2_score[i][2] = (l2_scores[0] * 5 + l2_scores[1] * 5 + l2_scores[2] * 20) / 30
            
        if app_name not in ['simulation']:
            
            ssim_file = os.path.join(model_parent_dir, dir, 'ssim.txt')
            psnr_file = os.path.join(model_parent_dir, dir, 'psnr.txt')
            
            ssim_score_str = open(ssim_file).read().replace('\n', '')
            psnr_score_str = open(psnr_file).read().replace('\n', '')
            
            ssim_score[i] = float(ssim_score_str)
            psnr_score[i] = float(psnr_score_str)
                            

    if app_name in ['denoising', 'simplified', 'post_processing']:
        if len(runtime) != neval:
            print(app_name)
            print(len(runtime), neval)
            print(dirs)
            raise
        
    return score, l2_score, ssim_score, psnr_score, runtime

def main():
    
    if len(sys.argv) < 2:
        print('Usage: python generate_result.py model_parent_dir [result_type]')
        raise
        
    global model_parent_dir, dataset_parent_dir
        
    model_parent_dir = os.path.join(sys.argv[1], 'models')
    dataset_parent_dir = os.path.join(sys.argv[1], 'datasets')
    
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        mode = 'all'
        
    if not os.path.isdir('result_figs'):
        os.mkdir('result_figs')
    
    if not os.path.isdir('result_figs/%s' % tex_prefix):
        os.mkdir('result_figs/%s' % tex_prefix)
    
    
    # barplot summary over all apps
    
    bar_x_ticks = []
    bar_avg = []
    bar_dif = []
    bar_sim = []
    bar_col = []
    bar_edge_width = []

    full_shader_idx = []
    simplified_shader_idx = []

    

    slice_start = np.zeros(len(app_names))
    slice_end = np.zeros(len(app_names))

    slice_start[0] = -0.5

    for k in range(len(app_names)):
        app_name = app_names[k]
        if k > 0:
            slice_start[k] = slice_end[k-1] + 1

        app_data = app_shader_dir_200[app_name]
        
        for shader_name in sorted(app_data.keys()):
            
            neval = len(app_data[shader_name]['dir'])
            if app_name == 'denoising':
                assert neval == 3
            else:
                assert neval >= 2
                #neval = 2
                
            score, l2_score, ssim_score, psnr_score, runtime = get_score(app_data[shader_name]['dir'], app_name)
            
            bar_avg.append(score[0, 2] / score[1, 2])
            if score[0, 1] > 0:
                bar_dif.append(score[0, 1] / score[1, 1])
                bar_sim.append(score[0, 0] / score[1, 0])
            else:
                bar_dif.append(None)
                bar_sim.append(None)

            app_data[shader_name]['perceptual'] = score
            
            if app_name not in ['simulation']:
                app_data[shader_name]['l2'] = l2_score
                app_data[shader_name]['ssim'] = ssim_score
                app_data[shader_name]['psnr'] = psnr_score
                
            app_data[shader_name]['runtime'] = runtime
            if app_name in ['denoising', 'simplified', 'post_processing']:
                time_stats_file = os.path.join(model_parent_dir, app_data[shader_name]['base_runtime_dir'], 'time_stats.txt')
                if os.path.exists(time_stats_file):
                    time_vals = open(time_stats_file).read().split(',')
                    median_runtime = float(time_vals[0])
                    base_runtime = median_runtime
                app_data[shader_name]['base_runtime'] = base_runtime
                
            if app_name == 'unified':
                continue


            if 'simplified' in shader_name or 'simplified' in app_name:
                bar_edge_width.append(1)
                simplified_shader_idx.append(len(bar_x_ticks))
            else:
                bar_edge_width.append(0)
                full_shader_idx.append(len(bar_x_ticks))

            for name in hue_shaders.keys():
                if shader_name.startswith(name):
                    current_hue = hue_shaders[name]
                    current_col = colorsys.hsv_to_rgb(*hue_shaders[name])
                    bar_x_ticks.append(name)
                    break

            #current_col = colorsys.hsv_to_rgb(current_hue, 1, 1)
            bar_col.append(current_col)

            if app_name == 'post_processing':
                if 'blur' in shader_name:
                    bar_x_ticks[-1] += '_blur'
                elif 'sharpen' in shader_name:
                    bar_x_ticks[-1] += '_sharpen'
                else:
                    raise

        if app_name == 'unified':
            continue

        slice_end[k] = len(bar_x_ticks) - 1 + 0.5
        bar_x_ticks.append('')
        bar_avg.append(0)
        bar_dif.append(0)
        bar_sim.append(0)
        bar_col.append((0, 0, 0))
        bar_edge_width.append(0)
        
    for app_name in app_shader_dir_200.keys():
        if app_name not in app_names:
            app_data = app_shader_dir_200[app_name]
            for shader_name in app_data.keys():
                score, l2_score, ssim_score, psnr_score, _ = get_score(app_data[shader_name]['dir'], app_name)
                app_data[shader_name]['perceptual'] = score
                app_data[shader_name]['l2'] = l2_score
                app_data[shader_name]['ssim'] = ssim_score
                app_data[shader_name]['psnr'] = psnr_score

    if 'fig2' in mode or mode == 'all':
        
        fontsize = 14
        
        bar_x = np.array([0, 1, 2, 3, 4, 5, 6,
                          7,
                          8, 9, 10, 11, 12, 
                          13,
                          14, 15, -2,
                          16,
                          -2, -2, -2, -2, -2, -2,
                          -2,
                          17,
                          -2])
        
        fig = plt.figure()
        fig.set_size_inches(9, 4.5)

        full_temporal_idx = [18, 20, 22]
        simplified_temporal_idx = [19, 21, 23]

        ax = plt.subplot(111)
        plt.bar(bar_x[full_shader_idx], [bar_avg[i] for i in full_shader_idx], color=[bar_col[i] for i in full_shader_idx], edgecolor='k', linestyle='-')
        plt.bar(bar_x[simplified_shader_idx], [bar_avg[i] for i in simplified_shader_idx], color=[bar_col[i] for i in simplified_shader_idx], edgecolor='k', linestyle='-')
        
        plt.xticks(bar_x, bar_x_ticks, rotation=90, fontsize=fontsize)

        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(fontsize)

        
        font = matplotlib.font_manager.FontProperties()
        font.set_family('serif')
        font.set_name(['Times New Roman'] + plt.rcParams['font.serif'])
        font.set_size(14)
                                
        text = plt.text(2.0, 1.07, 'denoising', fontproperties=font, fontweight='bold')
        text = plt.text(9.0, 1.07, 'simplified', fontproperties=font, fontweight='bold')
        text = plt.text(14.0, 1.07, 'post', fontproperties=font, fontweight='bold')
        text = plt.text(16.6, 1.07, 'sim', fontproperties=font, fontweight='bold')

        plt.axvspan(-0.75, 6.75, facecolor=[0, 0, 0], alpha=0.1)
        plt.axvspan(7.25, 12.75, facecolor=[0, 0, 0], alpha=0.1)
        plt.axvspan(13.25, 15.75, facecolor=[0, 0, 0], alpha=0.1)
        plt.axvspan(16.25, 17.75, facecolor=[0, 0, 0], alpha=0.1)
            
        plt.xlim(-0.6, 17.6)

        plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), 'k-', label='full shader')
        plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), 'k--', label='simplified shader')

        line, = plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), label='baseline')
        
        plt.text(7.0, 1.0, 'Strongest\nBaselines', bbox=dict(facecolor='white', alpha=1, edgecolor=line.get_color(), linewidth=line.get_linewidth(), boxstyle=BoxStyle("Round", pad=0.5)), fontsize=16, color=line.get_color(), ha='center', va='center')

        plt.ylim(0, 1.19)
        plt.ylabel('Relative Error', fontsize=fontsize)

        plt.tight_layout()
        plt.savefig('result_figs/fig2.png')
        plt.close(fig)
        
        print('Result saved in result_figs/fig2.png')
    
    
    if 'table' in mode or mode == 'all':
        # table

        str = ""
        
        
        str_transpose = ""
        
        str_extra = """
\setlength{\\tabcolsep}{4.0pt}
\\begin{table}[]
    \\begin{tabular}{c|l|c@{ / }c@{ / }cc@{ / }c@{ / }c}
        \\hline
        App & Shader & \multicolumn{3}{c}{RGBx} & \multicolumn{3}{c}{Ours} \\\\ \\hline
        """

        #for k in range(len(app_names)):
            #app_name = app_names[k]
            
        #for app_name in sorted(app_shader_dir_200.keys()):
        for app_name in app_names:
            if app_name not in app_shader_dir_200.keys():
                continue
            
            if app_name in ['simulation']:
                continue

            avg_ratio = np.empty([len(app_shader_dir_200[app_name].keys()), 2, 3])
                
            if app_name == 'denoising':
                str_extra += """
        {\multirow{7}{*}{\\rotatebox[origin=c]{90}{Denoising}}}"""
            elif app_name == 'simplified':
                str_extra += """
        {\multirow{5}{*}{\\rotatebox[origin=c]{90}{Simplified}}}"""
            elif app_name == 'post_processing':
                str_extra += """
        {\multirow{3}{*}{\\rotatebox[origin=c]{90}{Post}}}"""
            elif app_name == 'temporal':
                str_extra += """
        {\multirow{6}{*}{\\rotatebox[origin=c]{90}{Temporal}}}"""
            elif app_name == 'unified':
                str_extra += """
        {\multirow{4}{*}{\\rotatebox[origin=c]{90}{Shared}}}"""
            
            
            
            if app_name == 'denoising':
                str_transpose += """
    \\vspace{-1ex}
    \setlength{\\tabcolsep}{4.0pt}
    \\begin{table}[]
    \\begin{tabular}{l|c@{ / }c@{ / }cc@{ / }c@{ / }cc@{ (}r@{) / }c@{ / }c}
    \\hline

        Shader & \multicolumn{3}{c}{RGBx} & \multicolumn{3}{c}{Ours} & \multicolumn{4}{c}{Supersampling} \\\\ \\hline
    """
            else:
                str_transpose += """
    \\vspace{-1ex}
    \setlength{\\tabcolsep}{4.0pt}
    \\begin{table}[]
    \\begin{tabular}{l|c@{ / }c@{ / }cc@{ / }c@{ / }c}
    \\hline

        Shader & \multicolumn{3}{c}{RGBx} & \multicolumn{3}{c}{Ours} \\\\ \\hline
    """
                

            #for shader_name in sorted(app_shader_dir_200[app_name].keys()):
            for i in range(len(app_shader_dir_200[app_name].keys())):
                
                shader_name = sorted(app_shader_dir_200[app_name].keys())[i]
                
                print_name = shader_name
                if print_name not in hue_shaders.keys():
                    print_name = print_name.replace('_', '\\\\', 1)
                    print_name = print_name.replace('_', '\\ ')
                    
                if app_name == 'post_processing':
                        
                    print_names = print_name.split('\\\\')
                    assert len(print_names) == 2
                    print_a = print_names[0]
                    print_b = print_names[1]

                    if 'simplified' in print_b:
                        print_b = 'simp sharpen'
                    
                elif app_name == 'temporal':
                    print_names = print_name.split('\\\\')
                    assert len(print_names) <= 2
                    print_a = print_names[0]

                    if len(print_names) == 2:
                        print_b = print_names[1]
                        assert print_b == 'simplified', print_b
                        print_b = 'simp '
                    else:
                        print_b = ''
                
            
                data = app_shader_dir_200[app_name][shader_name]
                
                avg_ratio[i, 0] = data['l2'][0] / data['l2'][1]
                avg_ratio[i, 1] = data['perceptual'][0] / data['perceptual'][1]


                argmin_perceptual = np.argmin(data['perceptual'], 0)
                argmin_l2 = np.argmin(data['l2'], 0)
                argmax_ssim = np.argmax(data['ssim'])
                argmax_psnr = np.argmax(data['psnr'])
                if app_name == 'denoising':
                    row_data_rel = [1, 0, 2]
                else:
                    row_data_rel = [1, 0]
                count = 0
                
                if app_name != 'simulation':

                    if app_name == 'denoising':
                        transpose_strs = [None] * (3 * len(row_data_rel) + 1)
                    else:
                        transpose_strs = [None] * (3 * len(row_data_rel))

                    count_transpose = 0
                    for row in range(len(row_data_rel)):
                        for col in range(3):
                            data_row = row_data_rel[row]
                            idx = 2
                            if col == 0:
                                field = 'perceptual'
                                argmin_idx = argmin_perceptual[idx]
                            elif col == 1:
                                field = 'ssim'
                                argmin_idx = argmax_ssim
                            else:
                                field = 'psnr'
                                argmin_idx = argmax_psnr

                            if col == 0:
                                abs_val = '%.1e' % data[field][data_row, idx]

                                if row == 0:
                                    transpose_strs[count_transpose] = abs_val
                                elif row == 1:
                                    rel_val = '%02d' % (data[field][data_row, 2] / data[field][1, 2] * 100)
                                    transpose_strs[count_transpose] = abs_val + '(' + rel_val + '\\%)'
                                else:
                                    current_ratio = round_msaa(data[field][data_row, idx] / data[field][1, idx])
                                    #transpose_strs[count_transpose] = abs_val + ' (' + current_ratio + 'x)'
                                    transpose_strs[count_transpose] = abs_val
                                    count_transpose += 1
                                    transpose_strs[count_transpose] = current_ratio + 'x'
                            elif col == 1:
                                # SSIM
                                transpose_strs[count_transpose] = '%.3f' % data[field][data_row]
                            else:
                                # PSNR
                                transpose_strs[count_transpose] = '%.2f' % data[field][data_row]

                            if data_row == argmin_idx:

                                transpose_strs[count_transpose] = '\\textbf{' + transpose_strs[count_transpose] + '}'

                            count_transpose += 1
                            
                    if app_name == 'denoising':


                        str_transpose += """
            \\%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\ \\hline""" % ((print_name, ) + tuple(transpose_strs))

                    else:



                        if app_name == 'post_processing':

                            str_transpose += """
             %s & %s & %s & %s & %s & %s & %s \\\\ \\hline""" % ((print_b, ) + tuple(transpose_strs))


                        elif app_name == 'temporal':

                            str_transpose += """
             %s\\%s & %s & %s & %s & %s & %s & %s \\\\ \\hline""" % ((print_b, print_a, ) + tuple(transpose_strs))

                        else:

                            str_transpose += """
             \\%s & %s & %s & %s & %s & %s & %s \\\\ \\hline""" % ((print_name, ) + tuple(transpose_strs))
                        
                if app_name not in ['simulation']:
                    count_transpose = 0
                    extra_strs = [None] * 6
                    for row in range(2):
                        for col in range(4):
                            data_row = row_data_rel[row]
                            if col == 0:
                                field = 'perceptual'
                                argmin_idx = argmin_perceptual[2]
                            elif col == 1:
                                continue
                                field = 'l2'
                                argmin_idx = argmin_l2[2]
                            elif col == 2:
                                field = 'ssim'
                                argmin_idx = argmax_ssim
                            else:
                                field = 'psnr'
                                argmin_idx = argmax_psnr
                                
                            if col < 2:
                                
                                abs_val = '%.1e' % data[field][data_row, 2]
                                
                                if row == 0:
                                    extra_strs[count_transpose] = abs_val
                                    #val_before, val_after = abs_val.split('e')
                                    #extra_strs[count_transpose] = val_before
                                    #count_transpose += 1
                                    #extra_strs[count_transpose] = 'e' + val_after
                                else:
                                    rel_val = '%02d' % (data[field][data_row, 2] / data[field][1, 2] * 100)
                                    extra_strs[count_transpose] = abs_val + ' (' + rel_val + '\\%)'
                                    #extra_strs[count_transpose] = '%02d' % (data[field][data_row, 2] / data[field][1, 2] * 100)
                            elif col == 2:
                                # SSIM
                                extra_strs[count_transpose] = '%.3f' % data[field][data_row]
                            else:
                                # PSNR
                                extra_strs[count_transpose] = '%.2f' % data[field][data_row]
                            
                            if data_row == argmin_idx:
                                extra_strs[count_transpose] = '\\textbf{' + extra_strs[count_transpose] + '}'
                                    
                            count_transpose += 1
                            
                    if app_name in ['post_processing', 'temporal']:
                        current_print = '\\' + print_a + '\ ' + print_b
                    else:
                        current_print = '\\' + print_name

                    str_extra += """ & %s & %s & %s & %s & %s & %s & %s \\\\ 
    """ % ((current_print, ) + tuple(extra_strs))
                    
                    if i == len(app_shader_dir_200[app_name].keys()) - 1:
                        str_extra += """ \\hline """
                    else:
                        str_extra += """ \cline{2-8} 
                        """


                

            
            print('avg ratio for', app_name)
            print(numpy.mean(avg_ratio, 0))
            numpy.save('avg_ratio_%s'% app_name, avg_ratio)
            
         
            
            str_transpose = str_transpose + """
    \end{tabular}
    \caption{%s}
    \end{table}
    """ % app_name.replace('_', '\\ ')
            
        
            
        str_transpose = open('result_figs/table_begin.tex').read() + str_transpose + open('result_figs/table_end.tex').read()

        open('result_figs/table.tex', 'w').write(str_transpose)
        
        
        
        try:
            os.system('cd result_figs; pdflatex table.tex; cd ..')
            print('Table compiled as result_figs/table.pdf')
        except:
            print('pdflatex compilation failed, table tex file saved in result_figs/table.tex')
            
        
        
        str_extra = str_extra + """
    \end{tabular}
    \caption{}
    \end{table}
    """
        str_extra = open('result_figs/table_begin.tex').read() + str_extra + open('result_figs/table_end.tex').read()
        open('result_figs/table_extra.tex', 'w').write(str_extra)
        
        #try:
        #    os.system('cd result_figs; pdflatex table_extra.tex; cd ..')
        #    print('Table with extra PSNR and SSIM data compiled as result_figs/table_extra.pdf')
        #except:
        #    print('pdflatex compilation failed, table extra tex file saved in result_figs/table_extra.tex')
        
    
    
    if 'fig_main' in mode or mode == 'all':
        # result image for all apps
        crop_edge_col = [12, 144, 36]
        bbox_edge_w = 3
        
        post_processing_w = 780
        post_processing_cropped = (960 - post_processing_w) // 2

        str = ''

        no_zoom_defined = False

        for k in range(len(app_names)):
            
            app_name = app_names[k]
            
            if app_name in ['temporal', 'simulation', 'unified']:
                continue
            
            fig_start = True
            fig_rows = 0


            app_data = app_shader_dir_200[app_name]
            for shader_name in sorted(app_data.keys()):


                data = app_data[shader_name]

                if shader_name == 'oceanic_corrected':
                    print('here')

                if 'img_idx' in data.keys() or 'other_view' in data.keys():

                    if 'img_idx' in data.keys():
                        if shader_name == 'mandelbrot' and use_gamma_corrected:
                            for i in range(len(data['dir'])):
                                if data['dir'][i].endswith('/'):
                                    data['dir'][i] = data['dir'][i][:-1]
                                data['dir'][i] = data['dir'][i] + '_gamma_corrected'
                            if data['gt_dir'].endswith('/'):
                                data['gt_dir'] = data['gt_dir']
                            data['gt_dir'] = data['gt_dir'] + '_gamma_corrected'

                        orig_imgs = []
                        for i in range(len(data['dir'])):

                            current_dir = data['dir'][i]
                                
                            if app_name == 'temporal':
                                orig_img_name = os.path.join(model_parent_dir, current_dir, '%06d27.png' % data['img_idx'])
                            else:
                                orig_img_name = os.path.join(model_parent_dir, current_dir, '%06d.png' % data['img_idx'])
                                
                            if os.path.exists(orig_img_name):
                                orig_imgs.append(skimage.io.imread(orig_img_name))
                            else:
                                assert allow_missing_MSAA and app_name in ['denoising', 'simplified'] and i == len(data['dir']) - 1, orig_img_name
                                orig_imgs.append(np.copy(img_pl))

                        
                        if app_name == 'temporal':
                            gt_img = skimage.io.imread(os.path.join(dataset_parent_dir, data['gt_dir'], '29%05d.png' % (data['img_idx']-1)))
                        else:
                            gt_files = sorted(os.listdir(os.path.join(dataset_parent_dir, data['gt_dir'])))
                            gt_img = skimage.io.imread(os.path.join(dataset_parent_dir, data['gt_dir'], gt_files[data['img_idx']-1]))

                        orig_imgs = [gt_img] + orig_imgs
                    else:
                        orig_imgs = []
                        for i in range(len(data['other_view'])):
                            if i == 0:
                                other_view_filename = os.path.join(dataset_parent_dir, data['other_view'][i])
                            else:
                                other_view_filename = os.path.join(model_parent_dir, data['other_view'][i])
                            orig_imgs.append(skimage.io.imread(other_view_filename))


                    if 'gamma_correction' in data.keys():
                        for i in range(len(orig_imgs)):
                            orig_imgs[i] = skimage.exposure.adjust_gamma(orig_imgs[i], data['gamma_correction'])
                    elif 'sigmoid_correction' in data.keys():
                        for i in range(len(orig_imgs)):
                            #orig_imgs[i] = skimage.exposure.adjust_sigmoid(orig_imgs[i], cutoff=data['sigmoid_correction'], gain=10)
                            
                            #orig_imgs[i] = skimage.img_as_ubyte(skimage.exposure.equalize_hist(skimage.img_as_float(orig_imgs[i])))
                            pass
                    elif 'hsv_scale' in data.keys():
                        assert 'hsv_bias' in data.keys()
                        
                        for i in range(len(orig_imgs)):
                            
                            hsv_col = skimage.color.rgb2hsv(orig_imgs[i])
                            hsv_col[:, :, 2] *= data['hsv_scale']
                            hsv_col[:, :, 2] += data['hsv_bias']
                            hsv_col = np.clip(hsv_col, 0, 1)
                            orig_imgs[i] = skimage.img_as_ubyte(skimage.color.hsv2rgb(hsv_col))
                            
                            
                            
                            
                    for i in range(len(orig_imgs)):

                        if i == 0:
                            prefix = 'gt'
                        elif i == 1:
                            prefix = 'ours'
                        elif i == 2:
                            prefix = 'RGBx'
                        elif i == 3:
                            if app_name == 'denoising':
                                prefix = 'MSAA'
                            else:
                                prefix = 'input'
                        else:
                            raise

                        img = orig_imgs[i]

                        if app_name in ['denoising', 'post_processing', 'simplified']:
                            bbox = data['img_zoom_bbox']
                            crop1 = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                            crop_w = 2
                           
                            for c in range(len(crop_edge_col)):
                                crop1[:crop_w, :, c] = crop_edge_col[c]
                                crop1[-crop_w:, :, c] = crop_edge_col[c]
                                crop1[:, :crop_w, c] = crop_edge_col[c]
                                crop1[:, -crop_w:, c] = crop_edge_col[c]

                            skimage.io.imsave(os.path.join('result_figs', tex_prefix, '%s_%s_%s_zoom.png' % (app_name, shader_name, prefix)), crop1)

                            for i in range(len(bbox)):
                                edge = bbox[i]
                                for current_draw in range(edge-bbox_edge_w, edge+bbox_edge_w+1):
                                    for c in range(len(crop_edge_col)):
                                        if i < 2:
                                            img[current_draw, bbox[2]-bbox_edge_w:bbox[3]+bbox_edge_w, c] = crop_edge_col[c]
                                        else:
                                            img[bbox[0]-bbox_edge_w:bbox[1]+bbox_edge_w, current_draw, c] = crop_edge_col[c]

                            
                            if 'crop_box' in data:
                                crop_box = data['crop_box']
                                img = img[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
                            elif app_name == 'post_processing':
                                img = img[:, post_processing_cropped:-post_processing_cropped]
                            
                            skimage.io.imsave(os.path.join('result_figs', tex_prefix, '%s_%s_%s_box.png' % (app_name, shader_name, prefix)), img)
                        else:
                            skimage.io.imsave(os.path.join('result_figs', tex_prefix, '%s_%s_%s.png' % (app_name, shader_name, prefix)), img)

                    if app_name == 'denoising':
                        macro_suffix = 'WithZoom'
                    elif app_name == 'post_processing':
                        macro_suffix = 'ThreeCol'
                    elif app_name == 'simplified':
                        macro_suffix = 'InputWithZoom'
                    else:
                        # used for simplified without zoom
                        macro_suffix = 'WithoutZoom'

                    #if fig_start:
                    if fig_rows == 0 or fig_rows >= max_shader_per_fig:
                        macro = '\ResultsFigWithHeader' + macro_suffix
                        fig_start = False
                        if fig_rows == 0:
                            str += """
    \\begin{figure*}
    """
                        elif fig_rows >= max_shader_per_fig:
                            fig_rows -= max_shader_per_fig
                            str += """
    \\vspace{-2ex}
    \caption{%s}
    \end{figure*}

    \\begin{figure*}
    """ % app_name
                    else: 
                        macro = '\ResultsFigNoHeader' + macro_suffix

                    fig_rows += 1

                    print_name = shader_name
                    if False:
                    #if '_simplified' in shader_name:
                        for name in hue_shaders.keys():
                            if shader_name.startswith(name):
                                print_name = name + '\\ simplified'
                                break
                    print_name = print_name.replace('_', '\\ ')
                    
                    if app_name in ['denoising', 'simplified']:
                        # MSAA
                        current_ratio = data['perceptual'][2, 2] / data['perceptual'][1, 2]
                        msaa_str = round_msaa(current_ratio)
                        
                    if app_name in ['denoising', 'simplified', 'post_processing']:
                        runtime_strs = []
                        for runtime_idx in range(len(data['runtime'])):
                            current_ratio = data['base_runtime'] / data['runtime'][runtime_idx] * 1000
                            runtime_strs.append(round_msaa(current_ratio))

                    if app_name == 'denoising':
                        str += """
    %s{\%s}{%s%s_%s}{0\%%/1x}{%d\%%/%sx}{100\%%/%sx}{%d\,SPP, %sx/%sx}
    """ % (macro, print_name, tex_prefix, app_name, shader_name, int(data['perceptual'][0, 2] / data['perceptual'][1, 2] * 100), runtime_strs[0], runtime_strs[1], data['msaa_sample'], msaa_str, runtime_strs[2])
                    elif app_name == 'post_processing':
                        str += """
    %s{\%s: 0\%%/1x}{%s%s_%s}{100\%%/%sx (baseline)}{%d\%%/%sx}
    """ % (macro, print_name, tex_prefix, app_name, shader_name, runtime_strs[1], int(data['perceptual'][0, 2] / data['perceptual'][1, 2] * 100), runtime_strs[0])
                    elif app_name == 'simplified':
                        str += """
    %s{\%s}{%s%s_%s}{0\%%/1x}{%d\%%/%sx}{100\%%/%sx}{%sx/%sx}
    """ % (macro, print_name, tex_prefix, app_name, shader_name, int(data['perceptual'][0, 2] / data['perceptual'][1, 2] * 100), runtime_strs[0], runtime_strs[1], msaa_str, runtime_strs[2])
                    else:
                        str += """
    %s{\%s: 0\%%}{%s%s_%s}{%d\%%}{100\%% (baseline)}{%sx}
    """ % (macro, print_name, tex_prefix, app_name, shader_name, int(data['perceptual'][0, 2] / data['perceptual'][1, 2] * 100), msaa_str)
                    

            if not fig_start:
                str += """
    \\vspace{-2ex}
    \caption{%s}
    \end{figure*}
    """ % app_name.replace('_', ' ')
                
        str = open('result_figs/fig_begin.tex').read() + str + open('result_figs/table_end.tex').read()
        open('result_figs/fig_main.tex', 'w').write(str)
        
        try:
            os.system('cd result_figs; pdflatex fig_main.tex; cd ..')
            print('Table compiled as result_figs/fig_main.pdf')
        except:
            print('pdflatex compilation failed, table tex file saved in result_figs/fig_main.tex')
            
    if 'html' in mode or mode == 'all':
        html_img_format = '.jpg'
        
        # generate images for html viewer
        base_dir = os.path.join(sys.argv[1], 'html_viewer/images')
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
            
        app_name_str = ''

        for k in range(len(app_names)):
            app_name = app_names[k]

            if app_name in ['simulation', 'unified']:
                continue

            if app_name == 'post_processing':
                app_name_dir = 'post'
                app_name_print = 'Post Processing'
            elif app_name == 'simplified':
                app_name_dir = 'simplification'
                app_name_print = 'Simplification'
            else:
                app_name_dir = app_name
                if app_name == 'denoising':
                    app_name_print = 'Denoising'
                elif app_name == 'temporal':
                    app_name_print = 'Temporal Coherence'
                else:
                    raise
                    
            app_name_str += '%s,%s\n' % (app_name_dir, app_name_print)
            
            shader_name_str = ''

            app_name_dir = os.path.join(base_dir, app_name_dir)
            if not os.path.exists(app_name_dir):
                os.mkdir(app_name_dir)
                
            if app_name == 'denoising':
                shader_names = ['oceanic',
                                'bricks',
                                'gear',
                                'mandelbrot',
                                'mandelbulb',
                                'venice',
                                'trippy',
                               ]
            elif app_name == 'simplified':
                shader_names = ['venice',
                                'trippy',
                                'bricks',
                                'mandelbulb',
                                'mandelbrot'
                               ]
            elif app_name == 'post_processing':
                shader_names = ['mandelbulb_blur',
                                'trippy_sharpen',
                                'trippy_simplified_sharpen'
                               ]
            elif app_name == 'temporal':
                shader_names = ['trippy',
                                'trippy_simplified',
                               'mandelbrot',
                               'mandelbrot_simplified',
                               'mandelbulb',
                               'mandelbulb_simplified'
                ]
            
            #for shader_name in app_shader_dir_200[app_name].keys():
            for shader_name in shader_names:
                shader_dir = os.path.join(app_name_dir, shader_name)
                if not os.path.exists(shader_dir):
                    os.mkdir(shader_dir)
                    
                data = app_shader_dir_200[app_name][shader_name]

                if 'img_idx' in data.keys() or 'other_view' in data.keys():
                    
                    print(app_name, shader_name)
                    
                    shader_name_str += '%s,%s\n' % (shader_name, data['print'])
                    
                    conditions_str = ''

                    if 'img_idx' in data.keys():
                        
                        if shader_name == 'mandelbrot' and use_gamma_corrected:
                            for i in range(len(data['dir'])):
                                if data['dir'][i].endswith('/'):
                                    data['dir'][i] = data['dir'][i][:-1]
                                if 'gamma_corrected' not in data['dir'][i]:
                                    data['dir'][i] = data['dir'][i] + '_gamma_corrected'
                            if data['gt_dir'].endswith('/'):
                                data['gt_dir'] = data['gt_dir']
                            if 'gamma_corrected' not in data['gt_dir']:
                                data['gt_dir'] = data['gt_dir'] + '_gamma_corrected'


                        orig_imgs = []
                        for i in range(len(data['dir'])):
                            
                            if app_name in ['denoising', 'simplified'] and i == len(data['dir']) - 1:
                                dir_parent, dir_base = os.path.split(data['dir'][i])
                                current_dir = os.path.join(dir_parent, dir_base)
                            else:
                                current_dir = data['dir'][i]
                                                    
                            if app_name == 'temporal':
                                orig_imgs.append(os.path.join(model_parent_dir, current_dir, '%06d27.png' % data['img_idx']))
                            else:
                                orig_imgs.append(os.path.join(model_parent_dir, current_dir, '%06d.png' % data['img_idx']))
                        if app_name == 'temporal':
                            gt_img = os.path.join(dataset_parent_dir, data['gt_dir'], 'test_ground29%05d.png' % (data['img_idx']-1))
                        else:
                            gt_files = sorted(os.listdir(os.path.join(dataset_parent_dir, data['gt_dir'])))
                            gt_img = os.path.join(dataset_parent_dir, data['gt_dir'], gt_files[data['img_idx']-1])

                        orig_imgs = [gt_img] + orig_imgs
                    else:
                            
                        orig_imgs = []
                        for i in range(len(data['other_view'])):
                            if i == 0:
                                other_view_filename = os.path.join(dataset_parent_dir, data['other_view'][i])
                            else:
                                other_view_filename = os.path.join(model_parent_dir, data['other_view'][i])
                            orig_imgs.append(other_view_filename)
                            
                    if 'gamma_correction' in data.keys():
                        for i in range(len(orig_imgs)):
                            orig_imgs[i] = skimage.exposure.adjust_gamma(skimage.io.imread(orig_imgs[i]), data['gamma_correction'])

                    for i in range(len(orig_imgs)):
                        src = orig_imgs[i]
                        if i == 0:
                            dst_name = 'reference'
                            condition_print = 'Reference'
                            rel_perceptual = 0
                            rel_l2 = 0
                            rel_psnr = None
                            rel_ssim = 1
                        elif i == 1:
                            dst_name = 'ours'
                            condition_print = 'Ours'
                            rel_perceptual = int(100 * data['perceptual'][0, 2] / data['perceptual'][1, 2])
                            rel_l2 = int(100 * data['l2'][0, 2] / data['l2'][1, 2])
                            rel_psnr = data['psnr'][0]
                            rel_ssim = data['ssim'][0]
                        elif i == 2:
                            dst_name = 'baseline'
                            condition_print = 'RGBx Baseline'
                            rel_perceptual = 100
                            rel_l2 = 100
                            rel_psnr = data['psnr'][1]
                            rel_ssim = data['ssim'][1]
                        elif i == 3:
                            if app_name == 'denoising':
                                dst_name = 'baseline2'
                                condition_print = 'SuperSampling Baseline'
                            else:
                                dst_name = 'input'
                                condition_print = 'Input'
                            try:
                                rel_perceptual = int(100 * data['perceptual'][2, 2] / data['perceptual'][1, 2])
                                rel_l2 = int(100 * data['l2'][2, 2] / data['l2'][1, 2])
                                rel_psnr = data['psnr'][2]
                                rel_ssim = data['ssim'][2]
                            except:
                                print(shader_name, app_name)
                                raise
                        else:
                            raise
                        dst = os.path.join(shader_dir, dst_name + html_img_format)
                        
                        if rel_psnr is None:
                            psnr_str = 'NA'
                        else:
                            psnr_str = '%.2f' % rel_psnr
                        
                        if isinstance(src, 'a'.__class__):
                            if src.endswith(html_img_format):
                                shutil.copyfile(src, dst)
                            else:
                                src = skimage.io.imread(src)
                                skimage.io.imsave(dst, src)
                        else:
                            skimage.io.imsave(dst, src)
                        
                        caption = '%s / %s / %s' % (app_name_print, data['print'], condition_print)
                        #caption += ' / Relative Perceptual Error: %d%%; Relative L2 Error: %d%%.' % (rel_perceptual, rel_l2)
                        caption += ' / Relative Perceptual Error: %d%%; SSIM: %.3f; PSNR: %s.' % (rel_perceptual, rel_ssim, psnr_str)

                        conditions_str += '%s,%s,%s\n' % (dst_name + html_img_format, condition_print, caption)
                        
                    open(os.path.join(shader_dir, 'conditions.csv'), 'w').write(conditions_str)
            
            open(os.path.join(app_name_dir, 'shaders.csv'), 'w').write(shader_name_str)
            
        open(os.path.join(base_dir, 'applications.csv'), 'w').write(app_name_str)

if __name__ == '__main__':
    main()
