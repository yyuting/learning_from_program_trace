import os
import sys
import argparse_util
from generate_result import app_shader_dir_200

all_applications = ['denoising',
                    'simplified',
                    'post_processing',
                    'simulation',
                    'temporal',
                    'unified']

all_modes = ['train',
             'inference',
             'validation',
             'accurate_timing']

def generate_script(application, args):
    
    entries = app_shader_dir_200[application].keys()
    
    all_str = ''
    
    if args.mode == 'validation':
        warning_msg = 'Note! Due to the storage limit on anonymous Google Drive account, we did not include validation and training dataset. This is only an example showing how to apply validation after training finishes, actually running the code will fail because the validation dataset and intermediate models are not included. The output of this command can be found in the directory of each model, as validation.png and validation.npy.'
        print(warning_msg)
        
    if args.mode == 'train':
        warning_msg = 'Note! Due to the storage limit on anonymous Google Drive account, we did not include validation and training dataset. This is only an example showing how to train the models, actually running the code will fail because the training dataset is not included. The final trained model for every experiment can be found in their corresponding directory.'
        print(warning_msg)
        
        all_str += '# ' + warning_msg
        
    if application == 'simulation':
        # Hard-code the command for boids, as it's very different from other imagery models
        assert len(entries) == 1 and 'boids' in entries
        assert args.mode in ['train', 'inference']
        
        info = app_shader_dir_200[application]['boids']
        dataset_dir = info['gt_dir'].split('/')[0]
        
        for idx in range(len(info['dir'])):
            
            assert idx < 2
            
            if idx == 0:
                model_type = 'ours'
            elif idx == 1:
                model_type = 'RGBx'
            
            model_dir = info['dir'][idx].split('/')[0]
            
            eval_dir = info['dir'][idx]
            gt_dir = info['gt_dir']
            
            cmd = f"""
python model.py --name {args.modelroot}/models/{model_dir} --dataroot {args.modelroot}/datasets/{dataset_dir} --initial_layer_channels 48 --shader_name boids_coarse --geometry boids_coarse --data_from_gpu --epoch 200 --save_frequency 1 --use_batch --batch_size 1000 --niters 500000 --temporal_texture_buffer --train_with_zero_samples --interval_sample_geometric 1.055 --max_sample_time 64 --min_sample_time 20 --train_res --use_validation --random_switch_label --feature_normalize_lo_pct 5 --add_initial_layers"""
            
            if idx == 1:
                cmd += ' --manual_features_only --feature_reduction_ch 1173'
                
            if args.mode == 'train':
                cmd += ' --is_train'
                
            if args.mode == 'inference':
                cmd += ' --boids_seq_metric --boids_single_step_metric'
                
            all_str += f"""
# {args.mode} for Application simulation, shader boids, {model_type}
{cmd}
            """
            
        return all_str
    
    if application == 'unified':
        # Hard-code the command for unified network, as it's differnt from single models
        
        
                
        for idx in range(2):
            
            model_dir = None
        
            for shader in app_shader_dir_200[application].keys():
                info = app_shader_dir_200[application][shader]

                if model_dir is None:
                    model_dir = info['dir'][idx].split('/')[0]
                else:
                    assert model_dir == info['dir'][idx].split('/')[0]
                    
            if idx == 0:
                model_type = 'ours'
                choose_shaders = 4
            else:
                model_type == 'RGBx'
                choose_shaders = 2
                
            cmd = f"""
python unified_network.py --name {args.modelroot}/models/{model_dir} --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --identity_initialize --no_identity_output_layer --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping --dataroot_parent {args.modelroot}/datasets --no_preload --epoch 400 --choose_shaders {choose_shaders}"""
            
            if idx == 1:
                cmd += ' --manual_features_only --multiple_feature_reduction_ch 389,232,744,245'
                
            if args.mode == 'train':
                cmd += ' --is_train'
            
            if args.mode == 'validation':
                cmd += ' --test_training --collect_validate_loss'
            
            if args.mode == 'inference':
                cmd += ' --read_from_best_validation'
                
                for shader in app_shader_dir_200[application].keys():
                    info = app_shader_dir_200[application][shader]
                    
                    eval_dir = info['dir'][idx]
                    gt_dir = info['gt_dir']
                
                    cmd += f"""
                    
python metric_evaluation.py {args.modelroot}/models/{eval_dir} {args.modelroot}/datasets/{gt_dir}"""
                
            all_str += f"""
# {args.mode} for Application unified, {model_type}
{cmd}
"""
        return all_str
            
        
    for entry in entries:
        
        info = app_shader_dir_200[application][entry]
        
        dataset_dir = info['gt_dir'].split('/')[0]
        
        if application in ['denoising', 'simplified', 'simulation']:
            actual_shader = entry
        else:
            actual_shader = entry.split('_')[0]
            
        if application == 'simplified' or 'simplified' in entry:
            actual_shader = actual_shader + '_simplified'
            
        if application in ['simplified', 'temporal'] or 'simplified' in entry:
            epoch = 800
        else:
            epoch = 400
            
        for idx in range(len(info['dir'])):
            model_dir = info['dir'][idx].split('/')[0]
            
            eval_dir = info['dir'][idx]
            gt_dir = info['gt_dir']
            
            extra_flag = ''
            
            if idx == 0:
                model_type = 'ours'
            elif idx == 1:
                model_type = 'RGBx'
                extra_flag = '--manual_features_only --feature_reduction_ch %s' % info['RGBx_ch']
            elif idx == 2:
                if application == 'denoising':
                    model_type = 'supersample'
                    extra_flag = '--mean_estimator --estimator_samples %d' % info['msaa_sample']
                elif application == 'simplified':
                    model_type = 'input'
                    extra_flag = '--mean_estimator --estimator_samples 1'
                else:
                    print('Error! Unknown type')
                    raise
            else:
                print('Error! Unknown type')
                raise
                
            if 'stratified_subsample' in model_dir:
                extra_flag += f""" --specified_ind {args.modelroot}/models/{model_dir}/specified_ind.npy"""
                
            if 'fov' in info.keys():
                extra_flag += ' --fov %s' % info['fov']
                
            if 'geometry' in info.keys():
                extra_flag += ' --geometry %s' % info['geometry']
            else:
                if model_type != 'RGBx':
                    # only for plane
                    extra_flag += ' --no_additional_features --ignore_last_n_scale 7 --include_noise_feature'
                    
            if application in ['simplified', 'temporal'] or 'simplified' in entry:
                extra_flag += ' --patch_gan_loss --gan_loss_scale 0.05 --discrim_train_steps 8'
                
            if application != 'temporal':
                extra_flag += ' --use_batch --batch_size 6'
            else:
                extra_flag += ' --train_temporal_seq --use_queue --check_gt_variance 0.0001 --allow_non_gan_low_var'
                if 'simplified' not in entry:
                    extra_flag += ' --no_spatial_GAN'
                if args.mode == 'inference':
                    extra_flag += ' --inference_seq_len 30'
                
            if 'blur' in entry:
                extra_flag += ' --additional_input'
                
            if actual_shader.startswith('bricks'):
                extra_flag += ' --texture_maps %s/datasets/bricks_texture.npy' % args.modelroot
            elif actual_shader.startswith('venice'):
                extra_flag += ' --texture_maps %s/datasets/venice_texture.npy' % args.modelroot
                
            

            cmd = f"""
python model.py --name {args.modelroot}/models/{model_dir} --dataroot {args.modelroot}/datasets/{dataset_dir} --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --shader_name {actual_shader} --data_from_gpu --identity_initialize --no_identity_output_layer --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --render_sigma 0.3 --epoch {epoch} --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping {extra_flag}"""
            
            if args.mode in ['inference', 'accurate_timing']:
                cmd += ' --read_from_best_validation'
                
            if args.mode == 'validation':
                cmd += ' --test_training --collect_validate_loss'
            
            if args.mode == 'train':
                cmd += ' --is_train'
                
            if args.mode == 'accurate_timing':
                cmd += ' --accurate_timing --repeat_timing 10'
                
            if args.mode == 'inference':
                
                cmd += f"""

python metric_evaluation.py {args.modelroot}/models/{eval_dir} {args.modelroot}/datasets/{gt_dir}"""
                
                if application == 'temporal':
                    cmd += ' all --prefix test_ground29'
                
            all_str += f"""
# {args.mode} for Application {application}, shader {actual_shader}, {model_type}
{cmd}
            """
            
    return all_str
            
                
        

def main():
    
    parser = argparse_util.ArgumentParser(description='GenerateScript')
    parser.add_argument('--modelroot', dest='modelroot', default='', help='root directory storing trained model')
    parser.add_argument('--applications', dest='applications', default='', help='what applications to generate script, should be one of %s' % '/'.join(all_applications))
    parser.add_argument('--mode', dest='mode', default='', help='can be one of %s' % '/'.join(all_modes))
    
    args = parser.parse_args()
    
    assert args.mode in all_modes
    
    if args.applications != '':
        assert args.applications in all_applications
        applications = [args.applications]
    else:
        applications = all_applications
        
    all_str = ''
        
    for application in applications:
        all_str += generate_script(application, args)
        
    open('script.sh', 'w').write(all_str)
    
if __name__ == '__main__':
    main()