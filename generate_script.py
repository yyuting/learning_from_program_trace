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
             'accurate_timing',
             'prepare_dataset',
             'analyze']

def generate_dataset_script(application, args):
    
    if application == 'unified':
        msg = '#Unified does not need a different dataset, it can be trained using existing dataset.'
        print(msg[1:])
        return msg
        
        
    shaders = app_shader_dir_200[application].keys()
    
    all_str = """

cd apps
    """
        
    for shader in shaders:
                
        shader_filename = app_shader_dir_200[application][shader]['shader']
        shader_filename_short = shader_filename.split('.')[0][7:]
        
        dataset_dir = app_shader_dir_200[application][shader]['gt_dir'].split('/')[0]
        
        if application == 'simulation':
            gt_shader_filename = app_shader_dir_200[application][shader]['gt_shader']
        else:
            gt_shader_filename = shader_filename
            
        geometry = app_shader_dir_200[application][shader]['geometry']
        
        all_str += f"""
        
# {application} / {shader}
"""
        
        need_generate_dataset = False
        if application in ['denoising', 'simulation']:
            need_generate_dataset = True
        if application in ['temporal', 'post_processing'] and 'simplified' not in shader:
            need_generate_dataset = True
        
        
        if need_generate_dataset:
            
            if application == 'denoising':
                all_str += f"""
                
python {shader_filename}.py sample_camera_pos {args.modelroot}
                
python generate_tiled_score.py {args.modelroot}/preprocess/{shader}/train/{shader_filename_short}_{geometry}_normal_none train_noisy {args.modelroot}/saliency/{shader}_train {args.modelroot}/models/MLNet.model

python sample_pyramid.py {args.modelroot}/saliency/{shader}_train/tiles {args.modelroot}/preprocess/{shader} train

python update_dataset.py {args.modelroot}/saliency/{shader}_train/tiles {args.modelroot}/preprocess/{shader} train

python generate_tiled_score.py {args.modelroot}/preprocess/{shader}/validate/{shader_filename_short}_{geometry}_normal_none validate_noisy {args.modelroot}/saliency/{shader}_validate {args.modelroot}/models/MLNet.model

python sample_pyramid.py {args.modelroot}/saliency/{shader}_validate/tiles {args.modelroot}/preprocess/{shader} validate

python update_dataset.py {args.modelroot}/saliency/{shader}_validate/tiles {args.modelroot}/preprocess/{shader} validate

mv {args.modelroot}/preprocess/{shader}/test*.npy {args.modelroot}/datasets/{dataset_dir}

mv {args.modelroot}/saliency/{shader}_train/tiles/train*.npy {args.modelroot}/datasets/{dataset_dir}

mv {args.modelroot}/saliency/{shader}_validate/tiles/validate*.npy {args.modelroot}/datasets/{dataset_dir}
"""
            elif application == 'temporal':
                
                denoising_dataset_dir = app_shader_dir_200['denoising'][shader]['gt_dir'].split('/')[0]
                
                all_str += f"""
mkdir -p {args.modelroot}/datasets/{dataset_dir}

ln -s {args.modelroot}/datasets/{denoising_dataset_dir}/*.npy {args.modelroot}/datasets/{dataset_dir}
"""
            elif application == 'post_processing':
                
                denoising_dataset_dir = app_shader_dir_200['denoising'][shader.split('_')[0]]['gt_dir'].split('/')[0]
                
                all_str += f"""
mkdir -p {args.modelroot}/datasets/{dataset_dir}

ln -s {args.modelroot}/datasets/{denoising_dataset_dir}/*.npy {args.modelroot}/datasets/{dataset_dir}
"""
            
            if application in ['temporal', 'denoising', 'simulation']:
                if application == 'temporal':
                    generate_mode = 'generate_temporal_dataset'
                else:
                    generate_mode = 'generate_dataset'

                all_str += f"""
python {gt_shader_filename}.py {generate_mode} {args.modelroot}

"""
            else:
                assert application == 'post_processing'
                
                if 'sharpen' in shader:
                    all_str += f"""
python local_laplacian_postprocessing.py {args.modelroot}/datasets/{denoising_dataset_dir}/train_img {args.modelroot}/datasets/{dataset_dir}/train_img

python local_laplacian_postprocessing.py {args.modelroot}/datasets/{denoising_dataset_dir}/test_img {args.modelroot}/datasets/{dataset_dir}/test_img

python local_laplacian_postprocessing.py {args.modelroot}/datasets/{denoising_dataset_dir}/validate_img {args.modelroot}/datasets/{dataset_dir}/validate_img
"""
                else:
                    print('TODO: finish for blur')
                    continue
        else:
            if application == 'simplified':
                orig_dataset_dir = app_shader_dir_200['denoising'][shader]['gt_dir'].split('/')[0]
            else:
                assert 'simplified' in shader
                
                shader_short = shader.split('_')[0]
                orig_feature_dataset_dir = app_shader_dir_200['simplified'][shader_short]['gt_dir'].split('/')[0]
                
                if application == 'temporal':
                    orig_dataset_dir = app_shader_dir_200[application][shader_short]['gt_dir'].split('/')[0]
                else:
                    orig_dataset_dir = app_shader_dir_200[application][shader.replace('_simplified', '')]['gt_dir'].split('/')[0]
                
                all_str += f"""
mkdir -p {args.modelroot}/datasets/{dataset_dir}
                
ln -s {args.modelroot}/datasets/{orig_feature_dataset_dir}/feature*.npy {args.modelroot}/datasets/{dataset_dir}
"""
            
            all_str += f"""
# Simplified applications uses the same ground truth as the original dataset

ln -s {args.modelroot}/datasets/{orig_dataset_dir}/t* {args.modelroot}/datasets/{dataset_dir}

ln -s {args.modelroot}/datasets/{orig_dataset_dir}/v* {args.modelroot}/datasets/{dataset_dir}

"""
            
        need_preprocess_raw = True
        if application in ['temporal', 'post_processing']:
            need_preprocess_raw = False
           
        if need_preprocess_raw:
            all_str += f"""
        
python {shader_filename}.py collect_raw {args.modelroot}

python preprocess_raw_data.py --base_dirs {args.modelroot}/preprocess/{shader}/train --shadername {shader_filename} --geometry {geometry} --lo_pct 5

mv {args.modelroot}/preprocess/{shader}/train/{shader_filename_short}_{geometry}_normal_none/feature*.npy {args.modelroot}/datasets/{dataset_dir}
"""
        
    msg = '#Warning! Executing this script will overwrite any existing dataset in the modelroot path.'
    
    print(msg[1:])
    
    all_str = msg + all_str + """
cd ..
    """
        
    return all_str

def generate_script(application, args):
    
    if args.mode == 'prepare_dataset':
        return generate_dataset_script(application, args)
    
    if args.mode == 'analyze' and application not in ['denoising', 'simplified']:
        return ''
    
    entries = app_shader_dir_200[application].keys()
    
    all_str = ''
    
    if args.mode == 'validation':
        warning_msg = 'Note! Due to the storage limit on anonymous Google Drive account, we did not include validation and training dataset. This is only an example showing how to apply validation after training finishes. Actually running the code will fail because the validation dataset and intermediate models are not included. The output of this command can be found in the directory of each model, as validation.png and validation.npy.'
        print(warning_msg)
        
        all_str += '# ' + warning_msg
        
    if args.mode == 'train':
        warning_msg = 'Note! Due to the storage limit on anonymous Google Drive account, we did not include validation and training dataset. This is only an example showing how to train the models if training dataset is available. Actually running the code will fail because the training dataset is not included. The final trained model for every experiment can be found in their corresponding directory.'
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
            
        if 'every_nth' in info.keys():
            every_nth = info['every_nth']
        else:
            every_nth = 1
            
        for idx in range(len(info['dir'])):
            
            if args.mode == 'analyze' and idx > 0:
                continue
            
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
                geometry = info['geometry']
                extra_flag += ' --geometry %s' % geometry
            else:
                geometry = 'plane'
            
            if geometry == 'plane':
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
                
            if every_nth > 1:
                extra_flag += ' --every_nth %s' % every_nth
                
            

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
                    
            if args.mode == 'analyze':
                cmd = f"""
{cmd} --read_from_best_validation --get_col_aux_inds

{cmd} --read_from_best_validation --test_training

{cmd} --read_from_best_validation --test_training --analyze_channel --analyze_current_only
"""
                
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