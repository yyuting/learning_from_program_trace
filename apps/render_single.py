
from render_util import *
import importlib
import time
import sys
import json

def trace(frame, event, arg):
    #print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    if frame.f_code.co_filename.endswith('compiler.py'):
        print("%s:%d" % (event, frame.f_lineno))
    return trace

#sys.settrace(trace)

def render_single(base_dir, shader_name, geometry, normal_map, args, time_error=False, get_objective=False, use_objective=None, verbose=True, log_intermediates=None, nframes=10, check_kw={}, outdir=None, render_kw={}, end_t=None, check_keys=False, render_size=None):
    """
    Render a single shader.

    Arguments:
      - get_objective: if True, then just return the objective function (an Expr instance).
      - use_objective: if True, then use the given objective in place of generating one.
      - nframes:       number of frames to render (use 1 for fast test, None for default)
    """
    end_t   = end_t if end_t is not None else 1.0    # End time of animation sequence

    T0 = time.time()
    if verbose:
        print('Rendering %s' % shader_name, 'normal_map=', normal_map)
    kw = dict(render_kw)
    #print('render_single, render_kw:', render_kw)
    if '--no-ground-truth' in args:
        kw['ground_truth_samples'] = 1
    if '--novel-camera-view' in args:
        kw['num_cameras'] = 2
    else:
        kw['num_cameras'] = 1
    if '--camera-path' in args:
        try:
            camera_path_ind = args.index('--camera-path')
            kw['specific_camera_path'] = int(args[camera_path_ind+1])
        except:
            raise

    if '--log-intermediates' in args:
        log_intermediates = True

    if '--is-cuda' in args:
        kw['is_cuda'] = True

    if '--code-only' in args:
        kw['code_only'] = True

    if '--log_intermediates_level' in args:
        try:
            level_ind = args.index('--log_intermediates_level')
            kw['log_intermediates_level'] = int(args[level_ind+1])
        except:
            raise

    if '--log_intermediates_subset_level' in args:
        try:
            subset_level_ind = args.index('--log_intermediates_subset_level')
            kw['log_intermediates_subset_level'] = int(args[subset_level_ind+1])
        except:
            raise


    if '--collect_loop_statistic' in args:
        kw['collect_loop_statistic'] = True

    if '--sanity_check_loop_statistic' in args:
        assert kw.get('collect_loop_statistic', False) == True
        kw['sanity_check_loop_statistic'] = True

    if '--first_last_only' in args:
        kw['first_last_only'] = True

    if '--last_only' in args:
        kw['last_only'] = True

    if '--subsample_loops' in args:
        try:
            subsample_loops = args.index('--subsample_loops')
            kw['subsample_loops'] = int(args[subsample_loops+1])
        except:
            raise

    if '--last_n' in args:
        try:
            last_n = args.index('--last_n')
            kw['last_n'] = int(args[last_n+1])
        except:
            raise

    if '--first_n' in args:
        try:
            first_n = args.index('--first_n')
            kw['first_n'] = int(args[first_n+1])
        except:
            raise

    if '--first_n_no_last' in args:
        try:
            first_n_no_last = args.index('--first_n_no_last')
            kw['first_n_no_last'] = int(args[first_n_no_last+1])
        except:
            raise

    if '--mean_var_only' in args:
        kw['mean_var_only'] = True

    if '--every_nth' in args:
        try:
            every_nth = args.index('--every_nth')
            kw['every_nth'] = int(args[every_nth+1])
        except:
            raise

    if '--every_nth_stratified' in args:
        kw['every_nth_stratified'] = True

    if '--stratified_random_file' in args:
        try:
            stratified_random_file = args.index('--stratified_random_file')
            kw['stratified_random_file'] = args[stratified_random_file+1]
        except:
            raise

    if '--one_hop_parent' in args:
        kw['one_hop_parent'] = True
        
    if '--chron_order' in args:
        kw['chron_order'] = True
        
    if '--automatic_subsample' in args:
        kw['automate_loop_statistic'] = True
        
    if '--def_loop_log_last' in args:
        kw['def_loop_log_last'] = True
        
    if '--automate_raymarching_def' in args:
        kw['automate_raymarching_def'] = True
        
    if '--temporal_texture_buffer' in args:
        kw['temporal_texture_buffer'] = True
        
    if '--log_only_return_def_raymarching' in args:
        kw['log_only_return_def_raymarching'] = True
        
    if '--SELECT_FEATURE_THRE' in args:
        try:
            feature_thre_idx = args.index('--SELECT_FEATURE_THRE')
            kw['SELECT_FEATURE_THRE'] = int(args[feature_thre_idx+1])
        except:
            raise
    
    n_boids = None
    if '--n_boids' in args:
        try:
            n_boids_idx = args.index('--n_boids')
            n_boids = int(args[n_boids_idx+1])
        except:
            n_boids = None
            
    m = importlib.import_module(shader_name)
    if render_size is not None:
        if hasattr(m, 'width'):
            m.width = render_size[1]
        if hasattr(m, 'height'):
            m.height = render_size[0]
    if n_boids is not None:
        if hasattr(m, 'N_BOIDS'):
            m.N_BOIDS = n_boids
    shaders = m.shaders
    if not 'fov' in kw.keys():
        kw['fov'] = getattr(m, 'fov', None)
    
    check_kw = dict(check_kw)
    if time_error or '--time-error' in args:
        check_kw['time_error'] = True
        check_kw['nerror'] = 5000
        check_kw['nground'] = 1000
        check_kw['skip_save'] = True
        nframes = 1
        kw['ground_truth_samples'] = 0
    if not verbose:
        check_kw['print_command'] = False
    #print('check_kw:', check_kw, sys.argv[1:])
    
    if geometry.startswith('boids'):
        kw['log_getitem'] = False
        
    if '--no_log_getitem' in args:
        kw['log_getitem'] = False
    
    ans = render_any_shaders(shaders, render_size, use_triangle_wave=False, base_dir=base_dir, is_color=m.is_color, end_t=end_t, normal_map=normal_map, geometry=geometry, nframes=nframes, check_kw=check_kw, get_objective=get_objective, use_objective=use_objective, verbose=verbose, log_intermediates=log_intermediates, outdir=outdir, **kw)
    #print('render_single result:', ans)
    T1 = time.time()

    subdir = get_shader_dirname(base_dir, shader_name, normal_map, geometry, render_prefix=True)
    with open(os.path.join(subdir, 'render_command.txt'), 'wt') as f:
        f.write('python render_single.py %s %s %s %s' % (base_dir, shader_name, geometry, normal_map))
    with open(os.path.join(subdir, 'render_time.txt'), 'wt') as f:
        f.write(str(T1-T0))
    while isinstance(ans, list):
        assert len(ans) == 1
        ans = ans[0]
    if isinstance(ans, dict):
        save_dir = subdir
        if outdir is not None:
            save_dir = outdir
        with open(os.path.join(save_dir, 'render_info.json'), 'wt') as f:
            f.write(json.dumps(ans))

    if check_keys:
        check_single_keys(ans)
    return ans

def check_single_keys(ret_d):
    """
    Check that required keys are in the returned dictionary from render_single().
    """
    keys = ['time_f', 'time_g', 'error_f', 'error_g']
    for key in keys:
        if key not in ret_d:
            raise ValueError('missing key: %s' % key)

def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print('python render_single.py base_dir shadername geometry normal_map [--no-ground-truth] [--novel-camera-view] [--time-error] [--camera-path i]')
        print('  Renders a single shader by name, with specified geometry and normal map.')
        sys.exit(1)

    (base_dir, shader_name, geometry, normal_map) = args[:4]

    render_single(base_dir, shader_name, geometry, normal_map, args)

if __name__ == '__main__':
    main()
