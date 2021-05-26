
"""
modified from shadertoy
https://www.shadertoy.com/view/MltBWM
"""

from render_util import *
from render_single import *

def length(vec):
    ans = 0.0
    for i in range(vec.shape[0]):
        ans = ans + vec[i] ** 2.0
    return ans ** 0.5

time_scale = 1.0

def trippy_heart(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time=None):
    """
    Shader arguments:
     - position:   3 vector of x, y, z position on surface
     - tex_coords: 2 vector for s, t texture coords
     - normal:     3 vector of unit normal vector on surface
     - light_dir:  3 vector of unit direction towards light source
     - viewer_dir: 3 vector of unit direction towards camera
     - use_triangle_wave: if True then triangle_wave() should be used instead of fract() for shaders that perform tiling.
    """
    #new_p = (tex_coords / 10.0 + 1.0) * 10.0
    #new_p0 = (new_p[0] % 4.0 - 1.0) * 0.6
    #new_p1 = (new_p[1] % 4.0 - 1.0) * 0.6

    scale = 20.0

    new_p = tex_coords / scale
    #new_p[0] = new_p[0] / 2.0
    new_p = new_p + 2.0
    new_p0 = (new_p[0] % 4.0 - 2.0) * 0.6
    new_p1 = (new_p[1] % 4.0 - 2.0) * 0.6
    #new_p0 = (new_p[0] - 2.0) * 0.6
    #new_p1 = (new_p[1] - 2.0) * 0.6

    new_py = -0.1 - new_p1 * 1.2 + abs(new_p0) * (1.0 - abs(new_p0))
    #new_p[0] = new_p0
    #new_p[1] = new_py
    r = (new_p0 * new_p0 + new_py * new_py) ** 0.5
    d = 0.5
    mX = (sin(time * (0.5 * time_scale)) + 1.0) * 0.2
    mY = (cos(time * (0.5 * time_scale)) + 1.0) * 0.1

    hcol0 = (tex_coords[0] / scale + 1.0)
    hcol1 = (tex_coords[1] / scale + 1.0)
    hcol2 = mX

    for i in loop_generator(100):
        dot_hcol = hcol0 * hcol0 + hcol1 * hcol1 + hcol2 * hcol2
        hcol0 = 1.3 * abs(abs(hcol0) / dot_hcol - 1.0)
        new_hcol2 = 0.999 * abs(abs(hcol1) / dot_hcol - 1.0)
        hcol1 = 0.7 * abs(abs(hcol2) / dot_hcol - mY)
        hcol2 = new_hcol2

    dot_hcol = hcol0 * hcol0 + hcol1 * hcol1 + hcol2 * hcol2
    bcol0 = 1.3 * abs(abs(hcol0) / dot_hcol - 1.0) * 0.5
    bcol2 = 0.999 * abs(abs(hcol1) / dot_hcol - 1.0)
    bcol1 = 0.7 * abs(abs(hcol2) / dot_hcol - mY)

    hcol0 = hcol0 * 2.0

    col = mix(numpy.array([bcol0, bcol1, bcol2]), numpy.array([hcol0, hcol1, hcol2]), smoothstep(-0.15, 0.15, (d - r)))
    #col = numpy.array([smoothstep(-0.15, 0.15, d-r), smoothstep(-0.15, 0.15, d-r), smoothstep(-0.15, 0.15, d-r)])
    ans = output_color(col)
    for expr in normal.tolist():
        expr.log_intermediates_subset_rank = 1
    return ans

shaders = [trippy_heart]                 # This list should be declared so a calling script can programmatically locate the shaders
is_color = True                    # Set to True if the shader is 3 channel (color) and False if shader is greyscale
fov = 'small'

def main():
    
    if len(sys.argv) < 3:
        print('Usage: python render_[shader].py base_mode base_dir')
        raise
        
    base_mode = sys.argv[1]
    base_dir = sys.argv[2]
    
    camera_dir = os.path.join(base_dir, 'datasets/datas_trippy_new_extrapolation_subsample_2')
    preprocess_dir = os.path.join(base_dir, 'preprocess/trippy')
    
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir, exist_ok=True)
    
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir, exist_ok=True)
    
    if base_mode == 'collect_raw':
        
        camera_pos = numpy.load(os.path.join(camera_dir, 'train.npy'))
        render_t = numpy.load(os.path.join(camera_dir, 'train_time.npy'))
        nframes = render_t.shape[0]
        
        train_start = numpy.load(os.path.join(camera_dir, 'train_start.npy'))
        render_single(os.path.join(preprocess_dir, 'train'), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'every_nth': 2})
        
    elif base_mode == 'generate_dataset':
        for mode in ['train', 'test_close', 'test_far', 'test_middle', 'validate']:
            camera_pos = numpy.load(os.path.join(camera_dir, mode + '.npy'))            
            nframes = camera_pos.shape[0]
            
            if mode in ['train', 'validate']:
                tile_start = numpy.load(os.path.join(camera_dir, mode + '_start.npy'))[:nframes]
                render_size = (320, 320)
                tile_only = True
                render_t = numpy.load(os.path.join(camera_dir, mode + '_time.npy'))
            else:
                tile_start = None
                render_size = (640, 960)
                tile_only = False
                render_t_pool = numpy.load(os.path.join(camera_dir, 'test_time.npy'))
                if mode == 'test_close':
                    render_t = render_t_pool[:5]
                elif mode == 'test_far':
                    render_t = render_t_pool[5:10]
                else:
                    render_t = render_t_pool[10:]
                    
            render_t = render_t[:nframes]
                    
            outdir = get_shader_dirname(os.path.join(preprocess_dir, mode), shaders[0], 'none', 'plane')
                
            render_single(os.path.join(preprocess_dir, mode), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground' % mode, 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
            
            
            if mode in ['train', 'validate']:
                target_dir = os.path.join(camera_dir, mode + '_img')
            else:
                target_dir = os.path.join(camera_dir, 'test_img')
                
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
                
            
            for file in os.listdir(outdir):
                if file.startswith('%s_ground' % mode) and file.endswith('.png'):
                    os.rename(os.path.join(outdir, file),
                              os.path.join(target_dir, file))
                    
    elif base_mode == 'sample_camera_pos':
        
        test_render_t = None
        
        t_range = 4 * numpy.pi
        
        for mode in ['train', 'test_close', 'test_far', 'test_middle', 'validate']:
            
            if mode == 'train':
                nframes = 800
                z_min = 26
                z_max = 250
            elif mode == 'test_close':
                nframes = 5
                z_min = 20
                z_max = 26
            elif mode == 'test_far':
                nframes = 5
                z_min = 250
                z_max = 300
            elif mode == 'test_middle':
                nframes = 20
                z_min = 26
                z_max = 250
            else:
                nframes = 80
                z_min = 26
                z_max = 250
                
            camera_pos = numpy.array([[0.0, 0.0, 50.0, numpy.pi, 0.0, 0.0]] * nframes)
            
            for i in range(nframes):
                camera_pos[i, 2] = numpy.random.uniform(z_min, z_max)
                camera_pos[i, -1] = numpy.random.uniform(0, 2 * numpy.pi)
            numpy.save(os.path.join(preprocess_dir, mode + '.npy'), camera_pos)
            
            if mode in ['train', 'validate']:
                expand_boundary = 160
                render_t = np.random.rand(nframes) * t_range
                numpy.save(os.path.join(preprocess_dir, mode + '_time.npy'), render_t)
            else:
                expand_boundary = 0
                if test_render_t is None:
                    test_render_t = np.random.rand(30) * t_range
                    np.save(os.path.join(preprocess_dir, 'test_time.npy'), render_t)
                
                if mode == 'test_close':
                    render_t = test_render_t[:5]
                elif mode == 'test_far':
                    render_t = test_render_t[5:10]
                else:
                    render_t = test_render_t[10:]
                
            render_single(os.path.join(preprocess_dir, mode), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'z_min': z_min, 'z_max': z_max, 'camera_pos': camera_pos, 'gname': '%s_noisy' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'expand_boundary': expand_boundary})
            
    elif base_mode == 'generate_temporal_dataset':
        
        camera_dir = os.path.join(base_dir, 'datasets/datas_trippy_temporal_new_extrapolation_subsample_2')
        preprocess_dir = os.path.join(base_dir, 'preprocess/trippy_temporal')
        
        if not os.path.exists(camera_dir):
            os.makedirs(camera_dir, exist_ok=True)

        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir, exist_ok=True)
        
        for mode in ['train', 'test', 'validate']:
            
            if mode in ['train', 'validate']:
                tile_start = numpy.load(os.path.join(camera_dir, mode + '_start.npy'))
                render_size = (320, 320)
                tile_only = True
                render_t_base = numpy.load(os.path.join(camera_dir, mode + '_time.npy'))
                camera_pos = numpy.load(os.path.join(camera_dir, mode + '.npy'))        
                t_schedule = np.arange(8)
            else:
                tile_start = None
                render_size = (640, 960)
                tile_only = False
                render_t_base = numpy.load(os.path.join(camera_dir, 'test_time.npy'))
                
                camera_pos = np.concatenate((np.load(os.path.join(camera_dir, 'test_close.npy')),
                                             np.load(os.path.join(camera_dir, 'test_far.npy')),
                                             np.load(os.path.join(camera_dir, 'test_middle.npy'))), axis=0)
                t_schedule = [0, 1, 29]
                
            nframes = camera_pos.shape[0]
            outdir = get_shader_dirname(os.path.join(preprocess_dir, mode), shaders[0], 'none', 'plane')
            
            for t_val in t_schedule:
                render_t = render_t_base + t_val / 30

                render_single(os.path.join(preprocess_dir, mode), 'render_trippy_heart', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground_%d' % (mode, t_val), 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
                
            target_dir = os.path.join(camera_dir, '%s_img' % mode)
                
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            for file in os.listdir(outdir):
                if file.startswith('%s_ground' % mode) and file.endswith('.png'):
                    os.rename(os.path.join(outdir, file),
                              os.path.join(target_dir, file))
                    
    return

if __name__ == '__main__':
    main()
