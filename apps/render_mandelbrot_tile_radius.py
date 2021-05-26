from render_util import *
from render_single import *
import numpy

def mandelbrot_tile_radius(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time):

    c_x = Var('c_x', tex_coords[0] / 500.0 - 1.5)
    c_y = Var('c_y', tex_coords[1] / 500.0 - 1.3)

    z_x = ConstExpr(0.0)
    z_y = ConstExpr(0.0)
    zx2 = z_x * z_x
    zy2 = z_y * z_y
    z2 = zx2 + zy2

    niters = 20
    niters_float = float(niters)

    p_x = Var('p_x', -1.0 + 2.0 * tex_coords[0] / 640.0)
    p_y = Var('p_y', -1.0 + 2.0 * tex_coords[1] / 640.0)

    zoo = Var('zoo', 0.25 * sin(0.2 * time) + 0.75)

    c_x = Var('c_x', -0.533516 + p_x * zoo)
    c_y = Var('c_y', 0.526141 + p_y * zoo)

    cond_disc = ConstExpr(0.0)

    disc_pos = 0.6
    disc_len = 0.1
    n = ConstExpr(0.0)
    compiler.DEFAULT_FOR_LOOP_NAME = 'forloop1'
    for i in loop_generator(niters):
        compiler.DEFAULT_FOR_LOOP_ITER = i
        cond_diverge = Var('cond_diverge'+str(i), select(cond_disc, -10.0, 1024.0 - z2))

        new_zx = zx2 - zy2 + c_x
        new_zy = 2 * z_x * z_y + c_y

        z_x = Var('z_x'+str(i), select(cond_diverge > 0.0, new_zx, z_x))
        z_y = Var('z_y'+str(i), select(cond_diverge > 0.0, new_zy, z_y))

        zx2 = z_x * z_x
        zy2 = z_y * z_y
        z2 = zx2 + zy2

        dist = z2 % 4.0
        #dist = z2
        cond_disc = Var('cond_disc'+str(i), (dist > disc_pos) * (dist < (disc_pos + disc_len * sqrt(n))))

        n = Var('n'+str(i), n + select(cond_diverge > 0.0, 1.0, 0.0))

    compiler.DEFAULT_FOR_LOOP_NAME = None
    compiler.DEFAULT_FOR_LOOP_ITER = None
    phase_n = Var('phase_n', 2.0 * numpy.pi * n / niters_float)
    intensity = Var('intensity', sin(numpy.pi * (dist - disc_pos) / (disc_len * sqrt(n))))
    col0_disc = Var('col0_disc', intensity * (0.5 + 0.5 * sin(phase_n)))
    col1_disc = Var('col1_disc', intensity * (0.5 + 0.5 * sin(2.0 * phase_n)))
    col2_disc = Var('col2_disc', intensity * (0.5 + 0.5 * sin(3.0 * phase_n)))

    out_r = Var('out_r', select(cond_disc, col0_disc, 0.0))
    out_g = Var('out_g', select(cond_disc, col1_disc, 0.0))
    out_b = Var('out_b', select(cond_disc, col2_disc, 0.0))

    out_intensity = numpy.array([out_r, out_g, out_b])

    ans = output_color(out_intensity)

    return ans

shaders = [mandelbrot_tile_radius]
is_color = True
fov = 'small'

def main():
    
    if len(sys.argv) < 3:
        print('Usage: python render_[shader].py base_mode base_dir')
        raise
        
    base_mode = sys.argv[1]
    base_dir = sys.argv[2]
    
    camera_dir = os.path.join(base_dir, 'datasets/datas_mandelbrot_with_bg')
    preprocess_dir = os.path.join(base_dir, 'preprocess/mandelbrot')
    
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir, exist_ok=True)
    
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir, exist_ok=True)
    
    if base_mode == 'collect_raw':
        
        camera_pos = numpy.load(os.path.join(camera_dir, 'train.npy'))
        render_t = numpy.load(os.path.join(camera_dir, 'train_time.npy'))
        nframes = render_t.shape[0]
        
        train_start = numpy.load(os.path.join(camera_dir, 'train_start.npy'))
        render_single(os.path.join(preprocess_dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
        
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
                
            render_single(os.path.join(preprocess_dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground' % mode, 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
            
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
        
        t_range = 31.5
        
        for mode in ['train', 'test_close', 'test_far', 'test_middle', 'validate']:
            if mode == 'train':
                nframes = 800
                z_min = 40
                z_max = 240
            elif mode == 'validate':
                nframes = 80
                z_min = 40
                z_max = 240
            elif mode == 'test_close':
                nframes = 5
                z_min = 30
                z_max = 40
            elif mode == 'test_far':
                nframes = 5
                z_min = 240
                z_max = 360
            else:
                nframes = 20
                z_min = 40
                z_max = 240
                
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
                
            camera_pos = [None] * nframes
            render_single(os.path.join(preprocess_dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'z_min': z_min, 'z_max': z_max, 'camera_pos': camera_pos, 'gname': '%s_noisy' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'expand_boundary': expand_boundary})
            
            numpy.save(os.path.join(preprocess_dir, mode + '.npy'), camera_pos)
            
    elif base_mode == 'generate_temporal_dataset':
        
        camera_dir = os.path.join(base_dir, 'datasets/datas_mandelbrot_temporal_with_bg')
        preprocess_dir = os.path.join(base_dir, 'preprocess/mandelbrot_temporal')

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

                render_single(os.path.join(preprocess_dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground_%d' % (mode, t_val), 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})

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
