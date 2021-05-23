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
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'collect_raw':
            target_dir = os.path.join(os.environ['PREPROCESS_DATA_DIR'], '1x_1sample_mandelbrot_tile')
            camdir = os.path.join(os.environ['DATASET_DIR'], 'datas_mandelbrot_with_bg')
            
            camera_pos = numpy.load(os.path.join(camdir, 'train.npy'))
            render_t = numpy.load(os.path.join(camdir, 'train_time.npy'))
            train_start = numpy.load(os.path.join(camdir, 'train_start.npy'))
            nframes = render_t.shape[0]
            
            render_single(os.path.join(target_dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start})
            
    return
            
            
    
    # Write raw trace value both before and after sample salient tiles
    # then compute feature_scale and feature_bias to both variants, check which variant are we using for training
    
    dir_before = '/n/fs/scratch/yutingy/1x_1sample_mandelbrot_tile_before_sample'
    dir_after = '/n/fs/scratch/yutingy/1x_1sample_mandelbrot_tile_after_sample'
    
    camdir_before = '/n/fs/shaderml/1x_1sample_mandelbrot_tile_apr20'
    camdir_after = '/n/fs/shaderml/datasets/datas_mandelbrot_with_bg'
    
    for (dir, camdir) in [(dir_before, camdir_before), (dir_after, camdir_after)]:
        camera_pos = numpy.load(os.path.join(camdir, 'train.npy'))
        render_t = numpy.load(os.path.join(camdir, 'train_time.npy'))
        nframes = render_t.shape[0]
        
        if dir == dir_before:
            render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 120), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small'})
        else:
            train_start = numpy.load(os.path.join(camdir, 'train_start.npy'))
            render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start})
            
    return
    
    
    dir = '/n/fs/shaderml/1x_1sample_mandelbrot_tile'
    camdir = '/n/fs/shaderml/datas_mandelbrot_tile_all'
    
    camera_pos = numpy.load(os.path.join(camdir, 'train.npy'))
    render_t = numpy.load(os.path.join(camdir, 'train_time.npy'))
    nframes = render_t.shape[0]
    
    render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 120), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small'})
    return
    
    
    if False:
        dir = '/n/fs/visualai-scr/yutingy/mandelbrot_all_trace'
        camera_pos = numpy.load(os.path.join(camdir, 'train.npy'))[20:23]
        render_t = numpy.load(os.path.join(camdir, 'train_time.npy'))[20:23]
        train_start = numpy.load(os.path.join(camdir, 'train_start.npy'))[20:23]
        nframes = render_t.shape[0]

        render_single(os.path.join(dir, 'full_res'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'robust_simplification': True, 'tile_only': True, 'tile_start': train_start, 'base_ind': 20})
        return
    
    
        camera_pos = numpy.load(os.path.join(camdir, 'train.npy'))
        render_t = numpy.load(os.path.join(camdir, 'train_time.npy'))
        train_start = numpy.load(os.path.join(camdir, 'train_start.npy'))
        nframes = render_t.shape[0]

        render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'collect_loop_and_features': True, 'tile_only': True, 'tile_start': train_start, 'collect_feature_mean_only': True, 'feature_normalize_dir': camdir, 'reference_dir': os.path.join(camdir, 'train_img')})
        return
    
    camdir = '/n/fs/shaderml/1x_1sample_mandelbrot_tile_apr20'
    dir = '/n/fs/visualai-scr/yutingy/mandelbrot_all_trace'
    
    camera_pos = numpy.load(os.path.join(camdir, 'train.npy'))
    render_t = numpy.load(os.path.join(camdir, 'train_time.npy'))
    nframes = render_t.shape[0]

    render_single(dir, 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (10, 15), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small'})
    return
    
    if False:
        # camera pos range:
        # x: -10, 10
        # y: -10, 10
        # z: 40, 240
        # t: 0, 31.5
        camera_pos = numpy.load(os.path.join(dir, 'train.npy'))
        render_t = numpy.load(os.path.join(dir, 'train_time.npy'))
        nframes = render_t.shape[0]

        render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_noisy_expanded', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
        return
    
    if True:
        x_min = -10
        x_max = 10
        y_min = -10
        y_max = 10
        
        mode = 'train'
        nframes = 800
        z_min = 40
        z_max = 240
        render_t = numpy.random.rand(nframes) * 31.5
        camera_pos = [None] * nframes
        render_single(os.path.join(dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'z_min': z_min, 'z_max': z_max, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'camera_pos': camera_pos, 'is_tf': True, 'top_view_only': False, 'allowed_cos': 0.0, 'expand_boundary': 160, 'gname': 'train_noisy_expanded'})
        numpy.save(os.path.join(dir, mode + '.npy'), camera_pos)
        numpy.save(os.path.join(dir, mode + '_time.npy'), render_t)
        return
        
        mode = 'validate'
        
        nframes = 20
        z_min = 40
        z_max = 240
        
        render_t = numpy.random.rand(nframes) * 31.5
        camera_pos = [None] * nframes
        render_single(os.path.join(dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'z_min': z_min, 'z_max': z_max, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'camera_pos': camera_pos, 'is_tf': True, 'top_view_only': False})
        numpy.save(os.path.join(dir, mode + '.npy'), camera_pos)
        numpy.save(os.path.join(dir, mode + '_time.npy'), render_t)
        return
    
    #mode = 'pitch_yaw_all_space'
    #mode = 'roll_pitch_all_space'
    #mode = 'roll_yaw_all_space'
    mode = 'pitch_yaw_around_gt'
    
    #if mode == 'pitch_yaw_all_space':
    if False:
        dir = '/n/fs/visualai-scr/yutingy/1x_1sample_mandelbrot_tile/change_pitch_yaw_all_space'
        # this set of sample get ang2 fixed, which controls roll 
        nframes = 1000
        x_pos = -3.04244192
        y_pos = 3.6468296
        z_pos = 96.581471
        camera_pos = np.empty([nframes, 6])
        camera_pos[:, 0] = x_pos
        camera_pos[:, 1] = y_pos
        camera_pos[:, 2] = z_pos
        ang_low = 0
        ang_hi = 2 * np.pi
        camera_pos[:, 3] = np.random.rand(nframes) * (ang_hi - ang_low) + ang_low
        camera_pos[:, 4] = 4.14606393
        camera_pos[:, 5] = np.random.rand(nframes) * (ang_hi - ang_low) + ang_low
        numpy.save(os.path.join(dir, 'train.npy'), camera_pos)
    
    #elif mode == 'roll_pitch_all_space':
    if False:
        dir = '/n/fs/visualai-scr/yutingy/1x_1sample_mandelbrot_tile/change_roll_pitch_all_space'
        # this set of sample get ang2 fixed, which controls roll 
        nframes = 1000
        x_pos = -3.04244192
        y_pos = 3.6468296
        z_pos = 96.581471
        camera_pos = np.empty([nframes, 6])
        camera_pos[:, 0] = x_pos
        camera_pos[:, 1] = y_pos
        camera_pos[:, 2] = z_pos
        ang_low = 0
        ang_hi = 2 * np.pi
        camera_pos[:, 3] = np.random.rand(nframes) * (ang_hi - ang_low) + ang_low
        camera_pos[:, 4] = np.random.rand(nframes) * (ang_hi - ang_low) + ang_low
        camera_pos[:, 5] = 5.22841233
        numpy.save(os.path.join(dir, 'train.npy'), camera_pos)
    
    if False:
    #elif mode == 'roll_yaw_all_space':
        dir = '/n/fs/visualai-scr/yutingy/1x_1sample_mandelbrot_tile/change_roll_yaw_all_space'
        # this set of sample get ang2 fixed, which controls roll 
        nframes = 10000
        x_pos = -3.04244192
        y_pos = 3.6468296
        z_pos = 96.581471
        camera_pos = np.empty([nframes, 6])
        camera_pos[:, 0] = x_pos
        camera_pos[:, 1] = y_pos
        camera_pos[:, 2] = z_pos
        ang_low = 0
        ang_hi = 2 * np.pi
        camera_pos[:, 3] = 6.04481875
        camera_pos[:, 4] = np.random.rand(nframes) * (ang_hi - ang_low) + ang_low
        camera_pos[:, 5] = np.random.rand(nframes) * (ang_hi - ang_low) + ang_low
        numpy.save(os.path.join(dir, 'train.npy'), camera_pos)
    
    if False:
    #elif mode == 'pitch_yaw_around_gt':
        dir = '/n/fs/visualai-scr/yutingy/1x_1sample_mandelbrot_tile/change_pitch_yaw_around_gt'
        # this set of sample get ang2 fixed, which controls roll 
        nframes = 1000
        x_pos = -3.04244192
        y_pos = 3.6468296
        z_pos = 96.581471
        camera_pos = np.empty([nframes, 6])
        camera_pos[:, 0] = x_pos
        camera_pos[:, 1] = y_pos
        camera_pos[:, 2] = z_pos
        ang_low = 0
        ang_hi = 2 * np.pi
        camera_pos[:, 3] = 6.04481875 + ((np.random.rand(nframes) - 0.5) * (ang_hi - ang_low)) / 10
        camera_pos[:, 4] = 4.14606393
        camera_pos[:, 5] = 5.22841233 + ((np.random.rand(nframes) - 0.5) * (ang_hi - ang_low)) / 10
        numpy.save(os.path.join(dir, 'train.npy'), camera_pos)
    
    if False:
        dir = '/n/fs/visualai-scr/yutingy/1x_1sample_mandelbrot_tile/change_roll_pitch_around_gt'
        # this set of sample get ang2 fixed, which controls roll 
        nframes = 1000
        x_pos = -3.04244192
        y_pos = 3.6468296
        z_pos = 96.581471
        camera_pos = np.empty([nframes, 6])
        camera_pos[:, 0] = x_pos
        camera_pos[:, 1] = y_pos
        camera_pos[:, 2] = z_pos
        ang_low = 0
        ang_hi = 2 * np.pi
        camera_pos[:, 3] = 6.04481875 + ((np.random.rand(nframes) - 0.5) * (ang_hi - ang_low)) / 10
        camera_pos[:, 4] = 4.14606393 + ((np.random.rand(nframes) - 0.5) * (ang_hi - ang_low)) / 10
        camera_pos[:, 5] = 5.22841233
        numpy.save(os.path.join(dir, 'train.npy'), camera_pos)
        
    if False:
        dir = '/n/fs/visualai-scr/yutingy/1x_1sample_mandelbrot_tile/change_roll_yaw_around_gt'
        # this set of sample get ang2 fixed, which controls roll 
        nframes = 1000
        x_pos = -3.04244192
        y_pos = 3.6468296
        z_pos = 96.581471
        camera_pos = np.empty([nframes, 6])
        camera_pos[:, 0] = x_pos
        camera_pos[:, 1] = y_pos
        camera_pos[:, 2] = z_pos
        ang_low = 0
        ang_hi = 2 * np.pi
        camera_pos[:, 3] = 6.04481875
        camera_pos[:, 4] = 4.14606393 + ((np.random.rand(nframes) - 0.5) * (ang_hi - ang_low)) / 10
        camera_pos[:, 5] = 5.22841233 + ((np.random.rand(nframes) - 0.5) * (ang_hi - ang_low)) / 10
        numpy.save(os.path.join(dir, 'train.npy'), camera_pos)
    
    #render_t = numpy.zeros(nframes)
    #render_single(dir, 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t})
    #return
    
    dir = '/n/fs/shaderml/1x_1sample_mandelbrot_tile'
    camdir = '/n/fs/shaderml/datas_mandelbrot_tile_automatic_200'
    
    if False:
        for mode in ['validate']:
            if mode == 'validate':
                nframes = 100
                z_min = 40
                z_max = 240
            camera_pos = [None] * nframes
            render_single(os.path.join(dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (40, 60), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'z_min': z_min, 'z_max': z_max, 'camera_pos': camera_pos})
            numpy.save(os.path.join(dir, mode), camera_pos)

        camera_pos = np.load(os.path.join(dir, 'validate.npy'))
        nframes = camera_pos.shape[0]
        nsamples = 10
        sampled_camera_pos = np.tile(camera_pos, (nsamples, 1))
        camera_sigma = np.array([0.3, 0.3, 1, 0.1, 0.1, 0.1])
        time_sigma = 0.015
        render_t = np.random.rand(nframes)
        render_t = np.tile(render_t, nsamples)
        for n in range(1, nsamples):
            sampled_camera_pos[nframes*n:nframes*(n+1), :] += np.random.randn(nframes, 6) * camera_sigma
            render_t[nframes*n:nframes*(n+1)] += np.random.rand(nframes) * time_sigma
        numpy.save(os.path.join(dir, 'camera_sampled_validate_full_res.npy'), sampled_camera_pos)
        numpy.save(os.path.join(dir, 'camera_sampled_validate_time_full_res.npy'), render_t)
    
    
    
    if True:
        camera_pos = numpy.load(os.path.join(dir, 'train.npy'))
        render_t = numpy.load(os.path.join(dir, 'train_time.npy'))

        camera_pos_periodic_range = numpy.array([0, 0, 0, 2 * np.pi, 2 * np.pi, 2 * np.pi])
        random_half_idx = numpy.random.choice(camera_pos.shape[0], size=camera_pos.shape[0] // 2, replace=False)
        camera_pos -= camera_pos_periodic_range
        camera_pos[random_half_idx] += 2 * camera_pos_periodic_range

        nframes = camera_pos.shape[0]
        render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'automate_loop_statistic': True, 'SELECT_FEATURE_THRE': 200})
        #render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size=(80, 120), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'automate_loop_statistic': True, 'SELECT_FEATURE_THRE': 200})
    
        return
    
    if False:
        camera_pos = numpy.load(os.path.join(camdir, 'train.npy'))
        nframes = camera_pos.shape[0]
        render_t = numpy.load(os.path.join(camdir, 'train_time.npy'))
        tile_start = numpy.load(os.path.join(camdir, 'train_start.npy'))

        nsamples = 100
        sampled_camera_pos = np.tile(camera_pos, (nsamples, 1))
        camera_sigma = np.array([0.3, 0.3, 2, 0.1, 0.1, 0.1])
        time_sigma = 0.015
        tile_start = np.tile(tile_start, (nsamples, 1))
        render_t = np.tile(render_t, nsamples)
        
        for n in range(1, nsamples):
            sampled_camera_pos[nframes*n:nframes*(n+1), :] += np.random.randn(nframes, 6) * camera_sigma
            render_t[nframes*n:nframes*(n+1)] += np.random.rand(nframes) * time_sigma
        numpy.save(os.path.join(dir, 'camera_sampled_train_v2.npy'), sampled_camera_pos)
        numpy.save(os.path.join(dir, 'camera_sampled_tile_start.npy'), tile_start)
        numpy.save(os.path.join(dir, 'camera_sampled_train_time.npy'), render_t)
    
        camera_pos = np.load(os.path.join(dir, 'train.npy'))
        nframes = camera_pos.shape[0]
        nsamples = 100
        sampled_camera_pos = np.tile(camera_pos, (nsamples, 1))
        camera_sigma = np.array([0.3, 0.3, 1, 0.1, 0.1, 0.1])
        time_sigma = 0.015
        render_t = np.random.rand(nframes)
        render_t = np.tile(render_t, nsamples)
        for n in range(1, nsamples):
            sampled_camera_pos[nframes*n:nframes*(n+1), :] += np.random.randn(nframes, 6) * camera_sigma
            render_t[nframes*n:nframes*(n+1)] += np.random.rand(nframes) * time_sigma
        numpy.save(os.path.join(dir, 'camera_sampled_train_full_res.npy'), sampled_camera_pos)
        numpy.save(os.path.join(dir, 'camera_sampled_train_time_full_res.npy'), render_t)
    
    camera_pos0 = np.load(os.path.join(camdir, 'test_close.npy'))
    camera_pos1 = np.load(os.path.join(camdir, 'test_far.npy'))
    camera_pos2 = np.load(os.path.join(camdir, 'test_middle.npy'))
    camera_pos = np.concatenate((camera_pos0, camera_pos1, camera_pos2), 0)
    nframes = camera_pos.shape[0]
    nsamples = 10
    sampled_camera_pos = np.tile(camera_pos, (nsamples, 1))
    camera_sigma = np.array([0.3, 0.3, 1, 0.1, 0.1, 0.1])
    time_sigma = 0.015
    render_t = np.load(os.path.join(camdir, 'test_time.npy'))
    render_t = np.tile(render_t, nsamples)
    for n in range(1, nsamples):
        sampled_camera_pos[nframes*n:nframes*(n+1), :] += np.random.randn(nframes, 6) * camera_sigma
        render_t[nframes*n:nframes*(n+1)] += np.random.rand(nframes) * time_sigma
    numpy.save(os.path.join(dir, 'camera_sampled_test_full_res'), sampled_camera_pos)
    numpy.save(os.path.join(dir, 'camera_sampled_test_time_full_res.npy'), render_t)
    
    return
    
    #camera_pos = numpy.load('/localtmp/yuting/datas_mandelbrot_tile/train.npy')
    #render_t = numpy.load('/localtmp/yuting/datas_mandelbrot_tile/train_time.npy')
    #tile_start = numpy.load('/localtmp/yuting/datas_mandelbrot_tile/train_start.npy')
    
    camera_pos = numpy.load('/n/fs/visualai-scr/yutingy/1x_1sample_mandelbrot_simplified_proxy_temporal/render_view4/camera_pos.npy')
    render_t = numpy.load('/n/fs/visualai-scr/yutingy/1x_1sample_mandelbrot_simplified_proxy_temporal/render_view4/render_t.npy')
    nframes = camera_pos.shape[0]
    
    camera_pos = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbrot_full_temporal_automatic_200_correct_alpha/render/camera_pos.npy')[-1:]
    render_t = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbrot_full_temporal_automatic_200_correct_alpha/render/render_t.npy')[-1:]
    nframes = 1
    render_single('out', 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'mandelbrot_temporal_input', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t})
    return
    
    
    if False:
        camera_pos = numpy.load('/n/fs/shaderml/datas_mandelbrot_temporal_automatic_200/test_far.npy')[2:3]
        render_t = numpy.load('/n/fs/shaderml/datas_mandelbrot_temporal_automatic_200/test_time.npy')[7:8] + 1 / 60
        nframes = 1
        render_single('out', 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'mandelbrot_temporal_start_v3', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'batch_size': 10, 'base_ind': 1})
        return
        
        camera_pos = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbrot_tile_continued/render_fixed_sample/camera_pos.npy')[:2]
        render_t = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbrot_tile_continued/render_fixed_sample/render_t.npy')[:2]
        nframes = 2
        render_single('out', 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'mandelbrot_temporal_render_start', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'batch_size': 10})
                
        camera_pos = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/mandelbrot_render_long_camera_pos.npy')[1:2]
        render_t = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/mandelbrot_render_long_t.npy')[1:2]
        nframes = 1
        render_single('out', 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'mandelbrot_temporal_render_start2', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'batch_size': 10})
        return
    
    camera_pos = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbrot_simplified_proxy_temporal/render/camera_pos.npy')[:1]
    nframes = 2
    camera_pos = np.tile(camera_pos, [nframes, 1])
    camera_pos[:, 3] += 0.3
    camera_pos[:, 4] += 0.3
    camera_pos[:, 5] -= 0.3
    #camera_pos[0, 3] += 0.1
    #camera_pos[1, 3] -= 0.1
    #camera_pos[2, 4] += 0.1
    #camera_pos[3, 4] -= 0.1
    #camera_pos[4, 5] += 0.1
    #camera_pos[5, 5] -= 0.1
    render_t = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbrot_simplified_proxy_temporal/render/render_t.npy')[0]
    render_t = render_t + numpy.arange(nframes) / 60
    render_single('out', 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'temporal_start_v4', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'batch_size': 10})
    return

    #nframes = render_t.shape[0]
    #render_single(os.path.join('out', 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_ground', 'is_tf': True, 'tile_only': True, 'tile_start': tile_start, 'parallel_gpu': 4, 'batch_size': 10})
    camera_pos = numpy.load(os.path.join(camdir, 'train.npy'))
    nframes = camera_pos.shape[0]
    render_t = numpy.load(os.path.join(camdir, 'train_time.npy'))
    
    camera_pos = np.concatenate((
                                np.load(os.path.join(camdir, 'test_close.npy')),
                                np.load(os.path.join(camdir, 'test_far.npy')),
                                np.load(os.path.join(camdir, 'test_middle.npy'))
                                ), axis=0)
    render_t = numpy.load(os.path.join(camdir, 'test_time.npy'))
    nframes = render_t.shape[0]
    
    render_t += 29 / 30
    render_single(os.path.join(dir, 'test_temporal'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'test_ground_29', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'batch_size': 10})
    return
    
    for mode in ['test_close', 'test_far', 'test_middle']:
        camera_pos = numpy.load(os.path.join(camdir, mode + '.npy'))
        render_t = numpy.load(os.path.join(camdir, mode + '_time.npy'))
        for i in range(7):
            render_t += 1 / 30
            render_single(os.path.join(dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': '%s_ground%d' % (mode, i+1), 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'batch_size': 10})
    return
    
    #camera_pos = numpy.load(os.path.join(dir, 'train_raw.npy'))
    #render_t = numpy.load(os.path.join(dir, 'train_time_raw.npy'))
    #nframes = camera_pos.shape[0]
    #render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size=(80, 120), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t, 'automate_loop_statistic': True, 'SELECT_FEATURE_THRE': 200})
    #render_single('out', 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'render_t': render_t})
    
    tile_start = numpy.load(os.path.join(camdir, 'train_start.npy'))
    for i in range(7):
        render_t += 1 / 30
        render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(320, 320), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_ground%d' % (i+1), 'is_tf': True, 'collect_loop_and_features': True, 'efficient_trace': True, 'tile_start': tile_start, 'render_t': render_t, 'tile_only': True, 'batch_size': 10})
    #render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size=(80, 120), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'SELECT_FEATURE_THRE': 12})
    #render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size=(80, 120), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'collect_loop_and_features': True, 'first_n': 5, 'efficient_trace': True})
    #render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (40, 60), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'efficient_trace': True, 'partial_trace': 0.5})
    return

    #dir = '/localtmp/yuting/mandelbrot_tile_radius_wider_range'
    dir = '/n/fs/shaderml/global_opt/proj/apps/out/mandelbrot'
    #for mode in ['train', 'test_close', 'test_far', 'test_middle']:
    #for mode in ['train', 'test_far', 'test_middle']:
    if False:
        x_min = -1200.0 + 100.0
        x_max = 1200.0 + 100.0
        y_min = -1200.0 + 800.0
        y_max = 1200.0 + 800.0
        if mode == 'train':
            nframes = 200
            z_min = 80
            z_max = 1200
        elif mode == 'test_close':
            nframes = 5
            z_min = 20
            z_max = 55
        elif mode == 'test_far':
            nframes = 5
            z_min = 1800
            z_max =2400
        else:
            nframes = 20
            z_min = 80
            z_max = 1200
        if 'top_view' not in dir:
            z_min /= 2.0
            z_max /= 2.0
        camera_pos = [None] * nframes
        render_single(os.path.join('out', mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'z_min': z_min, 'z_max': z_max, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'camera_pos': camera_pos, 'is_tf': True, 'top_view_only': False})
        numpy.save(os.path.join(dir, mode), camera_pos)

    #for mode in ['train', 'test_close', 'test_far', 'test_middle']:
    for mode in ['train']:
        camera_pos = numpy.load(os.path.join(dir, mode) + '.npy').tolist()
        nframes = len(camera_pos)
        #render_single(os.path.join(dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size=(160, 240), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_small', 'is_tf': True, 'collect_loop_and_features': True, 'mean_var_only': True})
        render_single(os.path.join(dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size=(160, 240), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_small', 'is_tf': True, 'partial_trace': 0.5})
        #render_single(os.path.join(dir, mode), 'render_mandelbrot_tile_radius', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size=(640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_small', 'is_tf': True, 'collect_loop_and_features': True, 'subsample_loops': 4})


if __name__ == '__main__':
    main()
