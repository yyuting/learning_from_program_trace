from render_util import *
from render_single import *
import numpy
import skimage
import skimage.io

def mb(p, time):

    z = [p[0], p[1], p[2]]
    dr = 1.0
    t0 = 1.0

    cond = True

    power = 20.0

    for i in range(4):
        r = sqrt(z[0] ** 2.0 + z[1] ** 2.0 + z[2] ** 2.0)
        #cond *= r <= 2.0
        #cond = select(r <= 2.0, cond, False)
        cond = r <= 2.0
        theta = atan(z[1] / z[0]) * power
        phi = (asin(z[2] / r) + time * 0.1) * power

        #dr = select(cond, (r ** (power - 1.0)) * dr * power + 1.0, dr)

        #r = select(cond, r ** power, r)

        this_power = select(cond, power, 1.0)
        new_dr = (r ** (this_power - 1.0)) * dr * power + 1.0
        dr = select(cond, new_dr, dr)
        r = select(cond, r ** this_power, r)

        cos_phi = cos(phi)

        z[0] = select(cond, r * cos(theta) * cos_phi + p[0], z[0])
        z[1] = select(cond, r * sin(theta) * cos_phi + p[1], z[1])
        z[2] = select(cond, r * sin(phi) + p[2], z[2])

        t0 = select(cond, min_nosmooth(t0, r), t0)

    return [0.5 * log(r) * r / dr, t0]

def f(p, time):
    new_p = rotation_y(p, time * 0.2)
    return mb(new_p, time)

def intersect(ro, rd, time, orig_t):
    t = orig_t
    res_t = ConstExpr(0.0)
    res_c1 = ConstExpr(0.0)
    max_error = ConstExpr(1000.0)
    d = ConstExpr(1.0)
    pd = ConstExpr(100.0)
    os = ConstExpr(0.0)
    step = ConstExpr(0.0)
    error = ConstExpr(1000.0)
    cond1 = True
    c = [ConstExpr(0.0), ConstExpr(0.0)]
    for i in loop_generator(48, is_raymarching=True):
        compiler.DEFAULT_FOR_LOOP_ITER = i
        #cond1 *= (error >= 0.0) * (t <= 20.0)
        cond1 = (error >= 0.0) * (t <= 20.0)

        c = f(ro + rd * t, time)
        d = select(cond1, c[0], d)

        cond2 = d > os
        os = select(cond2, 0.4 * d * d / pd, 0.0)
        step = select(cond2, d + os, -os)
        pd = select(cond2, d, 100.0)
        d = select(cond2, d, 1.0)

        error = select(cond1, d / t, error)

        cond3 = cond1 * (error < max_error)

        max_error = select(cond3, error, max_error)
        res_t = select(cond3, t, res_t)
        res_c1 = select(cond3, c[1], res_c1)

        t = select(cond1, t + step, t)

    #compiler.DEFAULT_FOR_LOOP_NAME = None
    #compiler.DEFAULT_FOR_LOOP_ITER = None
    ro_len = sqrt(ro[0] ** 2 + ro[1] ** 2 + ro[2] ** 2)
    res_t = select(t > ro_len, -1.0, res_t)
    #res_t = select(t > 2.0, -1.0, res_t)
    #res_t = Var('res_t', select(t <= 1.0, -10.0, res_t))
    return [res_t, res_c1]

def mandelbulb_slim(ray_dir_p, ray_origin, time):

    sundir = numpy.array([0.1, 0.8, 0.6])
    sundir /= numpy.linalg.norm(sundir)

    sun = numpy.array([1.64, 1.27, 0.99])
    skycolor = numpy.array([0.6, 1.5, 1.0])

    ray_origin = numpy.array(ray_origin)
    ray_dir_p = numpy.array(ray_dir_p)

    orig_t = (ray_origin[0] ** 2.0 + ray_origin[1] ** 2.0 + ray_origin[2] ** 2.0) ** 0.5 / 3.0
    
    res = intersect(ray_origin, ray_dir_p, time, orig_t)

    t_ray = Var(log_prefix + 't_ray', res[0])
    t_ray.log_intermediates_rank = 2

    cond = t_ray > 0.0
    p = ray_origin + res[0] * ray_dir_p
        
    n = normal_functor(lambda x: f(x, time)[0], 0.001, 3)(p)

    # change log_intermediates_rank for input arguments
    old_log_intermediates_rank = compiler.log_intermediates_rank
    compiler.log_intermediates_rank = 1

    for list in [ray_dir_p, ray_origin, [time], [res[0]], n]:
        for item in list:
            item.log_intermediates_rank = compiler.log_intermediates_rank

    dif = max_nosmooth(0.0, n[0] * sundir[0] + n[1] * sundir[1] + n[2] * sundir[2])
    sky = 0.6 + 0.4 * max_nosmooth(0.0, n[1])
    bac = max_nosmooth(0.0, 0.3 + 0.7 * (-n[0] * sundir[0] - n[1] - n[2] * sundir[2]))
    
    

    lin_coef_a = 4.5 * dif + 0.8 * bac
    lin_coef_b = 0.6 * sky
    lin0 = sun[0] * lin_coef_a + skycolor[0] * lin_coef_b
    lin1 = sun[1] * lin_coef_a + skycolor[1] * lin_coef_b
    lin2 = sun[2] * lin_coef_a + skycolor[2] * lin_coef_b

    tc0_coef = 3.0 + 4.2 * (res[1] ** 0.55)
    col0 = lin0 * 0.9 * 0.2 * (0.5 + 0.5 * sin(tc0_coef))
    col1 = lin1 * 0.8 * 0.2 * (0.5 + 0.5 * sin(tc0_coef + 0.5))
    col2 = lin2 * 0.6 * 0.2 * (0.5 + 0.5 * sin(tc0_coef + 1.0))

    col0 = select(cond, col0 ** 0.45, 0.0)
    col1 = select(cond, col1 ** 0.45, 0.0)
    col2 = select(cond, col2 ** 0.45, 0.0)

    col = numpy.array([col0, col1, col2])
    col = col * 0.6 + 0.4 * col * col * (3.0 - 2.0 * col)
    col = col * 1.5 - 0.5 * 0.33 * (col[0] + col[1] + col[2])

    #col = select(res[0] <= -2.0, numpy.array([1.0, 1.0, 1.0]), col)

    compiler.log_intermediates_rank = old_log_intermediates_rank

    for expr in col.tolist() + n.tolist() + [t_ray]:
        expr.log_intermediates_subset_rank = 1

    return output_color(col)

shaders = [mandelbulb_slim]
is_color = True
# use a different rotation parameterization so can easily compute direction to world coord origin
fov = 'small_seperable'

x_center = 0.0
y_center = 0.0
z_center = 0.0
offset = np.array([0.4, 0.4, 0.4])

def pos_solver(x0, x1, x2):
    """
    given x (length 3) as camera position,
    solve a camera direction that satisfies:
    the center of the image points to the point (0.0, 0.4, 0.0) plus some noise,
    the actual center is (0.0, 0.4, 0.0) + (random(3) * 2.0 - 1.0) * (0.2, 0.2, 0.07)
    the horizonal axis in image is perpendicular to the upward (y axis) in world,
    the vertical axis upward in image is in the same direction of the upward y axis in world.
    """
    random_offset = (np.random.rand(3) * 2.0 - 1.0) * offset
    a = x_center - x0 + random_offset[0]
    b = y_center - x1 + random_offset[1]
    c = z_center - x2 + random_offset[2]
    norm = (a ** 2 + b ** 2 + c ** 2) ** 0.5
    d = a / norm
    e = b / norm
    f = c / norm
    
    ang1 = np.random.rand() * 2 * np.pi
    
    de_norm = (d ** 2 + e ** 2) ** 0.5
    if de_norm > 0:
        # assume cos2 > 0
        ang3 = math.atan2(e / de_norm, d / de_norm)
        
        cos3 = np.cos(ang3)
        if cos3 != 0:
            ang2 = math.atan2(-f, d / cos3)
        else:
            sin3 = np.sin(ang3)
            ang2 = math.atan2(-f, e / sin3)
    else:
        if f > 0:
            ang2 = - np.pi / 2
        else:
            ang2 = np.pi / 2
        ang3 = np.random.rand() * 2 * np.pi

    return ang1, ang2, ang3

def main():
    
    if len(sys.argv) < 3:
        print('Usage: python render_[shader].py base_mode base_dir')
        raise
        
    base_mode = sys.argv[1]
    base_dir = sys.argv[2]
    
    camera_dir = os.path.join(base_dir, 'datasets/datas_mandelbulb_with_bg')
    preprocess_dir = os.path.join(base_dir, 'preprocess/mandelbulb')
    
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir, exist_ok=True)
    
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir, exist_ok=True)
    
    if base_mode == 'collect_raw':
        
        camera_pos = numpy.load(os.path.join(camera_dir, 'train.npy'))
        render_t = numpy.load(os.path.join(camera_dir, 'train_time.npy'))
        nframes = render_t.shape[0]
        
        train_start = numpy.load(os.path.join(camera_dir, 'train_start.npy'))
        render_single(os.path.join(preprocess_dir, 'train'), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
        
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
            
                    
            outdir = get_shader_dirname(os.path.join(preprocess_dir, mode), shaders[0], 'none', 'none')
                
            render_single(os.path.join(preprocess_dir, mode), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground' % mode, 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
            
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
            
            x_min = -4
            x_max = 4
            y_min = -4
            y_max = 4
            z_min = -4
            z_max = 4
            
            if mode == 'train':
                nframes = 800
                x_max = 3.5
                y_max = 3.5
            elif mode == 'validate':
                nframes = 80
                x_max = 3.5
                y_max = 3.5
            elif mode == 'test_close':
                nframes = 5
                x_min = 3.5
            elif mode == 'test_far':
                nframes = 5
                y_min = 3.5
            elif mode == 'test_middle':
                nframes = 20
                x_max = 3.5
                y_max = 3.5

            camera_pos = numpy.empty([nframes, 6])

            for i in range(nframes):
                while True:
                    x = numpy.random.rand() * (x_max - x_min) + x_min
                    y = numpy.random.rand() * (y_max - y_min) + y_min
                    z = numpy.random.rand() * (z_max - z_min) + z_min
                    if (x ** 2 + y ** 2 + z ** 2) > 1.8 ** 2:
                        break
                ang1, ang2, ang3 = pos_solver(x, y, z)
                camera_pos[i] = np.array([x, y, z, ang1, ang2, ang3])

            numpy.save(os.path.join(preprocess_dir, '%s.npy' % mode), camera_pos)
            
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
                    
            render_single(os.path.join(preprocess_dir, mode), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_noisy' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'expand_boundary': expand_boundary})
            
    elif base_mode == 'generate_temporal_dataset':
        
        camera_dir = os.path.join(base_dir, 'datasets/datas_mandelbulb_temporal_with_bg')
        preprocess_dir = os.path.join(base_dir, 'preprocess/mandelbulb_temporal')
        
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
            outdir = get_shader_dirname(os.path.join(preprocess_dir, mode), shaders[0], 'none', 'none')
            
            for t_val in t_schedule:
                render_t = render_t_base + t_val / 30

                render_single(os.path.join(preprocess_dir, mode), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground_%d' % (mode, t_val), 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
                
            target_dir = os.path.join(camera_dir, '%s_img' % mode)
                
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            for file in os.listdir(outdir):
                if file.startswith('%s_ground' % mode) and file.endswith('.png'):
                    os.rename(os.path.join(outdir, file),
                              os.path.join(target_dir, file))
                    
    elif base_mode == 'generate_blur_additional':
        
        preprocess_dir = os.path.join(base_dir, 'preprocess/mandelbulb_blur')
        
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
            
            render_single(os.path.join(preprocess_dir, mode), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_noisy' % mode, 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'log_t_ray': True, 'log_intermediates_level': 2})
            
    return

if __name__ == '__main__':
    main()
