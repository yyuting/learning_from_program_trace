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
    
    dir = '/shaderml/playground/1x_1sample_mandelbulb'
    
    camera_pos = np.load('/mnt/shadermlnfs1/shadermlvm/playground/models/out_videos/mandelbulb_camera_pos.npy')
    render_t = np.load('/mnt/shadermlnfs1/shadermlvm/playground/models/out_videos/mandelbulb_render_t.npy')
    nframes = camera_pos.shape[0]
    
    render_single(os.path.join(dir, 'video'), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'video_small', 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'log_t_ray': True, 'log_intermediates_level': 2})
    return
    
    render_single(os.path.join(dir, 'video'), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'video_ground', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'batch_size': 2})
    return
    
    if True:
        mode = 'train'

        camera_pos = np.load(os.path.join(dir, '%s.npy' % mode))
        render_t = np.load(os.path.join(dir, '%s_time.npy' % mode))
        tile_start = np.load(os.path.join(dir, '%s_start.npy' % mode))
        nframes = camera_pos.shape[0]

        #render_single(os.path.join(dir, mode), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': '%s_ground' % mode, 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'batch_size': 10, 'tile_only': True, 'tile_start': tile_start})
        
        render_single(os.path.join(dir, mode), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': '%s_small' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'tile_only': True, 'tile_start': tile_start})
        return

        render_single(dir, 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (10, 15), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'robust_simplification': True, 'every_nth': 41})
        return
    
    
    
    # expected camera pos for train:
    # x: -4, 3.5
    # y: -4, 3.5
    # z: -4, 4
    # t: 0, 31.5
    
    if False:
        xyzs = [(0, 1.8, 0),
                (1.8, 0, 0),
                (0, 0, 1.8),
                (1.28, 0, 1.28),
                (1.28, 1.28, 0),
                (0, 1.28, 1.28),
                (1.04, 1.04, 1.04)]

        camera_pos = np.empty([len(xyzs), 6])
        for i in range(len(xyzs)):
            x, y, z = xyzs[i]
            ang1, ang2, ang3 = pos_solver(x, y, z)
            camera_pos[i] = np.array([x, y, z, ang1, ang2, ang3])

        render_t = numpy.zeros(camera_pos.shape[0])
        nframes = render_t.shape[0]

        render_single(dir, 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'test', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
        return
    
    if True:
        
        x_min = -4
        x_max = 4
        y_min = -4
        y_max = 4
        z_min = -4
        z_max = 4
        
        for mode in ['test_close', 'test_far', 'test_middle']:
        
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
            render_t = numpy.random.rand(nframes) * 31.5

            for i in range(nframes):
                while True:
                    x = numpy.random.rand() * (x_max - x_min) + x_min
                    y = numpy.random.rand() * (y_max - y_min) + y_min
                    z = numpy.random.rand() * (z_max - z_min) + z_min
                    if (x ** 2 + y ** 2 + z ** 2) > 1.8 ** 2:
                        break
                ang1, ang2, ang3 = pos_solver(x, y, z)
                camera_pos[i] = np.array([x, y, z, ang1, ang2, ang3])

            numpy.save(os.path.join(dir, '%s.npy' % mode), camera_pos)
            numpy.save(os.path.join(dir, '%s_time.npy' % mode), render_t)
        
        
        
            render_single(os.path.join(dir, mode), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': '%s_ground' % mode, 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'batch_size': 2})
        return
    
    camera_pos = numpy.load(os.path.join(dir, 'train.npy'))
    render_t = numpy.load(os.path.join(dir, 'train_time.npy'))
    nframes = render_t.shape[0]
    
    render_single(os.path.join(dir, 'train'), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'fat_comp', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
    return
    
    render_single(os.path.join(dir, 'train'), 'render_mandelbulb_slim', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_noisy_expanded', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
    return
    
    cam_dir = '/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbulb_tile_more_trace_sigma_03_continued/render_fixed_sample'
    
    camera_pos = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbulb_simplified_automatic_200/render_ours/camera_pos.npy')
    render_t = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbulb_simplified_automatic_200/render_ours/render_t.npy')
    nframes = camera_pos.shape[0]
    render_single('out', 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'video_gt', 'is_tf': True, 'log_intermediates_level': 1, 'fov': 'small', 'batch_size': 10})
    return
    
    if False:
        camera_pos = np.load('/n/fs/shaderml/datas_mandelbulb_full_temporal/test_far.npy')[2:3]
        render_t = np.load('/n/fs/shaderml/datas_mandelbulb_full_temporal/test_time.npy')[7:8] + 29 / 30
        nframes = 1
        render_single('out', 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'mandelbulb_full_temporal_input', 'is_tf': True, 'log_intermediates_level': 1})
        
    camera_pos = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbulb_tile_more_trace_sigma_03_continued/render_fixed_sample/camera_pos.npy')[29:30]
    render_t = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbulb_tile_more_trace_sigma_03_continued/render_fixed_sample/render_t.npy')[29:30]
    nframes = 1
    render_single('out', 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'mandelbulb_simplified_temporal_gt', 'is_tf': True, 'log_intermediates_level': 1, 'batch_size': 10, 'fov': 'small'})
    return
    
    #camera_pos = numpy.load(os.path.join(cam_dir, 'camera_pos.npy'))[:2]
    #render_t = numpy.load(os.path.join(cam_dir, 'render_t.npy'))[:2]
    #nframes = 2
    #render_single(dir, 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'render_ground', 'is_tf': True, 'log_intermediates_level': 1, 'fov': 'small'})
    
    #return

    if True:
        dir = '/n/fs/shaderml/1x_1sample_mandelbulb_tile'
        cam_dir = '/n/fs/shaderml/datas_mandelbulb_tile_automate_def_return_only'
        camera_pos = np.concatenate((
                                np.load(os.path.join(cam_dir, 'test_close.npy')),
                                np.load(os.path.join(cam_dir, 'test_far.npy')),
                                np.load(os.path.join(cam_dir, 'test_middle.npy'))
                                ), axis=0)
        render_t = numpy.load(os.path.join(cam_dir, 'test_time.npy'))
        nframes = render_t.shape[0]
        render_t += 29 / 30
        render_single(os.path.join(dir, 'test_temporal'), 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'test_ground_29', 'is_tf': True, 'log_intermediates_level': 1})
        return
        
        #for mode in ['train', 'test_close', 'test_far', 'test_middle']:
        for mode in ['test_close', 'test_far', 'test_middle']:
        #for mode in ['train']:
            camera_pos = numpy.load(os.path.join(cam_dir, mode) + '.npy').tolist()
            render_t = numpy.load(os.path.join(cam_dir, mode) + '_time.npy')
            nframes = len(camera_pos)
            if mode == 'train':
                tile_start = numpy.load(os.path.join(cam_dir, mode) + '_start.npy')
                for i in range(7):
                    render_t += 1 / 30
                    render_single(os.path.join(dir, mode), 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_ground' + str(i+1), 'is_tf': True, 'log_intermediates_level': 1, 'tile_only': True, 'tile_start': tile_start})
            else:
                for i in range(7):
                    render_t += 1 / 30
                    render_single(os.path.join(dir, mode), 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_ground' + str(i+1), 'is_tf': True, 'log_intermediates_level': 1})
            #if mode in ['test_close', 'test_far']:
            #    nframes = 5
            #else:
            #    nframes = 20
            #camera_pos = camera_pos[:nframes]
            #render_t = render_t[:nframes]
            #numpy.save(os.path.join(dir, mode) + '.npy', camera_pos)
            #numpy.save(os.path.join(dir, mode) + '_time.npy', render_t)
            #render_t = numpy.load(os.path.join(dir, mode) + '_time.npy')
            #render_t = numpy.linspace(0.0, 2.0 * numpy.pi, nframes)
            #numpy.save(os.path.join(dir, mode) + '_time.npy', render_t)
            #render_single(os.path.join('out', mode), 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_ground', 'is_tf': True, 'log_intermediates_level': 1})
            #render_single(os.path.join(dir, mode), 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (320, 320), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_ground', 'is_tf': True, 'log_intermediates_level': 1, 'tile_only': True, 'tile_start': tile_start, 'parallel_gpu': 4, 'batch_size': 10})
            
            #render_single(os.path.join(dir, mode), 'render_mandelbulb', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size=(80, 120), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': mode + '_ground', 'is_tf': True, 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True}
        return

if __name__ == '__main__':
    main()
