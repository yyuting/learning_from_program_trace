from render_util import *
from render_single import *
import numpy
import math
import shutil

# adapted from https://www.shadertoy.com/view/Xds3zN


def sdPlane(p):
    return p[1]

def sdTorus82Rotate(p, t):
    q0 = length(p[[0, 1]], 2) - t[0]
    q1 = p[2]
    return length(numpy.array([q0, q1]), 8) - t[1]

def sdCylinderRotate(p, h):
    d0 = abs(length(p[[0, 1]], 2)) - h[0]
    d1 = abs(p[2]) - h[1]
    return min(max(d0, d1), 0.0) + length(numpy.array([max(d0, 0.0), max(d1, 0.0)]), 2)

def opU(d1, d2):
    cond = d1[0] < d2[0]
    ans0 = select(cond, d1[0], d2[0])
    ans1 = select(cond, d1[1], d2[1])
    return [ans0, ans1]

def opS(d1, d2):
    return max(-d2, d1)

def opRep(p, c):
    return numpy.array([p[0] % c[0], p[1] % c[1], p[2] % c[2]]) - 0.5 * c


def map_primitives(pos):

    res = opU([sdPlane(pos), 1.0],
              [opS(sdTorus82Rotate(pos - numpy.array([0.0, 0.4, 0.0]), numpy.array([0.2, 0.1])),
                   sdCylinderRotate(opRep(numpy.array([atan2_direct(pos[0], pos[1] - 0.4) / (2.0 * numpy.pi / 20.0), 0.02 + 0.5 * length(pos - numpy.array([0.0, 0.4, 0.0]), 2), pos[2]]), numpy.array([0.05, 0.05, 1.0])), numpy.array([0.02, 0.6]))), 78.0])

    return res

def castRay(ro, rd):
    tmin = ConstExpr(0.0)
    tmax = ConstExpr(20.0)

    tp1 = (0.0 - ro[1]) / rd[1]
    tmax = select(tp1 > 0.0, min(tmax, tp1), tmax)
    tp2 = (0.8 - ro[1]) / rd[1]
    cond_roy = ro[1] > 0.8
    tmin = select(tp2 > 0.0, select(cond_roy, max(tmin, tp2), tmin), tmin)
    tmax = select(tp2 > 0.0, select(cond_roy, tmax, min(tmax, tp2)), tmax)
    t = tmin
    m = ConstExpr(-1.0)
    for i in loop_generator(128, is_raymarching=True):
        compiler.DEFAULT_FOR_LOOP_ITER = i
        precis = 0.0004 * t
        res = map_primitives(ro + rd * t)
        update_cond = (res[0] >= precis) * (t <= tmax)
        t = select(update_cond, t + res[0], t)
        m = select(update_cond, res[1], m)

    m = select(t > tmax, -1.0, m)
    return t, m

def calcSoftshadow(ro, rd, mint, tmax):
    res = ConstExpr(1.0)
    t = mint
    if isinstance(t, (int, float)):
        t = ConstExpr(t)
    for i in loop_generator(16, is_raymarching=True):
        cond_update = (res >= 0.005) * (t <= tmax)
        h = map_primitives(ro + rd * t)[0]
        res = select(cond_update, min(res, 8.0 * h / t), res)
        t = select(cond_update, t + max(min(h, 0.1), 0.02), t)
    return clip_0_1(res)

def calcAO(pos, nor):
    
    occ = ConstExpr(0.0)
    sca = ConstExpr(1.0)
    for i in loop_generator(5, is_raymarching=True):
        hr = 0.01 + 0.12 * float(i) / 4.0
        aopos = nor * hr + pos
        dd = map_primitives(aopos)[0]
        occ = occ -(dd - hr) * sca
        sca *= 0.95
    return clip_0_1(1.0 - 3.0 * occ)

def checkersGradBox(p0, p1):
    # using an aliased version of checkerboard
    s0 = sign(fract(p0 * 0.5) - 0.5)
    s1 = sign(fract(p1 * 0.5) - 0.5)
    return 0.5 - 0.5 * s0 * s1

def render(ray_origin, ray_dir_p):
    col_sky = numpy.array([0.7, 0.9, 1.0]) + ray_dir_p[1] * 0.8
    t, m = castRay(ray_origin, ray_dir_p)

    m.log_intermediates_rank = 1

    #primitives_t = Var(log_prefix + 't_ray', select(m > 1.5, t, -1.0))
    #return numpy.array([primitives_t, primitives_t, primitives_t])
    # until this point: 334 normal log_intermediates and 501 loop statistic
    cond_intersect_anything = m > -0.5

    pos = ray_origin + t * ray_dir_p
    
    nor = normal_functor(lambda x: map_primitives(x)[0], 0.5773 * 0.0005, 3)(pos)
        
    ref = ray_dir_p - 2.0 * dot(nor, ray_dir_p) * nor
    col_material0 = 0.45 + 0.35 * sin((m - 1.0) * 0.05)
    col_material1 = 0.45 + 0.35 * sin((m - 1.0) * 0.08)
    col_material2 = 0.45 + 0.35 * sin((m - 1.0) * 0.10)
    col_material = numpy.array([col_material0, col_material1, col_material2])
    cond_intersect_plane = m < 1.5

    f = checkersGradBox(5.0 * pos[0], 5.0 * pos[2])
    col_checker = 0.3 + f * 0.1

    col_item0 = select(cond_intersect_plane, col_checker, col_material0)
    col_item1 = select(cond_intersect_plane, col_checker, col_material1)
    col_item2 = select(cond_intersect_plane, col_checker, col_material2)
    col_item = numpy.array([col_item0, col_item1, col_item2])

    occ = calcAO(pos, nor)
    lig = normalize_const(numpy.array([-0.4, 0.7, -0.6]))

    hal = normalize(lig - ray_dir_p)

    amb = clip_0_1(0.5 + 0.5 * nor[1])
    dif = clip_0_1(dot(nor, lig))
    bac = clip_0_1(dot(nor, normalize(numpy.array([-lig[0], 0.0, -lig[2]])))) * clip_0_1(1.0 - pos[1])
    dom = smoothstep(-0.2, 0.2, ref[1])
    fre = clip_0_1(1.0 + dot(nor, ray_dir_p)) ** 2.0

    dif *= calcSoftshadow(pos, lig, 0.02, 2.5)
    dom *= calcSoftshadow(pos, ref, 0.02, 2.5)

    spe = clip_0_1(dot(nor, hal)) ** 16.0 * dif * (0.04 + 0.96 * clip_0_1(1.0 + dot(hal, ray_dir_p)) ** 5.0)

    lin = 1.30 * dif * numpy.array([1.0, 0.8, 0.55])
    lin += 0.4 * amb * numpy.array([0.4, 0.6, 1.0]) * occ
    lin += 0.4 * dom * numpy.array([0.4, 0.6, 1.0]) * occ
    lin += 0.5 * bac * numpy.array([0.25, 0.25, 0.25]) * occ
    lin += 0.25 * fre * numpy.array([1.0, 1.0, 1.0]) * occ

    col_item = col_item * lin
    col_item += 10.0 * spe * numpy.array([1.0, 0.9, 0.7])
    col_item = mix(col_item, numpy.array([0.8, 0.9, 1.0]), 1.0 - exp(-0.0002 * t * t * t))

    #col = select(cond_intersect_anything, col_item, col_sky)
    col0 = clip_0_1(select(cond_intersect_anything, col_item[0], col_sky[0]))
    col1 = clip_0_1(select(cond_intersect_anything, col_item[1], col_sky[1]))
    col2 = clip_0_1(select(cond_intersect_anything, col_item[2], col_sky[2]))

    for expr in [t, dif, spe] + nor.tolist():
        expr.log_intermediates_subset_rank = 1

    return numpy.array([col0, col1, col2])

def primitives_wheel_only(ray_dir_p, ray_origin, time):
    #return output_color(ray_origin)
    col = render(ray_origin, ray_dir_p)
    col = col ** 0.4545
    for expr in col.tolist():
        expr.log_intermediates_subset_rank = 1
    return output_color(col)

shaders = [primitives_wheel_only]
is_color = True
fov = 'small'

x_center = 0.0
y_center = 0.4
z_center = 0.0

def pos_solver(x0, x1, x2):
    """
    given x (length 3) as camera position,
    solve a camera direction that satisfies:
    the center of the image points to the point (0.0, 0.4, 0.0) plus some noise,
    the actual center is (0.0, 0.4, 0.0) + (random(3) * 2.0 - 1.0) * (0.2, 0.2, 0.07)
    the horizonal axis in image is perpendicular to the upward (y axis) in world,
    the vertical axis upward in image is in the same direction of the upward y axis in world.
    """
    random_offset = (np.random.rand(3) * 2.0 - 1.0) * np.array([0.2, 0.2, 0.07])
    a = x_center - x0 + random_offset[0]
    b = y_center - x1 + random_offset[1]
    c = z_center - x2 + random_offset[2]
    norm = (a ** 2 + b ** 2 + c ** 2) ** 0.5
    d = a / norm
    e = b / norm
    f = c / norm

    df_norm = (d ** 2 + f ** 2) ** 0.5
    ang1 = math.atan2(-e, -df_norm)
    ang2 = math.atan2(-d / df_norm, -f / df_norm)
    ang3 = math.atan2(0, 1)
    return ang1, ang2, ang3


def main():
    
    
    dir = '/shaderml/playground/1x_1sample_primitives'
    
    for mode in ['test_close', 'test_far', 'test_middle']:
        camera_pos = np.load(os.path.join(dir, '%s.npy' % mode))
        render_t = np.load(os.path.join(dir, '%s_time.npy' % mode))
        
        nframes = camera_pos.shape[0]
        
        render_single(os.path.join(dir, mode), 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname':'%s_ground' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'batch_size': 2})
        
    return

    camera_pos = numpy.load(os.path.join(dir, 'train.npy'))
    render_t = numpy.load(os.path.join(dir, 'train_time.npy'))
    train_start = numpy.load(os.path.join(dir, 'train_start.npy'))
    nframes = render_t.shape[0]

    render_single(os.path.join(dir, 'train'), 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname':'train_small', 'tile_only': True, 'tile_start': train_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
    return
    
    
    mode = 'validate'
        
    camera_pos = numpy.load(os.path.join(dir, '%s.npy' % mode))
    render_t = numpy.load(os.path.join(dir, '%s_time.npy' % mode))
    #tile_start = numpy.load(os.path.join(dir_cam, 'train_start.npy'))
    nframes = camera_pos.shape[0]

    #render_single(dir, 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (320, 320), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname':'train_small', 'tile_only': True, 'tile_start': tile_start, 'collect_feature_mean_only': True, 'feature_normalize_dir': dir, 'reference_dir': os.path.join(dir_cam, 'train_img'), 'collect_loop_and_features': True})
    #return

    # collect efficient trace for future correlation computation
    #render_single(dir, 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (10, 15), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'collect_loop_and_features': True})

    #render_single(dir, 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (10, 15), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'robust_simplification': True})
    #return

    render_single(os.path.join(dir, mode), 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': '%s_noisy_expanded' % mode, 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
    return

        #render_single(os.path.join(dir, 'train'), 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (40， 60), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname':'train_small', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'SELECT_FEATURE_THRE': 400})

    for mode in ['test_close', 'test_far', 'test_middle']:
        camera_pos = numpy.load(os.path.join(dir_cam, '%s.npy' % mode))
        nframes = camera_pos.shape[0]
        render_t = numpy.load(os.path.join(dir_cam, '%s_time.npy' % mode))
        render_single('out', 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname':'%s_ground' % mode})


        #render_single('out', 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (320, 320), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname':'train_ground', 'parallel_gpu': 4, 'tile_only': True, 'tile_start': tile_start, 'batch_size': 10})
        return

    if True:
        x_min = -1.0 + x_center
        x_max = 1.0 + x_center
        y_min = 0.8
        y_max = 1.5
        z_min = -1.0 + z_center
        z_max = 1.0 + z_center

        # test_close and test_far is not descriptive
        # they only represent novel camera views
        # use this name only to be consistent with other dataset
        for mode in ['validate']:
            if mode == 'train':
                nframes = 800
            elif mode == 'validate':
                nframes = 80
            elif mode == 'test_middle':
                nframes = 20
            else:
                nframes = 5
            if mode == 'test_close':
                x_min = -1.5 + x_center
                x_max = -1.0 + x_center
                z_min = -1.0 + z_center
                z_max = 1.0 + z_center
            elif mode == 'test_far':
                x_min = -1.0 + x_center
                x_max = 1.0 + x_center
                z_min = 1.0 + z_center
                z_max = 1.5 + z_center
            camera_pos = [None] * nframes
            for i in range(nframes):
                while True:
                    camera_x = numpy.random.rand() * (x_max - x_min) + x_min
                    camera_y = numpy.random.rand() * (y_max - y_min) + y_min
                    camera_z = numpy.random.rand() * (z_max - z_min) + z_min
                    if abs(camera_x) > 0.3 and abs(camera_y) > 0.8 and abs(camera_x) > 0.1:
                        break
                rotation1, rotation2, rotation3 = pos_solver(camera_x, camera_y, camera_z)
                camera_pos[i] = numpy.array([camera_x, camera_y, camera_z, rotation1, rotation2, rotation3])
            camera_pos = numpy.array(camera_pos)
            numpy.save(os.path.join(dir, mode + '.npy'), camera_pos)
            numpy.save(os.path.join(dir, '%s_time.npy' % mode), np.zeros(nframes))
            
        return
            
    for mode in ['train', 'test_close', 'test_far', 'test_middle']:
        camera_pos = numpy.load(os.path.join(dir, mode) + '.npy').tolist()
        nframes = len(camera_pos)
        render_single('out', 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': mode + '_small'})
        #render_single(os.path.join(dir, mode), 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 120), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'small', 'collect_loop_and_features': True, 'last_only': True})
        #render_single(os.path.join(dir, mode), 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': mode + '_small', 'log_intermediates_level': 1, 'log_t_ray': True})
    return

    nframes = 100
    extra_camera_pos = [None] * nframes
    if False:
    #for i in range(nframes):
        x_min = -2.0 + x_center
        x_max = 2.0 + x_center
        y_min = 0.0
        y_max = 2.0
        z_min = -2.0 + z_center
        z_max = 2.0 + z_center
        camera_x = numpy.random.rand() * (x_max - x_min) + x_min
        camera_y = numpy.random.rand() * (y_max - y_min) + y_min
        camera_z = numpy.random.rand() * (z_max - z_min) + z_min
        rotation1, rotation2, rotation3 = pos_solver(camera_x, camera_y, camera_z)
        extra_camera_pos[i] = numpy.array([camera_x, camera_y, camera_z, rotation1, rotation2, rotation3])

    #numpy.save(os.path.join(dir, 'camera_pos.npy'), numpy.array(camera_pos))
    camera_pos = numpy.load(os.path.join(dir, 'camera_pos.npy'))
    extra_camera_pos = numpy.load(os.path.join(dir_add, 'camera_pos.npy'))
    #render_single(dir_test, 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'small'})
    #render_single('out', 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False})
    #render_single(dir, 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (160, 240), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'small', 'collect_loop_and_features': True})
    compiler.log_intermediates_less = True
    # uncomment ling 249, 250 to run this command
    #render_single(dir_test, 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (160, 240), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'small'})
    #render_single(dir_add, 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (160, 240), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': extra_camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'small'})
    #numpy.save(os.path.join(dir_add, 'camera_pos.npy'), extra_camera_pos)
    compiler.log_intermediates_less = False
    avg_t = numpy.load(os.path.join(dir_test, 'primitives_aliasing_none_normal_none', 'avg_t.npy'))
    avg_t_sorted = numpy.sort(avg_t)
    ind = numpy.argsort(avg_t)
    lo_ind = int(0.05 * avg_t.shape[0])
    hi_ind = int(0.95 * avg_t.shape[0])
    close_thr = avg_t_sorted[lo_ind]
    far_thr = avg_t_sorted[hi_ind]

    train_lo = close_thr * 1.5
    train_hi = far_thr * 1.5

    avg_t_add = numpy.load(os.path.join(dir_add, 'primitives_aliasing_none_normal_none', 'avg_t.npy'))
    avg_t_ind = numpy.arange(nframes)
    avg_t_valid_ind = avg_t_ind[(avg_t_add >= train_lo) * (avg_t_add <= train_hi)]

    frames_needed = 0
    for mode in ['test_close', 'test_far', 'train', 'test_middle']:
        current_dir = os.path.join(dir, mode, 'primitives_aliasing_none_normal_none')
        if not os.path.isdir(current_dir):
            os.makedirs(current_dir)
        if mode == 'test_close':
            current_inds = [3, 140, 75, 126, 155, 0, 59, 47]
        elif mode == 'test_far':
            current_inds = [43, 52, 171, 156, 115, 73]
        elif mode == 'train':
            current_inds = ind[(avg_t_sorted >= train_lo) * (avg_t_sorted <= train_hi)]
            frames_needed += 200 - len(current_inds)
        else:
            current_inds = []
            frames_needed += 20 - len(current_inds)
        print(len(current_inds))
        print(current_inds)
        if True:
            current_camera_pos = numpy.empty([len(current_inds), 6])
            for i in range(len(current_inds)):
                current_ind = current_inds[i]
                current_camera_pos[i, :] = camera_pos[current_ind, :]
                ground_truth_src = 'out/primitives_aliasing_none_normal_none/ground%05d.png' % current_ind
                ground_truth_dst = os.path.join(current_dir, mode + '_ground%05d.png' %i)
                shutil.copyfile(ground_truth_src, ground_truth_dst)
                small_src = 'out/primitives_aliasing_none_normal_none/small%05d.png' % current_ind
                small_dst = os.path.join(current_dir, mode + '_small%05d.png' % i)
                shutil.copyfile(small_src, small_dst)
                #log_src = os.path.join(dir, 'primitives_aliasing_none_normal_none/g_intermediates%05d.npy' % current_ind)
                #log_dst = os.path.join(current_dir, 'g_intermediates%05d.npy' % i)
                #os.rename(log_src, log_dst)
            numpy.save(os.path.join(dir, '%s.npy' % mode), current_camera_pos)
    print("frames needed:", frames_needed)

    assert len(avg_t_valid_ind) > frames_needed
    print("success")
    camera_pos_valid = extra_camera_pos[avg_t_valid_ind[:frames_needed], :]
    #render_single(dir_add, 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=frames_needed, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos_valid, 'is_tf': True, 'zero_samples': False, 'gname': 'small'})
    #render_single(dir_add, 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=frames_needed, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos_valid, 'is_tf': True, 'zero_samples': False})

    add_frame_count = 0
    for mode in ['train', 'test_middle']:
        current_dir = os.path.join(dir, mode, 'primitives_aliasing_none_normal_none')
        if not os.path.isdir(current_dir):
            os.makedirs(current_dir)
        if mode == 'train':
            current_inds = ind[(avg_t_sorted >= train_lo) * (avg_t_sorted <= train_hi)]
            current_frame_needed = 200 - len(current_inds)
        else:
            current_inds = []
            current_frame_needed = 20 - len(current_inds)
        print(len(current_inds))
        print(current_inds)
        print(current_frame_needed)
        current_base = len(current_inds)
        if True:
            current_camera_pos = numpy.empty([current_frame_needed, 6])
            for i in range(current_frame_needed):
                current_camera_pos[i, :] = camera_pos_valid[i + add_frame_count, :]
                ground_truth_src = os.path.join(dir_add, 'primitives_aliasing_none_normal_none/ground%05d.png' % (i + add_frame_count))
                ground_truth_dst = os.path.join(current_dir, mode + '_ground%05d.png' % (i + current_base))
                shutil.copyfile(ground_truth_src, ground_truth_dst)
                small_src = os.path.join(dir_add, 'primitives_aliasing_none_normal_none/small%05d.png' % (i + add_frame_count))
                small_dst = os.path.join(current_dir, mode + '_small%05d.png' % (i + current_base))
                shutil.copyfile(small_src, small_dst)
                #log_src = os.path.join(dir, 'primitives_aliasing_none_normal_none/g_intermediates%05d.npy' % current_ind)
                #log_dst = os.path.join(current_dir, 'g_intermediates%05d.npy' % i)
                #os.rename(log_src, log_dst)
            add_frame_count += current_frame_needed

            base_camera_file = os.path.join(dir, '%s.npy' % mode)
            if os.path.exists(base_camera_file):
                base_camera_pos = numpy.load(base_camera_file)
                final_camera_pos = numpy.concatenate((base_camera_pos, current_camera_pos), axis=0)
            else:
                final_camera_pos = current_camera_pos

            numpy.save(os.path.join(dir, '%s.npy' % mode), final_camera_pos)

    return
    nframes = 200
    camera_pos = [None] * nframes
    # center: x = -2.75, z = -0.43
    # y range: 0 - 2
    # x, z range from center: -4 - 4
    center_x = -2.75
    center_z = -0.43
    x_min = -4.0
    x_max = 4.0
    y_min = 0.0
    y_max = 2.0
    z_min = -4.0
    z_max = 4.0
    for i in range(nframes):
        rotation1 = numpy.random.rand() * 2.0 * numpy.pi
        rotation2 = numpy.random.rand() * 2.0 * numpy.pi
        rotation3 = numpy.random.rand() * 2.0 * numpy.pi
        camera_x = numpy.random.rand() * (x_max - x_min) + x_min + center_x
        camera_y = numpy.random.rand() * (y_max - y_min) + y_min
        camera_z = numpy.random.rand() * (z_max - z_min) + z_min + center_z
        #while True:
        #    pass
    render_single('out', 'render_primitives_aliasing', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': True})

    # generate random camera pos

if __name__ == '__main__':
    main()
