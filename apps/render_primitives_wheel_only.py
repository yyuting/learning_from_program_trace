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
    
    if len(sys.argv) < 3:
        print('Usage: python render_[shader].py base_mode base_dir')
        raise
        
    base_mode = sys.argv[1]
    base_dir = sys.argv[2]
    
    camera_dir = os.path.join(base_dir, 'datasets/datas_primitives_correct_test_range')
    preprocess_dir = os.path.join(base_dir, 'preprocess/gear')
    
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir, exist_ok=True)
    
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir, exist_ok=True)
    
    if base_mode == 'collect_raw':
        
        camera_pos = numpy.load(os.path.join(camera_dir, 'train.npy'))
        render_t = numpy.load(os.path.join(camera_dir, 'train_time.npy'))
        nframes = render_t.shape[0]
        
        train_start = numpy.load(os.path.join(camera_dir, 'train_start.npy'))
        render_single(os.path.join(preprocess_dir, 'train'), 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
        
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
                
            render_single(os.path.join(preprocess_dir, mode), 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground' % mode, 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
            
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
        
        t_range = 1
        
        for mode in ['train', 'test_close', 'test_far', 'test_middle', 'validate']:
            
            x_min = -1.0 + x_center
            x_max = 1.0 + x_center
            y_min = 0.8
            y_max = 1.5
            z_min = -1.0 + z_center
            z_max = 1.0 + z_center

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
                    
            render_single(os.path.join(preprocess_dir, mode), 'render_primitives_wheel_only', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_noisy' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'expand_boundary': expand_boundary})
        
    return

if __name__ == '__main__':
    main()
