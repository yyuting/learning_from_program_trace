"""
modified from shadertoy
https://www.shadertoy.com/view/4sXGRM
"""

from render_util import *
from render_single import *

waterlevel = 70
wavegain = 1.0
large_waveheight = 1.0
small_waveheight = 1.0

fogcolor = numpy.array([0.5, 0.7, 1.1])
skybottom = numpy.array([0.6, 0.8, 1.2])
skytop = numpy.array([0.05, 0.2, 0.5])
reflskycolor = numpy.array([0.025, 0.10, 0.20])
watercolor = numpy.array([0.2, 0.25, 0.3])

light = numpy.array([0.10644926, 0.26612315, 0.95804333])

plot_sun = True

fbm_2d = fbm_2d_functor(pseudo_noise_2d, 4)

fbm_3d = fbm_3d_functor(pseudo_noise_3d, 4)

def water(p, time):
    height = waterlevel
    shift1_0 = 0.3 * 0.001 * time * 160 * 2
    shift1_1 = 0.3 * 0.001 * time * 120 * 2
    shift2_0 = 0.001 * time * 190 * 2
    shift2_1 = -0.001 * time * 130 * 2

    wave = 0.0
    wave = wave + sin(p[0] * 0.021   +                  shift2_0        ) * 4.5
    wave = wave + sin(p[0] * 0.0172  + p[1] * 0.010   + shift2_0 * 1.121) * 4.0
    wave = wave - sin(p[0] * 0.00104 + p[1] * 0.005   + shift2_0 * 0.121) * 4.0
    # ...added by some smaller faster waves...
    wave = wave + sin(p[0] * 0.02221 + p[1] * 0.01233 + shift2_0 * 3.437) * 5.0
    wave = wave + sin(p[0] * 0.03112 + p[1] * 0.01122 + shift2_0 * 4.269) * 2.5
    wave = wave * large_waveheight
    wave = wave - fbm_2d(p * 0.004 - numpy.array([shift2_0, shift2_1]) * 0.5) * small_waveheight * 24.0

    amp = 6.0 * small_waveheight
    shift1 = numpy.array([shift1_0, shift1_1])
    
    for i in loop_generator(7, is_raymarching=True):
        wave = wave - abs(sin((pseudo_noise_2d(p * 0.01 + shift1) - 0.5) * numpy.pi)) * amp
        amp *= 0.51
        shift1 = shift1 * 1.841
        new_p0 = 0.9331 * (1.6 * p[0] - 1.2 * p[1])
        new_p1 = 0.9331 * (1.2 * p[0] + 1.6 * p[1])
        p = numpy.array([new_p0, new_p1])

    height = wave + height
    return height
#water = def_generator(water)

def trace_fog(rStart, rDirection, time):
    
    shift0 = time * 80.0
    shift1 = time * 60.0
    sum = ConstExpr(0.0)
    q2 = ConstExpr(0.0)
    q3 = ConstExpr(0.0)
    for q in loop_generator(10, is_raymarching=True):
        c = (q2 + 350.0 - rStart[1]) / rDirection[1]
        cpos = rStart + c * rDirection + numpy.array([831.0, 321.0 + q3 - shift0 * 0.2, 1330.0 + shift1 * 3.0])
        alpha = smoothstep(0.5, 1.0, fbm_3d(cpos * 0.0015))
        sum = sum + (1.0 - sum) * alpha
        update_cond = sum <= 0.98
        q2 = select(update_cond, q2 + 120.0, q2)
        q3 = select(update_cond, q3 + 0.15, q3)
    return clip_0_1(1.0 - sum)

def trace(rStart, rDirection, time):
    h = ConstExpr(20.0)
    old_h = ConstExpr(0.0)
    t = -rStart[1] / rDirection[1]
    st = ConstExpr(0.5)
    alpha = ConstExpr(0.1)
    asum  = ConstExpr(0.0)
    return_val = ConstExpr(False)
    cond_look_up_sky = rDirection[1] > 0.0
    cond_update_st = ConstExpr(False)
    for j in loop_generator(20, is_raymarching=True):
        #st = select(t > 1000.0, 12.0, select(t > 800.0, 5.0, select(t > 500.0, 2.0, st)))
        st = select(t > 500.0, 1.0, st)
        st = select(t > 800.0, 2.0, st)
        st = select(t > 1500.0, 3.0, st)
        p = rStart + t * rDirection
        h = p[1] - water(p[[0, 2]], time)

        t = t + max(1.0, abs(h)) * st * sign(h)
        cond_update_st = select((h * old_h) < 0.0, True, cond_update_st)
        st = select(cond_update_st, st / 2.0, st)
        old_h = h;

    dist = t
    #return_val = select(h < 10, True, return_val)
    return h, dist

def oceanic_simple_all_raymarching(ray_dir_p, ray_origin, time):
    rd = ray_dir_p
    #rd[0] = rd[0] * 1.75
    campos = ray_origin
    sundot = clip_0_1(dot(rd, light))

    h, dist = trace(campos, rd, time)

    traced = (ray_dir_p[1] < 0.0) * (dist < 20000.0)
    fog = ConstExpr(0.0)

    t_sky = pow(1.0 - max(0.7 * rd[1], 0), 15.0)
    col_sky = 0.8 * (skybottom * t_sky + skytop * (1.0 - t_sky))
    if plot_sun:
        col_sky = col_sky + 0.47 * numpy.array([1.6, 1.4, 1.0]) * pow(sundot, 350.0)
    col_sky = col_sky + 0.4 * numpy.array([0.8, 0.9, 1.0]) * pow(sundot, 2.0)

    shift0 = time * 80.0
    shift1 = time * 60.0
    #shift0 = 0.0
    #shift1 = 0.0

    color_sum = numpy.array([ConstExpr(0.0), ConstExpr(0.0), ConstExpr(0.0)])
    color_sum3 = ConstExpr(0.0)

    if True:
        for q in loop_generator(3, is_raymarching=True):
            update_cond = color_sum3 <= 0.98
            c = (q * 12.0 + 350.0 - campos[1]) / rd[1]
            cpos = campos + c * rd + numpy.array([831.0, 321.0 + q * 0.15 - shift0 * 0.2, 1330.0 + shift1 * 3.0])
            alpha = smoothstep(0.5, 1.0, fbm_3d(cpos * 0.0015)) * 0.9
            localcolor = mix(numpy.array([1.1, 1.05, 1.0]), 0.7 * numpy.array([0.4, 0.4, 0.3]), alpha)
            alpha = (1.0 - color_sum3) * alpha
            sum_update = color_sum + localcolor * alpha
            color_sum[0] = select(update_cond, sum_update[0], color_sum[0])
            color_sum[1] = select(update_cond, sum_update[1], color_sum[1])
            color_sum[2] = select(update_cond, sum_update[2], color_sum[2])
            color_sum3 = select(update_cond, color_sum3 + alpha, color_sum3)

        #return output_color(color_sum)
    alpha = smoothstep(0.7, 1.0, color_sum3)
    color_sum = color_sum / (color_sum3 + 0.0001)

    color_sum = color_sum - 0.6 * numpy.array([0.8, 0.75, 0.7]) * pow(sundot, 13.0) * alpha
    color_sum = color_sum + 0.2 * numpy.array([1.3, 1.2, 1.0]) * pow(sundot, 5.0) * (1.0 - alpha)
    col_sky = mix(col_sky, color_sum, color_sum3 * (1.0 - t_sky))

    wpos = campos + dist * rd
    

    if False:
        xdiff = wavegain * 4.0 * numpy.array([0.1, 0.0])
        ydiff = wavegain * 4.0 * numpy.array([0.0, 0.1])
        water_norm0 = water(wpos[[0, 2]] - xdiff, time) - water(wpos[[0, 2]] + xdiff, time)
        water_norm1 = 1.0
        water_norm2 = water(wpos[[0, 2]] - ydiff, time) - water(wpos[[0, 2]] + ydiff, time)
        water_norm = numpy.array([water_norm0, water_norm1, water_norm2]) / sqrt((water_norm0 * water_norm0) + (water_norm1 * water_norm1) + (water_norm2 * water_norm2))
    
    water_norm = normal_functor(lambda x: water(x, time), -wavegain * 0.4, 2, extra_term=[1.0], extra_pos=[1])(wpos[[0, 2]])
    
    rd = rd - 2.0 * dot(water_norm, rd) * water_norm
    refl = 1.0 - clip_0_1(rd[1])

    sh = smoothstep(0.2, 1.0, trace_fog(wpos + 20.0 * rd, rd, time)) * 0.7 + 0.3
    #return output_color([sh, sh, sh])
    wsky = refl * sh
    wwater = (1.0 - refl) * sh

    sundot = clip_0_1(dot(rd, light))

    col_water = wsky * reflskycolor
    col_water = col_water + wwater * watercolor
    col_water = col_water + numpy.array([0.003, 0.005, 0.005]) * (wpos[1] - waterlevel + 30.0)
    #return output_color(col_water)
    #test = wpos[1] - waterlevel + 500.0
    #test = test / 20.0
    #return output_color([test, test, test])

    wsunrefl = wsky * (0.5 * pow(sundot, 10) + 0.25 * pow(sundot, 3.5) + 0.75 * pow(sundot, 300))
    col_water = col_water + wsunrefl * numpy.array([1.5, 1.3, 1.0])

    fo = 1.0 - exp(-pow(0.0003 * dist, 1.5))
    fco = fogcolor + 0.6 * numpy.array([0.6, 0.5, 0.4]) * pow(sundot, 4.0)
    col_water = mix(col_water, fco, fo)

    col0 = select(traced, col_water[0], col_sky[0])
    col1 = select(traced, col_water[1], col_sky[1])
    col2 = select(traced, col_water[2], col_sky[2])

    for expr in [dist, col0, col1, col2] + water_norm.tolist():
        expr.log_intermediates_subset_rank = 1

    return output_color(numpy.array([col0, col1, col2]))


shaders = [oceanic_simple_all_raymarching]
is_color = True
fov = 'small'

def random_pos():
    """
    constrains:
    dot product of ray from image center and verticle world axis should be between -0.3 and 0.0
    dot product of verticle axis in image and verticle axis in world should be smaller than -0.95
    """
    while True:
        ang1 = numpy.random.uniform(0.0, 2.0 * numpy.pi)
        ang2 = numpy.random.uniform(0.0, 2.0 * numpy.pi)
        ang3 = numpy.random.uniform(0.0, 2.0 * numpy.pi)
        ans1 = numpy.cos(ang1) * numpy.cos(ang3) + numpy.sin(ang1) * numpy.sin(ang2) * numpy.sin(ang3)
        ans2 = -numpy.sin(ang1) * numpy.cos(ang3) + numpy.cos(ang1) * numpy.sin(ang2) * numpy.sin(ang3)
        ans3 = light[0] * (numpy.sin(ang1) * numpy.sin(ang3) + numpy.cos(ang1) + numpy.sin(ang2) + numpy.cos(ang3))
        ans4 = light[1] * (-numpy.sin(ang1) * numpy.cos(ang3) + numpy.cos(ang1) * numpy.sin(ang2) * numpy.sin(ang3))
        ans5 = light[2] * (numpy.cos(ang1) * numpy.cos(ang2))
        if ans1 <= -0.95 and ans2 >= -0.3 and ans2 <= 0.0:
            if ans3 + ans4 + ans5 > 0.85:
                break
    assert numpy.cos(ang1) * numpy.cos(ang3) + numpy.sin(ang1) * numpy.sin(ang2) * numpy.sin(ang3) <= -0.95
    assert -numpy.sin(ang1) * numpy.cos(ang3) + numpy.cos(ang1) * numpy.sin(ang2) * numpy.sin(ang3) >= -0.3
    assert -numpy.sin(ang1) * numpy.cos(ang3) + numpy.cos(ang1) * numpy.sin(ang2) * numpy.sin(ang3) <= 0.0
    return ang1, ang2, ang3

def main():
    
    if len(sys.argv) < 3:
        print('Usage: python render_[shader].py base_mode base_dir')
        raise
        
    base_mode = sys.argv[1]
    base_dir = sys.argv[2]
    
    camera_dir = os.path.join(base_dir, 'datasets/datas_oceanic')
    preprocess_dir = os.path.join(base_dir, 'preprocess/oceanic')
    
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir, exist_ok=True)
    
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir, exist_ok=True)
    
    if base_mode == 'collect_raw':
        
        camera_pos = numpy.load(os.path.join(camera_dir, 'train.npy'))
        render_t = numpy.load(os.path.join(camera_dir, 'train_time.npy'))
        nframes = render_t.shape[0]
        
        train_start = numpy.load(os.path.join(camera_dir, 'train_start.npy'))
        render_single(os.path.join(preprocess_dir, 'train'), 'render_oceanic_simple_all_raymarching', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': 'train_small', 'tile_only': True, 'tile_start': train_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
        
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
                
            render_single(os.path.join(preprocess_dir, mode), 'render_oceanic_simple_all_raymarching', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = render_size, render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1000, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_ground' % mode, 'tile_only': tile_only, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
            
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
        
        t_range = 100
        
        for mode in ['train', 'test_close', 'test_far', 'test_middle', 'validate']:
            
            x_min = -100.0
            x_max = 100.0
            z_min = -100.0
            z_max = 100.0
            
            if mode == 'train':
                nframes = 800
                y_min = 100
                y_max = 415
            elif mode == 'validate':
                nframes = 80
                y_min = 100
                y_max = 415
            elif mode == 'test_middle':
                nframes = 20
                y_min = 100
                y_max = 415
            elif mode == 'test_close':
                nframes = 5
                y_min = 85
                y_max = 100
            else:
                nframes = 5
                y_min = 415
                y_max = 600
                
            camera_pos = [None] * nframes

            for i in range(nframes):
                ang1, ang2, ang3 = random_pos()
                assert numpy.cos(ang1) * numpy.cos(ang3) + numpy.sin(ang1) * numpy.sin(ang2) * numpy.sin(ang3) <= -0.95
                assert -numpy.sin(ang1) * numpy.cos(ang3) + numpy.cos(ang1) * numpy.sin(ang2) * numpy.sin(ang3) >= -0.3
                assert -numpy.sin(ang1) * numpy.cos(ang3) + numpy.cos(ang1) * numpy.sin(ang2) * numpy.sin(ang3) <= 0.0
                print(numpy.cos(ang1) * numpy.cos(ang3) + numpy.sin(ang1) * numpy.sin(ang2) * numpy.sin(ang3), -numpy.sin(ang1) * numpy.cos(ang3) + numpy.cos(ang1) * numpy.sin(ang2) * numpy.sin(ang3))
                pos_x = numpy.random.uniform(x_min, x_max)
                pos_y = numpy.random.uniform(y_min, y_max)
                pos_z = numpy.random.uniform(z_min, z_max)
                camera_pos[i] = [pos_x, pos_y, pos_z, ang1, ang2, ang3]
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
                    
            render_single(os.path.join(preprocess_dir, mode), 'render_oceanic_simple_all_raymarching', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'zero_samples': False, 'gname': '%s_noisy' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'expand_boundary': expand_boundary})
        
    return

if __name__ == '__main__':
    main()
    
