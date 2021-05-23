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

def main():
    
    camdir = '/n/fs/shaderml/1x_1sample_oceanic_simple_apr20'
    #dir = '/n/fs/visualai-scr/yutingy/oceanic_all_trace'
    dir = camdir
    
    camera_pos = np.load(os.path.join(camdir, 'train.npy'))
    render_t = np.load(os.path.join(camdir, 'train_time.npy'))
    nframes = camera_pos.shape[0]
    
    render_single(os.path.join(dir, 'train'), 'render_oceanic_simple_all_raymarching', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_noisy_expanded', 'robust_simplification': True, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
    return
    
    render_single(dir, 'render_oceanic_simple_all_raymarching', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (10, 15), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'robust_simplification': True, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
    return
    
    
    dir = '/shaderml/playground/1x_1sample_oceanic'
    
    mode = 'test_far'
        
    camera_pos = numpy.load(os.path.join(dir, '%s.npy' % mode))[1:2]
    render_t = numpy.load(os.path.join(dir, '%s_time.npy' % mode))[1:2]

    nframes = render_t.shape[0]

    render_kw = {'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'gname': '%s_small' % mode, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True}

    if mode in ['train', 'validate']:
        tile_start = numpy.load(os.path.join(dir, '%s_start.npy' % mode))
        render_kw['tile_only'] = True
        render_kw['tile_start'] = tile_start
        render_size = (80, 80)
    else:
        render_size = (640, 960)
        
    render_single(os.path.join(dir, mode), 'render_oceanic_simple_all_raymarching', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = render_size, render_kw=render_kw)
    return
    
    camera_pos = numpy.load(os.path.join(dir, 'train.npy'))
    render_t = numpy.load(os.path.join(dir, 'train_time.npy'))
    nframes = render_t.shape[0]
    
    render_single(os.path.join(dir, 'train'), 'render_oceanic_simple_all_raymarching', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_noisy_expanded', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True, 'expand_boundary': 160})
    return
    #render_single(os.path.join(dir, 'train'), 'render_oceanic_simple_all_raymarching', 'none', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 120), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'is_tf': True, 'zero_samples': False, 'gname': 'train_small', 'collect_loop_and_features': True, 'automate_loop_statistic': True, 'log_only_return_def_raymarching': True})

if __name__ == '__main__':
    main()
    
