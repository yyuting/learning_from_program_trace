from render_util import *
from render_single import *
import numpy

def mandelbrot_tile_radius_short_05(position, tex_coords, normal, light_dir, viewer_dir, use_triangle_wave, time):

    c_x = Var('c_x', tex_coords[0] / 500.0 - 1.5)
    c_y = Var('c_y', tex_coords[1] / 500.0 - 1.3)

    z_x = ConstExpr(0.0)
    z_y = ConstExpr(0.0)
    zx2 = z_x * z_x
    zy2 = z_y * z_y
    z2 = zx2 + zy2

    niters = 10
    niters_float = float(20)

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

shaders = [mandelbrot_tile_radius_short_05]
is_color = True
fov = 'small'

def main():
    
    camdir = '/shaderml/playground/datas_mandelbrot'
    dir = '/shaderml/playground/1x_1sample_manelbrot'
    
    camera_pos = np.load(os.path.join(camdir, 'train.npy'))
    render_t = np.load(os.path.join(camdir, 'train_time.npy'))
    tile_start = np.load(os.path.join(camdir, 'train_start.npy'))
    nframes = camera_pos.shape[0]
    
    render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius_short_05', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (80, 80), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'efficient_trace': True, 'tile_only': True, 'tile_start': tile_start, 'collect_loop_and_features': True, 'log_only_return_def_raymarching': True})
    return
    
    #camera_pos = numpy.load('/n/fs/shaderml/datas_mandelbrot_simplified_temporal/test_far.npy')[2:3]
    #render_t = numpy.load('/n/fs/shaderml/datas_mandelbrot_simplified_temporal/test_time.npy')[7:8] + 29 / 30
    camera_pos = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbrot_full_temporal_automatic_200_correct_alpha/render/camera_pos.npy')[-1:]
    render_t = numpy.load('/n/fs/shaderml/FastImageProcessing/CAN24_AN/1x_1sample_mandelbrot_full_temporal_automatic_200_correct_alpha/render/render_t.npy')[-1:]
    nframes = 1
    render_single('out', 'render_mandelbrot_tile_radius_short_05', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'mandelbrot_temporal_input', 'is_tf': True, 'efficient_trace': True})
    return
    
    camera_pos = numpy.load(os.path.join(dir, 'train.npy'))
    nframes = camera_pos.shape[0]
    
    render_single(os.path.join(dir, 'train'), 'render_mandelbrot_tile_radius_short_05', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=True, render_size = (40, 60), render_kw={'render_t': render_t, 'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'train_small', 'is_tf': True, 'efficient_trace': True})

    #camera_pos = numpy.load(os.path.join(dir, 'test_middle.npy'))
    #nframes = camera_pos.shape[0]
    #render_single(os.path.join(dir, 'test_middle'), 'render_mandelbrot_tile_radius_short_05', 'plane', 'none', sys.argv[1:], nframes=nframes, log_intermediates=False, render_size = (640, 960), render_kw={'compute_g': False, 'compute_f': False, 'ground_truth_samples': 1, 'random_camera': True, 'camera_pos': camera_pos, 'gname': 'test_middle_small', 'is_tf': True, 'efficient_trace': True})
    return

if __name__ == '__main__':
    main()
